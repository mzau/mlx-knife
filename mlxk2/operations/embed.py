"""``mlxk embed`` operation (ADR-015, Slice A: decoder vertical).

Generates embeddings via :class:`EmbeddingRunner` and emits JSONL — the default machine
format (ADR-015 §Output): the typical consumer is another program (cosine-search, index
builder), not a human at a terminal.

Detection/routing here is intentionally minimal and local to this module; the config-first
capability rewrite + gate [5] live in Slice C. Slice A runs the decoder path only and
surfaces encoder embedders honestly as not-yet-runnable.
"""

import json
import sys
from typing import Any, Dict, List, Optional

from ..core.model_resolution import resolve_model_for_operation, model_display_identity
from ..core.embedding_runner import EmbeddingRunner, resolve_model_dir
from .common import _load_config_json

# Encoder-only model_types — never causal LMs, so a bare model_type match is safe. These are
# declared embedders that this decoder-only build does not run yet (vendoring is Slice B+).
# NB: gemma3_text is deliberately NOT here — it is shared with ordinary causal Gemma-3 text LMs,
# and EmbeddingGemma is distinguished by the sentence-transformers sidecar / bidirectional
# attention, i.e. the config-first detection of Slice C (ADR-015 §Coupled detection fix). Slice A
# does not attempt that distinction — a gemma3_text model falls through to "not an embedder".
_ENCODER_MODEL_TYPES = frozenset(
    {"bert", "xlm-roberta", "modernbert", "nomic_bert"}
)
# Curated decoder embedders whose name may lack an "embed" token (extend in later slices).
_KNOWN_DECODER_EMBEDDERS = ("qwen3-embedding",)


def _err(command: str, etype: str, message: str) -> Dict[str, Any]:
    return {
        "status": "error",
        "command": command,
        "data": None,
        "error": {"type": etype, "message": message},
    }


def _route_embedding(resolved_name: str, commit_hash: Optional[str]) -> Optional[str]:
    """Return ``"decoder"`` | ``"encoder"`` | ``None``. Slice A runs the decoder path only."""
    try:
        model_dir = resolve_model_dir(resolved_name, commit_hash)
    except FileNotFoundError:
        return None
    config = _load_config_json(model_dir) or {}
    model_type = str(config.get("model_type", "")).lower()
    archs = [str(a).lower() for a in (config.get("architectures") or [])]
    name = resolved_name.lower()

    is_causal = any("forcausallm" in a for a in archs) or model_type in ("qwen3", "mistral")
    name_says_embed = "embed" in name or "embedding" in name
    on_known_list = any(k in name for k in _KNOWN_DECODER_EMBEDDERS)
    if is_causal and (name_says_embed or on_known_list):
        return "decoder"
    if model_type in _ENCODER_MODEL_TYPES:
        return "encoder"
    return None


def _parse_batch_items(lines: List[str]) -> Any:
    """Parse JSONL ``--batch`` input into ``[{"text":..., **passthrough}, ...]``.

    Returns an error envelope (a dict with ``status == "error"``) on malformed input.
    """
    items: List[Dict[str, Any]] = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError as e:
            return _err("embed", "ValidationError", f"Malformed JSONL on line {i + 1}: {e}")
        if not isinstance(obj, dict) or "text" not in obj:
            return _err("embed", "ValidationError", f"JSONL line {i + 1} missing 'text' field")
        items.append(obj)
    if not items:
        return _err("embed", "ValidationError", "No input records found (--batch)")
    return items


def embed_operation(
    model_spec: str,
    text: Optional[str] = None,
    *,
    batch: bool = False,
    batch_lines: Optional[List[str]] = None,
    input_type: str = "document",
    instruct: Optional[str] = None,
    cpu: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Embed text(s) and return the standard ``{status, command, data, error}`` envelope."""
    # 1. Resolve model.
    resolved_name, commit_hash, ambiguous = resolve_model_for_operation(model_spec)
    if ambiguous:
        return _err("embed", "AmbiguousMatch",
                    f"Ambiguous model specification '{model_spec}'. Could be: {ambiguous}")
    if not resolved_name:
        return _err("embed", "ModelNotFound", f"Model '{model_spec}' not found")

    # 2. Route (Slice A: decoder only).
    route = _route_embedding(resolved_name, commit_hash)
    if route == "encoder":
        return _err("embed", "NotImplemented",
                    f"'{resolved_name}' is an encoder embedder, which this build does not run yet "
                    f"(decoder embedders only). Use a decoder embedder such as Qwen3-Embedding.")
    if route is None:
        return _err("embed", "NotAnEmbedder",
                    f"Model '{resolved_name}' is not recognized as an embedding model.")

    # 3. Assemble work items (preserve --batch passthrough fields).
    if batch:
        items = _parse_batch_items(batch_lines or [])
        if isinstance(items, dict):  # error envelope
            return items
        texts = [it["text"] for it in items]
    else:
        if text is None:
            return _err("embed", "ValidationError", "No input text provided")
        items = [{"text": text}]
        texts = [text]

    # 4. Run.
    try:
        with EmbeddingRunner(resolved_name, cpu=cpu, verbose=verbose) as runner:
            vectors = runner.embed(texts, input_type=input_type, instruct=instruct)
            dims = runner.dimensions
    except NotImplementedError as e:
        return _err("embed", "NotImplemented", str(e))
    except FileNotFoundError as e:
        return _err("embed", "ModelNotFound", str(e))
    except Exception as e:  # noqa: BLE001 — surface any load/inference failure as an envelope
        return _err("embed", "EmbeddingError", str(e))

    # 5. Build records (metadata stamping). The model identity is a portable org/name plus the
    # content_hash (never an absolute path) so a consumer can honor the same-model rule. `device`
    # is stamped because CPU vs GPU vectors of the same model/text diverge (~0.98 cosine on 4-bit)
    # — enough to break dedup/threshold logic — so a consumer can detect a mixed-device store.
    model_display, content_hash = model_display_identity(resolved_name, commit_hash)
    device = "cpu" if cpu else "gpu"
    records: List[Dict[str, Any]] = []
    for it, vec in zip(items, vectors):
        rec = dict(it)
        rec["embedding"] = vec
        rec["metadata"] = {
            "model": model_display,
            "dimensions": dims,
            "content_hash": content_hash,
            "device": device,
        }
        records.append(rec)

    return {"status": "success", "command": "embed", "data": {"records": records}, "error": None}


def emit_embed_result(result: Dict[str, Any], *, human: bool = False, output_file: Optional[str] = None) -> None:
    """Emit JSONL (default) or a human summary. Errors go to stderr; stdout stays clean."""
    if result.get("status") == "error":
        print(json.dumps(result.get("error", {}), ensure_ascii=False), file=sys.stderr)
        return

    records = result["data"]["records"]
    stream = open(output_file, "w", encoding="utf-8") if output_file else sys.stdout
    try:
        if human:
            for rec in records:
                vec = rec["embedding"]
                meta = rec["metadata"]
                preview = ", ".join(f"{x:.4f}" for x in vec[:4])
                text = str(rec.get("text", ""))
                if len(text) > 60:
                    text = text[:57] + "..."
                print(f"{text!r} -> {meta['dimensions']}-dim [{preview}, ...] "
                      f"(model: {meta['model']})", file=stream)
        else:
            for rec in records:
                print(json.dumps(rec, ensure_ascii=False), file=stream)
    finally:
        if output_file:
            stream.close()
