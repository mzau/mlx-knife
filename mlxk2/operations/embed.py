"""``mlxk embed`` operation (ADR-015, Slices A + B + C).

Generates embeddings via :class:`EmbeddingRunner` and emits JSONL — the default machine
format (ADR-015 §Output): the typical consumer is another program (cosine-search, index
builder), not a human at a terminal.

Routing delegates to :func:`mlxk2.core.capabilities.classify_embedder` — the single,
config-first source of truth shared with the list/show metadata path and the serve-load probe
(Slice C). The decoder path plus the vendored ``bert`` encoder are runnable; other declared
encoder types (xlm-roberta/modernbert/nomic_bert) are surfaced honestly as declared-but-not-runnable.
"""

import json
import sys
from typing import Any, Dict, List, Optional, Tuple

from ..core.model_resolution import resolve_model_for_operation, model_display_identity
from ..core.embedding_runner import EmbeddingRunner, resolve_model_dir
from ..core.capabilities import classify_embedder
from .common import _load_config_json


def _err(command: str, etype: str, message: str) -> Dict[str, Any]:
    return {
        "status": "error",
        "command": command,
        "data": None,
        "error": {"type": etype, "message": message},
    }


def _route_embedding(resolved_name: str, commit_hash: Optional[str]) -> Tuple[Optional[str], str]:
    """Return ``(route, model_type)`` where route is
    ``"decoder"`` | ``"encoder"`` | ``"encoder_declared"`` | ``None``.

    - ``decoder`` — runnable via mlx-lm (qwen3/mistral causal embedder).
    - ``encoder`` — runnable via the vendored BERT (``model_type: bert``).
    - ``encoder_declared`` — a declared encoder embedder this build does not vendor (honest reject).
    - ``None`` — not recognized as an embedder.
    """
    try:
        model_dir = resolve_model_dir(resolved_name, commit_hash)
    except FileNotFoundError:
        return None, ""
    config = _load_config_json(model_dir) or {}
    return classify_embedder(config, resolved_name)


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

    # 2. Route (Slice B: decoder + vendored bert encoder; other encoders honest-reject).
    route, model_type = _route_embedding(resolved_name, commit_hash)
    if route == "encoder_declared":
        return _err("embed", "NotImplemented",
                    f"'{resolved_name}' is a {model_type} encoder embedder, which this build does "
                    f"not vendor yet. Use bge/e5 (model_type bert) or a decoder embedder such as "
                    f"Qwen3-Embedding.")
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
