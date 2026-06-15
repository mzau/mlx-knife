"""Embedding runner for MLX-Knife (ADR-015, Slice A: decoder vertical).

Stateless ``text -> vector``. Slice A implements the *decoder* path only (rides
``mlx-lm``, zero vendored model code); the encoder path (vendored BERT/XLM-R) is a
``NotImplementedError`` stub filled in Slice B.

Pre-LM-head hidden states (verified, mlx-lm 0.31.3): for a decoder model loaded via
``mlx_lm.load()``, ``model.model(inputs)`` returns the transformer body output — already
through the final RMSNorm — and therefore *bypasses the LM head*. That is exactly the
embedding hidden state. Last-token pooling + L2-normalize follows. The
``self.model.model`` access is kept in one place as a single point of repair if a future
mlx-lm changes the qwen3 ``Model``/``Qwen3Model`` attribute layout.
"""

import gc
from pathlib import Path
from typing import List, Optional

from .cache import get_current_model_cache, hf_to_cache_dir
from .model_resolution import resolve_model_for_operation
from ..operations.workspace import is_workspace_path

# Defer MLX / MLX-LM imports to runtime to avoid init crashes during test collection.
# Patchable names for tests (set lazily inside load_model or patched by tests).
mx = None  # type: ignore[assignment]
load = None  # type: ignore[assignment]

# Default Qwen3-Embedding query instruction. Applied ONLY for input_type="query"; the
# document/passage side is embedded raw on the decoder path (corpus-safe, symmetric).
_QWEN3_DEFAULT_TASK = (
    "Given a web search query, retrieve relevant passages that answer the query"
)


def resolve_model_dir(resolved_name: str, commit_hash: Optional[str] = None) -> Path:
    """Resolve an already-resolved model name/path to an on-disk model directory.

    Workspace paths (ADR-022) are used directly; cache names map to the pinned (or latest)
    snapshot directory. Mirrors the path block in ``MLXRunner.load_model``.
    """
    if is_workspace_path(resolved_name):
        return Path(resolved_name)
    model_cache = get_current_model_cache()
    base = Path(model_cache) / hf_to_cache_dir(resolved_name)
    snapshots = base / "snapshots"
    if commit_hash:
        return snapshots / commit_hash
    if snapshots.exists():
        dirs = [d for d in snapshots.iterdir() if d.is_dir()]
        if dirs:
            # Prefer most recently modified snapshot.
            return max(dirs, key=lambda d: d.stat().st_mtime)
    raise FileNotFoundError(
        f"Model '{resolved_name}' not found in cache (expected under {base}). "
        f"Try: mlxk pull {resolved_name}"
    )


class EmbeddingRunner:
    """Decoder-path embedding engine (Slice A). Use as a context manager."""

    def __init__(self, model_name_or_path: str, *, cpu: bool = False, verbose: bool = False):
        self.model_spec = model_name_or_path
        self.cpu = cpu
        self.verbose = verbose
        self.model = None
        self.tokenizer = None
        self._mx = None
        self._model_type: Optional[str] = None
        self._dimensions: Optional[int] = None
        # Decoder default is last-token; the "mean" branch is exercised by Slice B encoders.
        self._pool_strategy = "last_token"
        self._model_loaded = False
        self._context_entered = False

    # -- context manager -----------------------------------------------------
    def __enter__(self):
        if self._context_entered:
            raise RuntimeError("EmbeddingRunner context manager cannot be entered multiple times")
        self._context_entered = True
        try:
            self.load_model()
            return self
        except Exception:
            self._context_entered = False
            self.cleanup()
            raise

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._context_entered = False
        self.cleanup()
        return False

    # -- loading -------------------------------------------------------------
    def _resolve_model_path(self) -> Path:
        resolved_name, commit_hash, ambiguous = resolve_model_for_operation(self.model_spec)
        if ambiguous:
            raise ValueError(
                f"Ambiguous model specification '{self.model_spec}'. Could be: {ambiguous}"
            )
        if not resolved_name:
            resolved_name = str(self.model_spec)
        return resolve_model_dir(resolved_name, commit_hash)

    def load_model(self):
        """Load the decoder model + tokenizer via mlx-lm."""
        if self._model_loaded:
            return
        try:
            import mlx.core as _mx  # type: ignore
        except Exception as e:
            raise RuntimeError(f"Failed to import MLX core: {e}") from e
        if self.cpu:
            # One-shot process: setting the default device is the simplest, correct override.
            _mx.set_default_device(_mx.cpu)
        # Prefer a test-patched load if present.
        _load = globals().get("load")
        if _load is None:
            try:
                from mlx_lm import load as _load  # type: ignore
            except Exception as e:
                raise RuntimeError(f"Failed to import MLX-LM load(): {e}") from e

        model_path = self._resolve_model_path()
        if self.verbose:
            print(f"Loading embedding model from {model_path}...")
        self.model, self.tokenizer = _load(str(model_path))
        # model_type and hidden_size come straight off the loaded mlx-lm Model.
        self._model_type = (getattr(self.model, "model_type", None) or "")
        args = getattr(self.model, "args", None)
        self._dimensions = getattr(args, "hidden_size", None)
        self._mx = _mx
        self._model_loaded = True

    # -- embedding -----------------------------------------------------------
    def _apply_prefix(self, text: str, input_type: str, instruct: Optional[str]) -> str:
        """Model-aware prefix resolver. Slice A populates only the qwen3 decoder entry;
        Slice B adds bge/e5 encoder branches here (E5's document side is ``passage:``, not raw).
        """
        mt = (self._model_type or "").lower()
        if mt == "qwen3" and input_type == "query":
            task = instruct or _QWEN3_DEFAULT_TASK
            return f"Instruct: {task}\nQuery:{text}"
        # document / none -> raw (qwen3 corpus side).
        return text

    def _pool(self, hidden):
        if self._pool_strategy == "mean":
            return hidden[0].mean(axis=0)
        # last-token (default for qwen3 decoders).
        return hidden[0, -1, :]

    def embed(
        self,
        texts: List[str],
        *,
        input_type: str = "document",
        instruct: Optional[str] = None,
    ) -> List[List[float]]:
        """Embed each text to an L2-normalized vector (decoder last-token pooling)."""
        if not self._model_loaded:
            raise RuntimeError("Model not loaded. Use EmbeddingRunner as a context manager.")
        mx_core = self._mx
        eos = getattr(self.tokenizer, "eos_token_id", None)
        out: List[List[float]] = []
        for text in texts:
            prepared = self._apply_prefix(text, input_type, instruct)
            ids = list(self.tokenizer.encode(prepared))
            # Qwen3-Embedding pools the trailing EOS token; ensure it is present.
            if eos is not None and (not ids or ids[-1] != eos):
                ids.append(eos)
            inputs = mx_core.array([ids])              # (1, seq_len)
            hidden = self.model.model(inputs)          # (1, seq_len, hidden) — bypasses LM head
            pooled = self._pool(hidden)                # (hidden,)
            pooled = pooled.astype(mx_core.float32)    # embeddings are conventionally float32
            normed = pooled / mx_core.linalg.norm(pooled)
            mx_core.eval(normed)                       # force lazy graph before .tolist()
            vec = [float(x) for x in normed.tolist()]
            out.append(vec)
            if self._dimensions is None:
                self._dimensions = len(vec)
        return out

    def embed_encoder(self, texts: List[str]) -> List[List[float]]:
        """Encoder (BERT/XLM-R) path — vendored in Slice B."""
        raise NotImplementedError(
            "Encoder embedding (BERT/XLM-R) is not implemented in this release "
            "(ADR-015 Slice B). Use a decoder embedder such as Qwen3-Embedding."
        )

    @property
    def dimensions(self) -> Optional[int]:
        return self._dimensions

    # -- cleanup -------------------------------------------------------------
    def cleanup(self):
        mx_core = self._mx
        self.model = None
        self.tokenizer = None
        self._model_loaded = False
        gc.collect()
        try:
            if mx_core is not None:
                mx_core.clear_cache()
        except (ImportError, AttributeError):
            pass
        except Exception:
            pass
