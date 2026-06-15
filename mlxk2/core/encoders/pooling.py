"""Encoder pooling + prompt-prefix inference (ADR-015 Slice B).

Pure string/dict logic — no MLX, fully unit-testable under the lightweight test stub.

mlx-community conversions strip ``1_Pooling/config.json`` (only ``modules.json`` survives, which
confirms a Normalize step but not the CLS-vs-mean mode), so pooling is inferred per model from the
name/family, per ADR-015 §Decision: Verified-Encoder List (bge→CLS, e5/sentence-transformers→mean).
L2-normalization is applied to every encoder output regardless.
"""

from typing import Any, Dict, Optional, Tuple

# bge and mxbai share the same retrieval query instruction and both pool on the CLS token.
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def infer_pool_and_family(
    name: str,
    config: Optional[Dict[str, Any]] = None,
    modules_json: Optional[Any] = None,
) -> Tuple[str, str]:
    """Infer ``(pool_strategy, family)`` from the model name.

    - ``bge`` / ``mxbai`` → (``"cls"``, ``"bge"``) — CLS pooling + the bge/mxbai query instruction.
    - ``e5`` → (``"mean"``, ``"e5"``) — mean pooling + ``query:`` / ``passage:`` prefixes.
    - everything else (MiniLM, generic sentence-transformers) → (``"mean"``, ``"generic"``).

    ``config`` / ``modules_json`` are accepted for forward-compatibility (a future sidecar-based
    refinement) but are intentionally unused: the verified conversions strip the pooling sidecar.
    """
    n = (name or "").lower()
    if "bge" in n or "mxbai" in n:
        return "cls", "bge"
    if "e5" in n:
        return "mean", "e5"
    return "mean", "generic"


def encoder_prefix(family: str, input_type: str) -> str:
    """Return the prompt prefix for an encoder family + input role (``query`` | ``document``).

    - ``bge``: query gets the retrieval instruction; documents are embedded raw.
    - ``e5``: both sides are prefixed (``query: `` / ``passage: ``) — asymmetric, always-on.
    - ``generic``: no prefix either way.
    """
    if family == "bge":
        return _BGE_QUERY_PREFIX if input_type == "query" else ""
    if family == "e5":
        return "query: " if input_type == "query" else "passage: "
    return ""
