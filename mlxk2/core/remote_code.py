"""Refuse a checkpoint that asks mlx-knife to execute its own Python (CVE-2026-5843).

Both backends build a model architecture from a file *inside* the checkpoint when its
``config.json`` declares ``model_file``: mlx-lm in ``load_model`` — ungated in 0.31.3, the
version we pin — and mlx-vlm in ``get_model_and_args``, ungated on every version that has
the branch at all. Upstream mlx-lm put the same key behind ``trust_remote_code`` on
2026-06-11 and has released nothing since 2026-04-22, so there is no version to upgrade
to. mlx-knife gates itself instead.

Backend-independent on purpose: the key lives in the config, not in the loader, so one
check wherever mlx-knife hands a directory to a backend covers the mlx-lm path today and
the mlx-vlm path after a bump. ⚠ One caller hands a *name* rather than a directory —
``VisionRunner`` passes the repo id when its selected snapshot carries no ``config.json``,
and mlx-vlm then resolves the name itself. In that case the checked directory and the
loaded one can differ; it is inert on the current pin, which has no ``model_file`` branch,
and has to be closed before one arrives. Without an opt-in on purpose: ADR-024 rejects before
execution and does not negotiate, and the alternative for a checkpoint whose ``model_file``
merely *overrides* a loader mlx-lm already has would be to run its built-in one — which is
the wrong answer for the one case we measured, where upstream's own implementation is the
broken one. Refusing is honest; degrading silently is not (ADR-023 §4).

⚠ Scope: ``model_file`` only. Python a checkpoint declares through transformers'
``auto_map`` is a *different* mechanism and this module does not see it — mlx-vlm forces
``trust_remote_code=True`` for the model types it ships processors for, and mlx-audio
hardcodes it in its STT loaders, so the vision and audio paths can still execute
checkpoint-supplied code. Not fixable by adding a key here: ``auto_map`` also appears in
checkpoints under which nothing is ever executed, so the discriminator has to be *would
this loader run it*, not *does the config name it*.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

# What `list`/`show` put in their Reason column. Short because it shares a line with a model
# name; the long form below belongs to `run`, which has the whole line to itself.
MODEL_FILE_REASON = "requires executing model-supplied code (config.json: model_file)"


class UntrustedModelCodeError(RuntimeError):
    """A checkpoint declared ``model_file``; mlx-knife will not execute it.

    Subclasses ``RuntimeError`` so every ``except Exception`` handler on the existing load
    paths keeps behaving as it did.
    """


def declared_model_file(config: Optional[Dict[str, Any]]) -> Optional[str]:
    """The ``model_file`` a config declares, or ``None`` — the predicate every surface reads.

    One function so the runner's refusal and the capability surfaces cannot drift apart: a
    checkpoint mlx-knife will not load is not runnable, and `list` / `show` have to say the
    same thing `run` says (ADR-024, capability = declared ∩ runnable).

    Tests ``is not None``, exactly as mlx-lm does. mlx-vlm tests truthiness instead, so
    ``"model_file": ""`` executes nothing there and merely fails in mlx-lm; taking the
    stricter reading would reject a ``null`` for no gain.
    """
    if not isinstance(config, dict):
        return None
    model_file = config.get("model_file")
    return None if model_file is None else str(model_file)


def reject_untrusted_model_code(model_path: Optional[Union[str, Path]]) -> None:
    """Raise when the config at ``model_path`` asks for its own Python to be imported.

    Silent for a missing or unreadable ``config.json``: whether a directory is a usable
    model is ``health``'s question, and answering it here would add a second, divergent
    health gate.
    """
    if model_path is None:
        return

    # Lazy like embed_server_base.py: operations imports core, not the other way round.
    from ..operations.common import _load_config_json

    model_file = declared_model_file(_load_config_json(Path(model_path)))
    if model_file is None:
        return

    raise UntrustedModelCodeError(
        f"The model at {model_path} requires importing and running a custom module "
        f"({model_file!r}). mlx-knife refuses to execute it."
    )
