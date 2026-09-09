"""The model_file gate: mlx-knife refuses a checkpoint that ships its own Python.

A config declaring ``model_file`` makes mlx-lm import and execute that file from inside the
checkpoint (CVE-2026-5843). It is ungated in 0.31.3 — the version we pin — and ungated in
every mlx-vlm that carries the branch at all, and upstream's fix has never been released, so
mlx-knife refuses on its own.

Two properties are pinned here, and they are the whole point:

1. The refusal happens **before** the backend is called. A test that only checks the message
   would pass even if the module had already been executed.
2. The capability surfaces say what the runner says. A model mlx-knife will not load is not
   runnable, so ``check_runtime_compatibility`` and the Reason on ``list``/``show`` carry it
   instead of reporting a model compatible that ``run`` then refuses.

A ``config.json`` is all the gate reads, so the fixtures need no weights and no model — and
they live in ``tmp_path``, never in a cache or workspace of the machine running the tests.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mlxk2.core.remote_code import (
    MODEL_FILE_REASON,
    UntrustedModelCodeError,
    declared_model_file,
    reject_untrusted_model_code,
)


_REPO = Path(__file__).resolve().parents[1]


def _checkpoint(root: Path, config, name: str = "cp") -> Path:
    """A directory shaped enough like a model for the gate: config.json, or none at all."""
    path = root / name
    path.mkdir(parents=True)
    if config is not None:
        (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return path


# -- the predicate ---------------------------------------------------------------------

def test_predicate_reports_the_declared_file():
    assert declared_model_file({"model_type": "llama", "model_file": "arch.py"}) == "arch.py"


def test_predicate_ignores_a_config_without_the_key():
    assert declared_model_file({"model_type": "llama"}) is None


def test_predicate_ignores_an_explicit_null():
    """mlx-lm tests ``is not None``; a null executes nothing, so neither do we."""
    assert declared_model_file({"model_file": None}) is None


def test_predicate_tolerates_a_non_dict():
    assert declared_model_file(None) is None
    assert declared_model_file(["not", "a", "config"]) is None


# -- the guard ------------------------------------------------------------------------

def test_guard_rejects_and_names_what_it_refuses(tmp_path):
    path = _checkpoint(tmp_path, {"model_type": "llama", "model_file": "arch.py"})

    with pytest.raises(UntrustedModelCodeError) as exc:
        reject_untrusted_model_code(path)

    message = str(exc.value)
    assert "arch.py" in message, "the reject must name the file it refuses to run"
    assert str(path) in message, "the reject must name the checkpoint"


def test_guard_passes_an_ordinary_checkpoint(tmp_path):
    reject_untrusted_model_code(_checkpoint(tmp_path, {"model_type": "llama"}))


def test_guard_is_silent_without_a_config(tmp_path):
    """Whether a directory is a usable model is health's question, not the gate's."""
    reject_untrusted_model_code(_checkpoint(tmp_path, None))


def test_guard_is_silent_on_an_unreadable_config(tmp_path):
    path = tmp_path / "broken"
    path.mkdir()
    (path / "config.json").write_text("{not json", encoding="utf-8")

    reject_untrusted_model_code(path)


def test_guard_accepts_a_string_path(tmp_path):
    """Callers hand it a Path today; a str must not silently skip the check."""
    path = _checkpoint(tmp_path, {"model_file": "arch.py"})

    with pytest.raises(UntrustedModelCodeError):
        reject_untrusted_model_code(str(path))


def test_guard_ignores_none():
    reject_untrusted_model_code(None)


# -- pre-exec, not post-hoc -----------------------------------------------------------

def test_text_runner_refuses_before_calling_load(tmp_path):
    """The backend must never be reached — that is the difference between a gate and a log."""
    from mlxk2.core.runner import MLXRunner

    path = _checkpoint(tmp_path, {"model_type": "llama", "model_file": "arch.py"})

    with patch("mlxk2.core.runner.load") as spy_load, \
         patch("mlxk2.core.runner.resolve_model_for_operation") as resolve:
        resolve.return_value = (str(path), None, None)

        with pytest.raises(UntrustedModelCodeError):
            MLXRunner(str(path)).load_model()

        spy_load.assert_not_called()


def test_embedding_runner_refuses_before_calling_load(tmp_path):
    from mlxk2.core.embedding_runner import EmbeddingRunner

    path = _checkpoint(tmp_path, {"model_type": "qwen3", "model_file": "arch.py"})

    with patch("mlxk2.core.embedding_runner.load") as spy_load, \
         patch("mlxk2.core.embedding_runner.resolve_model_for_operation") as resolve, \
         patch("mlxk2.core.embedding_runner.resolve_model_dir") as resolve_dir:
        resolve.return_value = (str(path), None, None)
        resolve_dir.return_value = path

        with pytest.raises(UntrustedModelCodeError):
            EmbeddingRunner(str(path)).load_model()

        spy_load.assert_not_called()


def test_vision_runner_refuses_before_calling_load(tmp_path):
    """mlx-vlm 0.6.10 has no model_file branch; the ones that do have no gate of their own.

    Spies ``_load_model_impl`` rather than ``runner.model``: ``__init__`` already sets that to
    None, so asserting on it would hold whether the backend ran or not.
    """
    from mlxk2.core.vision_runner import VisionRunner

    path = _checkpoint(tmp_path, {"model_type": "qwen2_vl", "model_file": "arch.py"})
    runner = VisionRunner(path, str(path))

    with patch.object(runner, "_load_model_impl") as spy_impl:
        with pytest.raises(UntrustedModelCodeError):
            runner.load_model()

        spy_impl.assert_not_called()


@pytest.mark.parametrize(
    "relative_path, function",
    [
        ("mlxk2/core/runner/__init__.py", "load_model"),
        ("mlxk2/core/embedding_runner.py", "load_model"),
        ("mlxk2/core/vision_runner.py", "load_model"),
        ("mlxk2/core/audio_runner.py", "load_model"),
        ("mlxk2/operations/convert.py", "_quantize_text_model"),
        ("mlxk2/operations/convert.py", "_quantize_vision_model"),
        ("mlxk2/operations/run.py", "run_model"),
    ],
)
def test_every_call_site_still_carries_the_guard(relative_path, function):
    """All seven call sites, pinned by source rather than by behaviour.

    Four of them are covered behaviourally above. ``audio_runner`` cannot be: importing it
    applies the Whisper bridge at module scope, which pulls in the real ``mlx.nn`` and
    collides with the stubbed ``mlx.core`` this tree collects under — the same reason the
    canaries read source instead of importing. Reading source is the weaker instrument, but
    a deleted call is exactly the mutation it catches, and without it a guard could be
    dropped from a backend with the whole suite staying green.
    """
    import ast

    source = (_REPO / relative_path).read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef) and node.name == function:
            calls = {
                n.func.id
                for n in ast.walk(node)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            }
            assert "reject_untrusted_model_code" in calls, (
                f"{relative_path}:{function}() no longer calls reject_untrusted_model_code. "
                f"Every path that hands a checkpoint to a backend has to refuse first "
                f"(CVE-2026-5843); if this site genuinely moved, move the assertion with it."
            )
            return

    pytest.fail(f"{relative_path} no longer defines {function}()")


@pytest.mark.parametrize(
    "backend_module, quantize_fn, model_type",
    [
        ("mlx_lm", "_quantize_text_model", "llama"),
        ("mlx_vlm", "_quantize_vision_model", "qwen2_vl"),
    ],
)
def test_convert_refuses_before_calling_the_backend(
    tmp_path, backend_module, quantize_fn, model_type
):
    """``convert --quantize`` reaches the same exec: both backends load the source in order
    to quantize it. The module is replaced wholesale so the reject cannot be mistaken for an
    import failure, and so `convert` is not the one covered verb no test pins."""
    from mlxk2.operations import convert as convert_mod

    source = _checkpoint(tmp_path, {"model_type": model_type, "model_file": "arch.py"}, name="src")
    backend = MagicMock()

    with patch.dict(sys.modules, {backend_module: backend}):
        with pytest.raises(UntrustedModelCodeError):
            getattr(convert_mod, quantize_fn)(source, tmp_path / "out", 4)

    backend.convert.assert_not_called()


# -- one truth across the surfaces ----------------------------------------------------

def test_runtime_compatibility_carries_the_reason(tmp_path):
    from mlxk2.operations.health import check_runtime_compatibility

    path = _checkpoint(tmp_path, {"model_type": "llama", "model_file": "arch.py"})
    (path / "model.safetensors").write_bytes(b"\0" * 64)

    compatible, reason = check_runtime_compatibility(path, "MLX")

    assert compatible is False
    assert reason == MODEL_FILE_REASON


def test_runtime_compatibility_prefers_it_over_an_unknown_model_type(tmp_path):
    """18 of the checkpoints measured in the wild carry a model_type mlx-lm lacks. The
    honest reason is the refusal, not "model_type not supported" — that one would suggest a
    backend bump would help, and it would not."""
    from mlxk2.operations.health import check_runtime_compatibility

    path = _checkpoint(tmp_path, {"model_type": "no_such_architecture", "model_file": "a.py"})
    (path / "model.safetensors").write_bytes(b"\0" * 64)

    _, reason = check_runtime_compatibility(path, "MLX")

    assert reason == MODEL_FILE_REASON


def test_runtime_compatibility_reports_it_without_a_model_type(tmp_path):
    """One measured checkpoint declares no model_type at all; the refusal still wins."""
    from mlxk2.operations.health import check_runtime_compatibility

    path = _checkpoint(tmp_path, {"model_file": "arch.py"})
    (path / "model.safetensors").write_bytes(b"\0" * 64)

    _, reason = check_runtime_compatibility(path, "MLX")

    assert reason == MODEL_FILE_REASON


@pytest.mark.parametrize(
    "config",
    [
        {"model_type": "llama", "model_file": "arch.py"},
        # The audio and embedding branches of build_model_object never reach
        # check_runtime_compatibility, so the gate has to sit above the modality split.
        {"model_type": "whisper", "model_file": "arch.py"},
        {"model_type": "bert", "model_file": "arch.py"},
    ],
)
def test_list_reason_column_carries_it_on_every_modality(tmp_path, config):
    from mlxk2.operations import common

    path = _checkpoint(tmp_path, config)
    (path / "model.safetensors").write_bytes(b"\0" * 64)

    # A directory carrying config.json is a workspace (ADR-022), so health comes from
    # health_check_workspace; only the framework hint needs standing in for a README.
    with patch.object(common, "detect_framework", return_value="MLX"):
        model = common.build_model_object(str(path), path, path)

    assert model["health"] == "healthy", "fixture must pass health, or reason would be health's"

    assert model["runtime_compatible"] is False
    assert model["reason"] == MODEL_FILE_REASON


def test_run_returns_the_reject_instead_of_raising(tmp_path):
    """ADR-024 Class A form: the reject is the result string, so --json stays clean."""
    from mlxk2.operations import run as run_mod

    path = _checkpoint(tmp_path, {"model_type": "llama", "model_file": "arch.py"})

    with patch.object(run_mod, "MLXRunner") as runner_cls:
        result = run_mod.run_model(str(path), "hi", json_output=True)

    assert result.startswith("Error: ")
    assert "arch.py" in result
    runner_cls.assert_not_called()
