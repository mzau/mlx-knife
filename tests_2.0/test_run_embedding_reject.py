"""Regression: `mlxk run <embedder>` gives the honest 'use mlxk embed' reject (ADR-015 Slice C).

A config-first encoder embedder (model_type `bert`, e.g. bge) classifies as `embedding` and
surfaces as runnable in `mlxk list`; running it (the wrong verb) must give the honest reject that
points to `mlxk embed`, not a cryptic mlx-lm loader error.

Tested via the WORKSPACE path (ADR-022 workspace-first, the supported fixture style). The workspace
path reaches the path-agnostic Class-A reject directly, so this guards the honest-reject contract +
the "use mlxk embed" hint. (The cache-path's extra compat shadowing is the interim local fix in
run.py; its full path-uniform coverage rides with the run.py split / ADR-026.)
"""

import json
from unittest.mock import patch

import pytest


def _run_workspace_embedder(tmp_path, model_type):
    from mlxk2.operations.run import run_model

    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": model_type, "architectures": ["BertModel"]})
    )
    (tmp_path / "model.safetensors").write_bytes(b"x" * 64)
    (tmp_path / ".mlxk-workspace.json").write_text(json.dumps({"managed_by": "mlxk", "source": "test"}))
    (tmp_path / "README.md").write_text("---\nlibrary_name: mlx\n---\n# test\n")

    with patch("mlxk2.operations.run.resolve_model_for_operation",
               return_value=(str(tmp_path), None, None)):
        return run_model(model_spec=str(tmp_path), prompt="hi")


@pytest.mark.parametrize("model_type", ["bert", "modernbert"])
def test_workspace_encoder_embedder_run_gives_honest_reject(tmp_path, model_type):
    result = _run_workspace_embedder(tmp_path, model_type)
    assert result is not None
    assert "is an embedding model" in result
    assert "mlxk embed" in result
    # Not the cryptic mlx-lm loader error:
    assert "not supported" not in result
    assert "not compatible" not in result
