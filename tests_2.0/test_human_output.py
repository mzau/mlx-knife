import re

from mlxk2.output.human import render_list, render_health


def sample_list_data():
    return {
        "status": "success",
        "command": "list",
        "data": {
            "models": [
                {
                    "name": "mlx-community/TinyChat",
                    "hash": "abcdef0123456789abcdef0123456789abcdef01",
                    "size_bytes": 1_234_567,
                    "last_modified": "2025-08-30T12:00:00Z",
                    "framework": "MLX",
                    "model_type": "chat",
                    "capabilities": ["text-generation", "chat"],
                    "health": "healthy",
                    "cached": True,
                },
                {
                    "name": "other-org/some-gguf",
                    "hash": None,
                    "size_bytes": 2_000,
                    "last_modified": "2025-08-30T11:00:00Z",
                    "framework": "GGUF",
                    "model_type": "base",
                    "capabilities": ["text-generation"],
                    "health": "unhealthy",
                    "cached": True,
                },
            ],
            "count": 2,
        },
        "error": None,
    }


def test_list_human_compact_filters_and_headers():
    out = render_list(sample_list_data(), show_health=False, show_all=False, verbose=False)
    # No Framework column in compact mode
    header = out.splitlines()[0]
    assert "Framework" not in header
    assert "Modified" in header
    # Only MLX model should be shown, with compact name
    assert "TinyChat" in out
    assert "mlx-community/" not in out
    assert "some-gguf" not in out


def test_list_human_all_and_verbose_shows_framework_and_full_names():
    out = render_list(sample_list_data(), show_health=False, show_all=True, verbose=True)
    header = out.splitlines()[0]
    assert "Framework" in header
    assert "mlx-community/TinyChat" in out
    assert "other-org/some-gguf" in out
    # Framework labels present
    assert "MLX" in out and "GGUF" in out


def test_health_human_summary_and_entries():
    data = {
        "status": "success",
        "command": "health",
        "data": {
            "healthy": [
                {"name": "model-a", "status": "healthy", "reason": "ok"}
            ],
            "unhealthy": [
                {"name": "model-b", "status": "unhealthy", "reason": "missing"}
            ],
            "summary": {"total": 2, "healthy_count": 1, "unhealthy_count": 1},
        },
        "error": None,
    }
    out = render_health(data)
    assert "Summary: total 2, healthy 1, unhealthy 1" in out
    assert "model-a" in out
    assert "model-b" in out

