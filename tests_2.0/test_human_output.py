import re

from mlxk2.output.human import render_list, render_health, render_clone


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
                    "runtime_compatible": True,
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
                    "runtime_compatible": False,
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


def test_health_human_single_item_no_summary():
    """When total==1, summary line is redundant and should be omitted."""
    data = {
        "status": "success",
        "command": "health",
        "data": {
            "healthy": [
                {"name": "./workspace", "status": "healthy", "reason": "Multi-file model complete"}
            ],
            "unhealthy": [],
            "summary": {"total": 1, "healthy_count": 1, "unhealthy_count": 0},
        },
        "error": None,
    }
    out = render_health(data)
    # No summary line for single item
    assert "Summary:" not in out
    # Result line present
    assert "healthy   ./workspace — Multi-file model complete" in out


def test_health_human_zero_items_shows_summary():
    """When total==0, summary is useful to indicate nothing was found."""
    data = {
        "status": "success",
        "command": "health",
        "data": {
            "healthy": [],
            "unhealthy": [],
            "summary": {"total": 0, "healthy_count": 0, "unhealthy_count": 0},
        },
        "error": None,
    }
    out = render_health(data)
    # Summary line present for zero results
    assert "Summary: total 0, healthy 0, unhealthy 0" in out


def test_list_human_filters_mlx_base_default():
    from mlxk2.output.human import render_list

    data = {
        "status": "success",
        "command": "list",
        "data": {
            "models": [
                {
                    "name": "org/MLXChat",
                    "hash": "abcdef0123456789abcdef0123456789abcdef01",
                    "size_bytes": 1000,
                    "last_modified": "2025-08-30T12:00:00Z",
                    "framework": "MLX",
                    "model_type": "chat",
                    "capabilities": ["text-generation", "chat"],
                    "health": "healthy",
                    "runtime_compatible": True,
                    "cached": True,
                },
                {
                    "name": "org/MLXBase",
                    "hash": "abcdef0123456789abcdef0123456789abcdef02",
                    "size_bytes": 2000,
                    "last_modified": "2025-08-30T12:00:00Z",
                    "framework": "MLX",
                    "model_type": "base",
                    "capabilities": ["text-generation"],
                    "health": "healthy",
                    "runtime_compatible": True,
                    "cached": True,
                },
                {
                    "name": "org/Unhealthy",
                    "hash": None,
                    "size_bytes": 500,
                    "last_modified": "2025-08-30T12:00:00Z",
                    "framework": "MLX",
                    "model_type": "chat",
                    "capabilities": ["text-generation"],
                    "health": "unhealthy",
                    "runtime_compatible": False,
                    "cached": True,
                },
            ],
            "count": 3,
        },
        "error": None,
    }

    # Default: shows healthy + runtime_compatible models (both MLXChat and MLXBase)
    out_default = render_list(data, show_health=False, show_all=False, verbose=False)
    assert "MLXChat" in out_default
    assert "MLXBase" in out_default
    assert "Unhealthy" not in out_default

    # Verbose: same filter, more columns
    out_verbose = render_list(data, show_health=False, show_all=False, verbose=True)
    assert "MLXChat" in out_verbose
    assert "MLXBase" in out_verbose
    assert "Unhealthy" not in out_verbose


def test_list_human_filters_by_healthy_and_runtime_compatible():
    """Test that default/verbose filters by healthy + runtime_compatible."""
    from mlxk2.output.human import render_list

    data = {
        "status": "success",
        "command": "list",
        "data": {
            "models": [
                {"name": "org/Runnable", "hash": None, "size_bytes": 1, "last_modified": "2025-08-30T12:00:00Z", "framework": "MLX", "model_type": "chat", "capabilities": ["text-generation", "chat"], "health": "healthy", "runtime_compatible": True, "cached": True},
                {"name": "org/Unhealthy", "hash": None, "size_bytes": 1, "last_modified": "2025-08-30T12:00:00Z", "framework": "MLX", "model_type": "base", "capabilities": ["text-generation"], "health": "unhealthy", "runtime_compatible": True, "cached": True},
                {"name": "org/NotCompatible", "hash": None, "size_bytes": 1, "last_modified": "2025-08-30T12:00:00Z", "framework": "PyTorch", "model_type": "base", "capabilities": ["text-generation"], "health": "healthy", "runtime_compatible": False, "cached": True},
            ],
            "count": 3,
        },
        "error": None,
    }

    out_verbose = render_list(data, show_health=False, show_all=False, verbose=True)
    # Shows only healthy + runtime_compatible
    assert "Runnable" in out_verbose
    # Hides unhealthy
    assert "Unhealthy" not in out_verbose
    # Hides not runtime_compatible
    assert "NotCompatible" not in out_verbose


def test_list_human_all_shows_all_frameworks():
    from mlxk2.output.human import render_list

    data = {
        "status": "success",
        "command": "list",
        "data": {
            "models": [
                {"name": "org/MLXChat", "hash": None, "size_bytes": 1, "last_modified": "2025-08-30T12:00:00Z", "framework": "MLX", "model_type": "chat", "capabilities": ["text-generation", "chat"], "health": "healthy", "cached": True},
                {"name": "org/OtherGGUF", "hash": None, "size_bytes": 1, "last_modified": "2025-08-30T12:00:00Z", "framework": "GGUF", "model_type": "base", "capabilities": ["text-generation"], "health": "unhealthy", "cached": True},
                {"name": "org/OtherPT", "hash": None, "size_bytes": 1, "last_modified": "2025-08-30T12:00:00Z", "framework": "PyTorch", "model_type": "base", "capabilities": ["text-generation"], "health": "healthy", "cached": True},
            ],
            "count": 3,
        },
        "error": None,
    }

    out_all = render_list(data, show_health=False, show_all=True, verbose=False)
    assert "MLXChat" in out_all
    assert "OtherGGUF" in out_all
    assert "OtherPT" in out_all


def test_list_human_type_shows_vision_flag():
    from mlxk2.output.human import render_list

    data = {
        "status": "success",
        "command": "list",
        "data": {
            "models": [
                {
                    "name": "org/VisionChat",
                    "hash": None,
                    "size_bytes": 1,
                    "last_modified": "2025-08-30T12:00:00Z",
                    "framework": "MLX",
                    "model_type": "chat",
                    "capabilities": ["text-generation", "chat", "vision"],
                    "health": "healthy",
                    "runtime_compatible": True,
                    "cached": True,
                }
            ],
            "count": 1,
        },
        "error": None,
    }

    out = render_list(data, show_health=False, show_all=True, verbose=False)
    assert "chat+vision" in out


def test_clone_unhealthy_shows_warning():
    """Test that unhealthy clone result shows warning in human output."""
    data = {
        "status": "success",
        "command": "clone",
        "data": {
            "model": "mlx-community/broken-model",
            "target_dir": "/path/to/workspace",
            "clone_status": "success",
            "health": "unhealthy",
            "health_reason": "Missing weight shards: model-00001-of-00010.safetensors",
            "message": "Cloned to /path/to/workspace"
        },
        "error": None
    }

    out = render_clone(data, quiet=False)
    # Should show warning symbol and reason
    assert "⚠" in out
    assert "unhealthy" in out
    assert "Missing weight shards" in out
    assert "/path/to/workspace" in out


def test_clone_healthy_shows_checkmark():
    """Test that healthy clone result shows checkmark."""
    data = {
        "status": "success",
        "command": "clone",
        "data": {
            "model": "mlx-community/good-model",
            "target_dir": "/path/to/workspace",
            "clone_status": "success",
            "health": "healthy",
            "health_reason": "Multi-file model complete",
            "message": "Cloned to /path/to/workspace"
        },
        "error": None
    }

    out = render_clone(data, quiet=False)
    # Should show checkmark
    assert "✓ healthy" in out
    assert "/path/to/workspace" in out


def test_clone_no_health_check_omits_status():
    """Test that clone without health check doesn't show health status."""
    data = {
        "status": "success",
        "command": "clone",
        "data": {
            "model": "mlx-community/model",
            "target_dir": "/path/to/workspace",
            "clone_status": "success",
            "message": "Cloned to /path/to/workspace"
            # No health or health_reason fields
        },
        "error": None
    }

    out = render_clone(data, quiet=False)
    # Should not show health status
    assert "healthy" not in out
    assert "unhealthy" not in out
    assert "✓" not in out
    assert "⚠" not in out
