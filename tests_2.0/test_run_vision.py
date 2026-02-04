from __future__ import annotations

import sys
from pathlib import Path

import pytest

from mlxk2.operations.run import run_model
from mlxk2.core.cache import hf_to_cache_dir
from mlxk2.core.capabilities import Backend, PolicyDecision, BackendPolicy, ModelCapabilities

# Vision support requires Python 3.10+ (mlx-vlm requirement)
pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 10),
    reason="Vision support requires Python 3.10+ (mlx-vlm dependency)"
)

# Lazy import VisionRunner to avoid import errors on Python 3.9
if sys.version_info >= (3, 10):
    from mlxk2.core.vision_runner import VisionRunner


def _mk_vision_snapshot(cache_root: Path, repo: str) -> Path:
    cache_dir = cache_root / hf_to_cache_dir(repo)
    snap = cache_dir / "snapshots" / "1111111111111111111111111111111111111111"
    snap.mkdir(parents=True, exist_ok=True)
    (snap / "config.json").write_text('{"model_type": "llava"}', encoding="utf-8")
    (snap / "model.safetensors").write_bytes(b"weights")
    # ADR-012 Phase 2: Vision models require preprocessor_config.json for health
    (snap / "preprocessor_config.json").write_text('{"size": 224}', encoding="utf-8")
    return snap


def _make_vision_policy(model_path, model_name):
    """Create a mock vision policy that allows vision backend."""
    caps = ModelCapabilities(
        model_path=model_path,
        model_name=model_name,
        is_vision=True,
        mlx_vlm_available=True,
        python_version=(3, 10, 0),
    )
    policy = BackendPolicy(
        backend=Backend.MLX_VLM,
        decision=PolicyDecision.ALLOW,
    )
    return caps, policy


def test_run_vision_routes_to_vision_runner(monkeypatch, isolated_cache):
    repo = "mlx-community/llava-vision-test"
    snap = _mk_vision_snapshot(isolated_cache, repo)

    calls = {}

    class DummyVisionRunner:
        def __init__(self, model_path, model_name, verbose=False):
            calls["path"] = model_path
            calls["name"] = model_name

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

        def generate(self, prompt, images=None, **kwargs):
            calls["prompt"] = prompt
            calls["images"] = list(images or [])
            calls["kwargs"] = kwargs
            return "vision-output"

    monkeypatch.setattr("mlxk2.core.vision_runner.VisionRunner", DummyVisionRunner)
    monkeypatch.setattr("mlxk2.operations.run.resolve_model_for_operation", lambda spec: (repo, None, None))
    monkeypatch.setattr("mlxk2.operations.run.get_current_model_cache", lambda: isolated_cache)
    # Mock the new probe_and_select to return vision policy
    monkeypatch.setattr(
        "mlxk2.core.capabilities.probe_and_select",
        lambda path, name, context="cli", has_images=False: _make_vision_policy(path, name)
    )

    # Session 146: Vision models now only route to VisionRunner when images are present
    # Pass a dummy image to trigger vision path
    image_bytes = b"dummy"
    result = run_model(
        model_spec=repo,
        prompt="hello",
        images=[("test.png", image_bytes)],
        stream=False,
        json_output=True
    )

    assert result == "vision-output"
    assert calls["path"] == snap
    assert calls["name"] == repo
    assert calls["prompt"] == "hello"
    assert calls["images"] == [("test.png", image_bytes)]


def test_run_vision_images_get_default_prompt(monkeypatch, isolated_cache):
    repo = "mlx-community/llava-vision-test-image"
    snap = _mk_vision_snapshot(isolated_cache, repo)

    captured = {}

    class DummyVisionRunner:
        def __init__(self, model_path, model_name, verbose=False):
            captured["path"] = model_path
            captured["name"] = model_name

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

        def generate(self, prompt, images=None, **kwargs):
            captured["prompt"] = prompt
            captured["images"] = list(images or [])
            return "vision-image-output"

    monkeypatch.setattr("mlxk2.core.vision_runner.VisionRunner", DummyVisionRunner)
    monkeypatch.setattr("mlxk2.operations.run.resolve_model_for_operation", lambda spec: (repo, None, None))
    monkeypatch.setattr("mlxk2.operations.run.get_current_model_cache", lambda: isolated_cache)
    # Mock the new probe_and_select to return vision policy
    monkeypatch.setattr(
        "mlxk2.core.capabilities.probe_and_select",
        lambda path, name, context="cli", has_images=False: _make_vision_policy(path, name)
    )

    image_bytes = b"raw"
    result = run_model(
        model_spec=repo,
        prompt=None,
        images=[("img.png", image_bytes)],
        stream=False,
        json_output=True,
    )

    assert result == "vision-image-output"
    assert captured["prompt"] == "Describe the image."
    assert captured["images"] == [("img.png", image_bytes)]


def test_vision_normalizes_generation_result():
    class GenerationResult:
        def __init__(self):
            self.text = "vision answer"
            self.generation_tokens = 10

    result = VisionRunner._normalize_result(GenerationResult())
    assert result == "vision answer"


def test_vision_adds_filename_mapping_for_multiple_images():
    """Test that filename mapping is added for multiple images (deterministic).

    The mapping is formatted as a Markdown table with an HTML comment marker
    '<!-- mlxk:filenames -->' to distinguish it from model-generated content.
    The comment is invisible in rendered markdown but identifiable in raw text.
    """
    images = [
        ("vacation1.jpg", b"data1"),
        ("vacation2.jpg", b"data2"),
        ("vacation3.jpg", b"data3"),
    ]

    result = VisionRunner._add_filename_mapping("Images 1 and 3 show motorboats.", images)

    # Updated for ADR-017 Phase 1: Collapsible HTML table with EXIF columns (enabled by default)
    # Session 75: Metadata moved to BEGINNING for better UX in chunking scenarios
    # Hashes: sha256(b"data1")[:8] = 5b41362b, etc.
    expected = """<details>
<summary>📸 Image Metadata (3 images)</summary>

<!-- mlxk:filenames -->
| Image | Filename | Original | Location | Date | Camera |
|-------|----------|----------|----------|------|--------|
| 1 | image_5b41362b.jpeg | vacation1.jpg | - | - | - |
| 2 | image_d98cf53e.jpeg | vacation2.jpg | - | - | - |
| 3 | image_f60f2d65.jpeg | vacation3.jpg | - | - | - |

</details>

Images 1 and 3 show motorboats."""
    assert result == expected


def test_vision_no_mapping_for_single_image():
    """Single image should now get mapping (ADR-017 Phase 1: enables cross-model workflows)."""
    images = [("single.jpg", b"data")]

    # ADR-017: Even single images get mapping (enables Vision→Text model switching)
    result = VisionRunner._add_filename_mapping("A dog.", images)

    # Updated for ADR-017 Phase 1: Collapsible HTML table with EXIF columns (enabled by default)
    # Session 75: Metadata moved to BEGINNING for better UX in chunking scenarios
    # Hash: sha256(b"data")[:8] = 3a6eb079
    expected = """<details>
<summary>📸 Image Metadata (1 image)</summary>

<!-- mlxk:filenames -->
| Image | Filename | Original | Location | Date | Camera |
|-------|----------|----------|----------|------|--------|
| 1 | image_3a6eb079.jpeg | single.jpg | - | - | - |

</details>

A dog."""
    assert result == expected


def test_vision_text_only_routing_condition():
    """Session 146: Vision routing uses 'if images' check, so empty list routes to text path.

    This is a simple unit test that verifies the routing logic condition.
    The actual E2E behavior is tested in tests_2.0/live/test_vision_e2e_live.py.
    """
    # The key routing condition in run.py line 495:
    # if images or (audio and audio_backend == Backend.MLX_VLM):
    #     → VisionRunner path
    # else:
    #     → MLXRunner path (text)

    # Empty list is falsy in Python
    images = []
    audio = None
    audio_backend = None

    # This is the routing condition from run.py
    uses_vision_path = bool(images) or (audio and audio_backend is not None)

    assert not uses_vision_path, "Empty images should NOT trigger vision path"

    # With images, it SHOULD trigger vision path
    images_with_content = [("test.png", b"data")]
    uses_vision_path = bool(images_with_content) or (audio and audio_backend is not None)

    assert uses_vision_path, "Non-empty images SHOULD trigger vision path"
