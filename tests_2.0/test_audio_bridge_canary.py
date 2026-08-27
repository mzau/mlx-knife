"""Canary for the mlx-audio#645 bridge (ADR-023).

A failure here is **not** a defect in mlx-knife. It means upstream mlx-audio
moved: either the bridge in `mlxk2/core/audio_runner.py` has become redundant,
or the contract it serves has drifted. Read the assertion that failed and
decide whether the bridge retires — that decision is the whole point of this
file, and it replaces the version gate that used to guess at it.

Deliberately source-based. Importing upstream's Whisper module would pull in
the real `mlx.core`, which TESTING-DETAILS forbids inside the stub-collecting
tree (in-process real MLX there reintroduces the nanobind abort). Reading the
files as text keeps this a plain unit test: no mlx, no model, no network.
"""

import ast
import importlib.util
import re
from pathlib import Path

import pytest

WHISPER = "stt/models/whisper"


def _upstream_source(relative_path: str) -> str:
    """Read a file out of the installed mlx-audio without importing it."""
    spec = importlib.util.find_spec("mlx_audio")
    if spec is None or not spec.origin:
        pytest.skip("mlx-audio not installed")
    path = Path(spec.origin).parent / relative_path
    if not path.exists():
        pytest.fail(f"mlx-audio no longer ships {relative_path} — re-read the bridge")
    return path.read_text(encoding="utf-8")


def _definition(source: str, *names: str) -> str:
    """Source of a nested definition, e.g. ('Model', 'get_tokenizer')."""
    node = ast.parse(source)
    for name in names:
        matches = [
            child
            for child in ast.iter_child_nodes(node)
            if isinstance(child, (ast.FunctionDef, ast.ClassDef)) and child.name == name
        ]
        if not matches:
            pytest.fail(f"upstream no longer defines {'.'.join(names)} — re-read the bridge")
        node = matches[0]
    return ast.get_source_segment(source, node) or ""


def test_upstream_get_tokenizer_still_needs_a_processor():
    """If it stops raising, upstream serves processor-less repos again."""
    body = _definition(_upstream_source(f"{WHISPER}/whisper.py"), "Model", "get_tokenizer")

    assert "_processor" in body and "raise ValueError" in body, (
        "Model.get_tokenizer no longer refuses when there is no HuggingFace "
        "processor — the get_tokenizer bridge may be redundant"
    )


def test_upstream_ships_no_tokenizer_factory():
    """If the factory returns, the vendored copy in mlxk2/audio/ can go."""
    source = _upstream_source(f"{WHISPER}/tokenizer.py")
    defined = {
        node.name
        for node in ast.iter_child_nodes(ast.parse(source))
        if isinstance(node, ast.FunctionDef)
    }

    assert "get_tokenizer" not in defined, (
        "upstream restored the get_tokenizer factory dropped in 0.3.1 — "
        "retire mlxk2/audio/whisper_tokenizer.py"
    )


def test_upstream_post_load_hook_has_no_processor_fallback():
    """If a canonical fallback appears, #712 landed and _processor gets filled."""
    body = _definition(_upstream_source(f"{WHISPER}/whisper.py"), "Model", "post_load_hook")

    assert "WhisperProcessor.from_pretrained" in body, (
        "post_load_hook no longer loads a WhisperProcessor — re-read the bridge"
    )
    assert "openai/whisper" not in body, (
        "post_load_hook gained a canonical openai/whisper-* fallback (#712?) — "
        "repos shipping no processor may now work unaided"
    )


def test_vendored_tokenizer_satisfies_the_decoder_contract():
    """The bridge only helps while what it returns still fits upstream's decoder."""
    from mlxk2.audio.whisper_tokenizer import get_tokenizer

    decoder_source = _upstream_source(f"{WHISPER}/decoding.py")
    required = set(re.findall(r"tokenizer\.([a-z_]+)", decoder_source))
    assert required, "found no tokenizer attribute reads in decoding.py — probe is stale"

    tokenizer = get_tokenizer(True)
    missing = sorted(name for name in required if not hasattr(tokenizer, name))

    assert not missing, (
        f"upstream's decoder reads attributes the vendored tokenizer lacks: {missing}"
    )
