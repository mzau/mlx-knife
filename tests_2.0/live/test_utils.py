"""Shared utilities for live E2E tests (ADR-011).

Provides:
- Portfolio discovery functions (reused from test_stop_tokens_live.py)
- RAM gating utilities
- Common test constants
"""

from __future__ import annotations

import math
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Dict, Any, Tuple

# Import portfolio discovery infrastructure from test_stop_tokens_live.py
_parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(_parent_dir))

try:
    from test_stop_tokens_live import (
        apply_cache_wins_workspace_fallback,
        discover_mlx_models_in_user_cache,
        get_safe_ram_budget_gb,
        get_system_ram_gb,
        should_skip_model,
        TEST_MODELS,
    )
finally:
    sys.path.remove(str(_parent_dir))


# =============================================================================
# KNOWN BROKEN MODELS - per-capability runtime exclusions
# =============================================================================
# These models pass static health checks (files present, config valid) but fail
# at runtime on at least one capability. They are excluded from the portfolio
# of that capability only — a model whose text path is broken still belongs in
# the vision portfolio if its vision path was measured working.
#
# Policy: Add a model here ONLY when all four hold:
#   1. Static health check passes (healthy files)
#   2. mlxk's own verdict does NOT already exclude it (a `runtime_compatible:
#      false` model never reaches discovery, so an entry for it is dead weight)
#   3. The failure is reproduced against the CURRENT pin set, and the entry
#      records that measurement as a CONDITION — not as an upstream issue
#      number. An issue can close while the behaviour stays; the condition is
#      the whole claim (same idiom as mlxk2/operations/common.py's audio gates)
#   4. The `breaks` set names every capability that fails, and only those
#
# Format: cache ids are `org/name` and match exactly — org matters. A bare
# basename additionally matches every cache id and workspace path with that
# name; is_known_broken() strips the basename off the QUERY, never off the
# ENTRY, so a full `org/name` entry can never match a workspace path.
# CAUTION: use a bare basename only when no fixed same-name conversion exists.
#
# SSOT trajectory (Issue #53): this list compensates for runtime_compatible
# false-positives. Discovery already filters on mlxk's healthy+runnable
# verdict — once #53 param-reconciliation makes that verdict honest,
# load-failure entries here become redundant and should be dropped. Only
# execution-time failures (e.g., non-terminating forwards) stay list-worthy.
# =============================================================================

# The capability axes a portfolio is built for. Tokens are mlxk's own
# vocabulary (source: mlxk2/core/capabilities.py, class Capability) but are
# repeated here on purpose: these files reach mlxk only through the CLI/JSON
# boundary, so the oracle must not import the system under test.
# "text-generation" is deliberately absent — one axis, one name.
BROKEN_CAPABILITIES = frozenset({"chat", "vision", "audio"})

KNOWN_BROKEN_MODELS: Dict[str, Dict[str, Any]] = {
    # Checkpoint that corrupts stdout on the text path, so `run --json` returns
    # output no JSON parser accepts. Generation itself is fine.
    #
    # Condition (measured 2026-09-12; transformers 5.14.1 / mlx-lm 0.31.3):
    # config.json declares `auto_map`, so transformers prints its remote-code
    # consent prompt ("Do you wish to run the custom code? [y/N]") to STDOUT
    # while resolving the config — ahead of mlxk's JSON envelope. The envelope
    # that follows is complete and correct; it is simply no longer the first
    # byte, and json.loads() fails at column 1.
    #
    # It does NOT hang, which is what the pre-2026-09 entry claimed: measured
    # exit 0 in 16.6s with stdin at EOF and in 10.5s with an open, unwritten
    # pipe, answering correctly both times. No model-supplied code executes
    # either — AutoConfig refuses at EOF and mlx-lm loads its own Klear module.
    # Only the JSON contract breaks (test_run_json_output), so: "chat" only.
    #
    # The condition holds for ANY checkpoint whose load makes a library write
    # to stdout; the deeper defect is that mlxk's --json path does not fence
    # its own stdout. Excluding one model does not fix that — see the §T7
    # follow-up note. Recheck this entry when that path is hardened.
    "mlx-community/Klear-46B-A2.5B-Instruct-3bit": {
        "breaks": {"chat"},
        "condition": (
            "transformers prints its auto_map remote-code consent prompt to "
            "stdout during load, ahead of the JSON envelope, so `run --json` "
            "emits unparseable output; generation itself succeeds"
        ),
    },

    # Multimodal checkpoint whose TEXT path alone is broken. Vision and audio
    # load and answer correctly, so it belongs in the vision portfolio — this
    # entry must never widen past "chat".
    #
    # Condition (measured 2026-09-12; mlx-vlm 0.6.10 / mlx-lm 0.31.3 /
    # transformers 5.14.1 / mlx 0.32.0): a text-only `run` exits 1 in ~3s with
    #   "Received 126 parameters not in model: language_model.model.layers.24
    #    .self_attn.k_norm.weight, ...k_proj.{biases,scales,weight}, ...v_proj.*"
    # (126 parameters, layers 24-41). The weights live under `language_model.*`;
    # mlx-lm's text loader is handed a layout it cannot map. The same checkpoint
    # answers correctly through mlx-vlm with --image and with --audio.
    #
    # Not a version peg: every measured cell is identical on mlx-vlm 0.6.8 and
    # 0.6.10. The condition holds as long as mlx-lm's text loader receives a
    # checkpoint whose weights sit under `language_model.*`.
    #
    # Retires with Issue #53, not with a release: once runtime_compatible is
    # honest on the text axis, discovery drops this model on its own and the
    # entry becomes redundant (see TESTING-DETAILS.md, known-broken exclusion).
    #
    # Bare-basename entry — matches the workspace copy and any cache twin.
    "gemma-4-e4b-it-4bit": {
        "breaks": {"chat"},
        "condition": (
            "mlx-lm text load raises 'Received 126 parameters not in model' "
            "(layers 24-41, language_model.*.self_attn.{k,v}_proj); the same "
            "checkpoint answers correctly via mlx-vlm with --image / --audio"
        ),
    },
}


# Policy guard: a typo in `breaks` would silently disable an exclusion, and an
# entry without a condition is exactly the issue-number rationale this list was
# rewritten to remove. Both are cheap to catch at import.
for _model_id, _entry in KNOWN_BROKEN_MODELS.items():
    assert set(_entry["breaks"]) <= BROKEN_CAPABILITIES, (
        f"{_model_id}: unknown capability in breaks={_entry['breaks']!r}"
    )
    assert _entry["condition"].strip(), f"{_model_id}: a measured condition is required"
del _model_id, _entry


def is_known_broken(model_id: str, capability: str) -> bool:
    """True if `model_id` is measured broken on `capability`.

    `capability` is required on purpose. A default would let any call site that
    was not migrated keep the old behaviour — excluding a model from every
    portfolio at once — and that silent, global exclusion is the defect this
    list was rewritten to remove.

    An `org/name` entry matches that cache id exactly. A bare-basename entry
    additionally matches every cache id and workspace path with that basename;
    the basename is stripped off the query, never off the entry.
    """
    if capability not in BROKEN_CAPABILITIES:
        raise ValueError(
            f"unknown capability {capability!r}; "
            f"expected one of {sorted(BROKEN_CAPABILITIES)}"
        )
    entry = (
        KNOWN_BROKEN_MODELS.get(model_id)
        or KNOWN_BROKEN_MODELS.get(model_id.rsplit("/", 1)[-1])
    )
    return entry is not None and capability in entry["breaks"]


# RAM calculation utilities (modularized for different model types)

def calculate_text_model_ram_gb(size_bytes: int) -> float:
    """Calculate RAM requirement for text-only models.

    Text models use 1.2x overhead for inference based on empirical testing.
    This accounts for model weights + KV cache + temporary buffers.

    Args:
        size_bytes: Model size in bytes (from mlxk list --json)

    Returns:
        Estimated RAM needed in GB

    References:
        - ADR-009 (Stop Token Portfolio Discovery)
        - ADR-011 (E2E Test Architecture)
        - TESTING-DETAILS.md "RAM-Aware Model Selection"
    """
    return (size_bytes / (1024**3)) * 1.2


def calculate_vision_model_ram_gb(size_bytes: int, system_memory_bytes: int) -> float:
    """Calculate RAM requirement for vision models.

    Vision models have different memory characteristics:
    - Vision Encoder adds significant overhead (varies by model)
    - Metal OOM crashes occur above 70% system memory (ADR-016)
    - No simple multiplier - use direct threshold check

    Args:
        size_bytes: Model size in bytes (from mlxk list --json)
        system_memory_bytes: Total system RAM in bytes

    Returns:
        Estimated RAM needed in GB. Returns float('inf') if model exceeds
        70% system memory threshold (signals: skip this model).

    References:
        - ADR-012 (Vision Support Roadmap)
        - ADR-016 (Memory-Aware Model Loading)
        - capabilities.py MEMORY_THRESHOLD_PERCENT (imported, not mirrored)
    """
    from mlxk2.core.capabilities import MEMORY_THRESHOLD_PERCENT

    if system_memory_bytes == 0:
        return float('inf')  # Cannot determine, skip

    memory_ratio = size_bytes / system_memory_bytes

    # Vision models crash above the threshold due to Vision Encoder overhead
    if memory_ratio > MEMORY_THRESHOLD_PERCENT:
        return float('inf')  # Signal: Too large, will be skipped

    # Return actual size (no 1.2x multiplier for vision)
    # Encoder overhead is handled by the conservative threshold
    return size_bytes / (1024**3)


def get_system_memory_bytes() -> int:
    """Get total system memory in bytes via sysctl (macOS).

    Returns:
        Total physical RAM in bytes, or 0 if unavailable.
    """
    import subprocess

    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return int(result.stdout.strip())
    except (subprocess.SubprocessError, ValueError, FileNotFoundError):
        pass

    return 0


def parse_vm_stat_page_size(output: str) -> int:
    """Extract vm_stat page size in bytes, falling back to 4096."""
    match = re.search(r"page size of (\d+) bytes", output)
    if match:
        return int(match.group(1))
    return 4096


# =============================================================================
# TIME LIMITS - staged by model size
# =============================================================================
# Every wall-clock limit in this suite is a HANG GUARD, not a performance
# budget. The runaway it was originally built for is structurally excluded
# since max_tokens is bounded (#66), and the performance signal lives in the
# benchmark stream (duration + size_gb). What a limit still has to catch is a
# real hang, and a real hang is unbounded - not merely slow. So being generous
# costs nothing, while being tight costs a red line nobody can attribute.
#
# A limit has two terms and they must not be confused:
#   base  - the WORK the row does (generate MAX_TOKENS, transcribe 30 minutes
#           of audio). A property of the test, not of the model. It stays
#           whatever the call site already says, and it is the LOWER BOUND.
#   tier  - the part that scales with the MODEL: what it costs to get its bytes
#           resident when the suite has already filled memory. That is the term
#           this table adds.
#
# Measured 2026-09-12 (wet umbrella + benchmark stream; mlx 0.32.0, mlx-lm
# 0.31.3, mlx-vlm 0.6.10, transformers 5.14.1) on GLM-4.7-Flash-8bit, 29.65 GB
# on disk - the same CLI run under two memory conditions:
#     relaxed machine    30 s
#     under the umbrella 135 s   (102 s sys, 9.5 s user, GPU 1.6 %; ram_free at
#                                 0.0 GB and vm_pressure ~22000 - 29.7 GB being
#                                 made resident in ~11 GB of free memory)
# => 105 s of load penalty over 29.65 GB = 0.28 GB/s. The base (90 s there)
# already covers the relaxed run in full, so the tier covers the penalty and
# nothing else.
#
# WHAT THE DRIVER IS NOT (measured 2026-09-13, and the earlier note here said
# the opposite): it is not a cold file cache. `sudo purge` followed by the same
# row took 16.3 s - FASTER than the 30 s above, because purge frees memory
# rather than exhausting it. The expensive condition is memory PRESSURE built up
# by the preceding tests, which is why this row is only reproducible inside the
# umbrella and not in an isolated run.
#
# Control from the same stream: the largest duration below 8 GB is 145.5 s - a
# 0.82 GB Whisper translating 30 minutes of German. Pure work, zero size
# component. A size-staged limit must not try to explain that; the base does.
#
# Each tier is the load penalty at its UPPER BOUND plus roughly 25 % reserve.
LOAD_PENALTY_RATE_GB_S = 0.3
TIMEOUT_RESERVE = 1.3

LOAD_ALLOWANCE_S = (
    (8.0, 40),    # small  - up to  8 GB (load penalty  28 s, reserve 1.41x)
    (20.0, 90),   # medium - up to 20 GB (load penalty  71 s, reserve 1.27x)
    (None, 140),  # large  - beyond      (load penalty 113 s at the 32 GB text
                  #                       gate, reserve 1.24x)
)

# Size unresolvable: what cannot be bounded gets the widest tier. Happens when
# `mlxk list --json` fails or the model is not in the portfolio at all.
UNKNOWN_SIZE_ALLOWANCE_S = 140


@lru_cache(maxsize=1)
def _model_size_index() -> Dict[str, float]:
    """Map model name -> size on disk in GB, from `mlxk list --json`.

    One subprocess per pytest process. Keys are stored twice, under the full
    name AND under the basename: workspace models arrive as an absolute path,
    cache models as `org/name`, and a call site may hold either form. Same
    matching idiom as is_known_broken(), for the same reason.

    Never raises. A broken index means generous limits, which is the safe
    direction for a hang guard - the suite must still run.
    """
    import json
    import os
    import subprocess

    index: Dict[str, float] = {}
    try:
        result = subprocess.run(
            [sys.executable, "-m", "mlxk2.cli", "list", "--json"],
            capture_output=True,
            text=True,
            timeout=30,
            env=os.environ.copy(),
        )
        if result.returncode != 0:
            return index
        models = json.loads(result.stdout).get("data", {}).get("models", [])
    except Exception:
        return index

    for model in models:
        name = model.get("name")
        size_bytes = model.get("size_bytes") or 0
        if not name or not size_bytes:
            continue
        size_gb = size_bytes / (1024**3)
        index[name] = size_gb
        # Basename collision (a cache copy and a workspace clone of the same
        # name): keep the larger, so the guard errs wide rather than narrow.
        basename = name.rsplit("/", 1)[-1]
        index[basename] = max(index.get(basename, 0.0), size_gb)

    return index


def model_size_gb(model_id: str | None) -> float | None:
    """Size on disk in GB for `model_id`, or None if it cannot be resolved.

    Deliberately reads the index rather than the portfolio dicts: their
    `ram_needed_gb` is 1.2x disk on the text axis and `inf` above the 70 %
    threshold on the vision axis, so it is not a size. It also serves the call
    sites that hold no portfolio dict at all (a hardcoded model constant, a
    fixture that returns only the id).
    """
    if not model_id:
        return None
    index = _model_size_index()
    if model_id in index:
        return index[model_id]
    return index.get(model_id.rsplit("/", 1)[-1])


def size_allowance_s(size_gb: float | None) -> int:
    """Seconds to add on top of a call site's base limit for a model this size."""
    if size_gb is None:
        return UNKNOWN_SIZE_ALLOWANCE_S

    allowance = LOAD_ALLOWANCE_S[-1][1]
    for bound, tier in LOAD_ALLOWANCE_S:
        if bound is None or size_gb <= bound:
            allowance = tier
            break

    # The table is calibrated against the text RAM gate (~32 GB on disk). The
    # vision gate sits higher (0.70 x system RAM), so a checkpoint past the last
    # bound would get LESS than the measured rate predicts - precisely the
    # defect this staging removes. Above the table the rate takes over; below
    # it the tier always wins by construction (35<=40, 87<=90, 139<=140).
    return max(allowance, math.ceil(size_gb / LOAD_PENALTY_RATE_GB_S * TIMEOUT_RESERVE))


def model_timeout(base_s: float, model_id: str | None) -> float:
    """A call site's base limit widened by what a model of this size costs.

    `base_s` stays the lower bound - it encodes the work, so nothing ever gets
    tighter than it is today.
    """
    return base_s + size_allowance_s(model_size_gb(model_id))


def discover_text_models() -> list[Dict[str, Any]]:
    """Discover text-only models (filter out Vision and Audio models).

    Uses discover_mlx_models_in_user_cache() and filters out models
    with "vision" or "audio" in their capabilities list.

    This enables deterministic text-only test portfolios that won't
    change when Vision or Audio models are added/removed from cache.

    Chat-broken models are dropped here, on every return path: the shared base
    discovery carries no test-tree policy, because it also feeds the vision
    axis, where a chat break is irrelevant.

    Returns:
        List of text-only model dicts (same format as discover_mlx_models_in_user_cache):
        [{"model_id": "...", "ram_needed_gb": X.X, "snapshot_path": None, "weight_count": None}, ...]
    """
    import json
    import subprocess
    import os

    # Get all discovered models (already filtered: MLX + healthy + runtime_compatible + chat)
    all_models = discover_mlx_models_in_user_cache()
    if not all_models:
        return []

    # The fallbacks below must apply this too — a transient `mlxk list` failure
    # must not silently restore an unfiltered portfolio.
    runnable = [m for m in all_models if not is_known_broken(m["model_id"], "chat")]

    # Get capabilities from mlxk list --json
    env = os.environ.copy()

    try:
        result = subprocess.run(
            [sys.executable, "-m", "mlxk2.cli", "list", "--json"],
            capture_output=True,
            text=True,
            timeout=30,
            env=env
        )

        if result.returncode != 0:
            return runnable  # Fall back to all models minus chat-broken ones

        # Parse JSON and build vision model ID set
        data = json.loads(result.stdout)
        models = data.get("data", {}).get("models", [])

        # Filter out vision AND audio models (text-only portfolio)
        non_text_model_ids = {
            m["name"] for m in models
            if "vision" in m.get("capabilities", []) or "audio" in m.get("capabilities", [])
        }

        # Filter out vision and audio models
        return [m for m in runnable if m["model_id"] not in non_text_model_ids]

    except Exception:
        return runnable  # Fall back to all models minus chat-broken ones


def discover_vision_models() -> list[Dict[str, Any]]:
    """Discover vision-capable models only.

    Uses discover_mlx_models_in_user_cache() and filters to only models
    with "vision" in their capabilities list.

    IMPORTANT: Recalculates RAM requirements using Vision-specific formula
    (ADR-016 0.70 threshold instead of 1.2x multiplier).

    This enables deterministic vision-only test portfolios separate from
    text-only tests.

    Returns:
        List of vision-capable model dicts (same format as discover_mlx_models_in_user_cache):
        [{"model_id": "...", "ram_needed_gb": X.X, "snapshot_path": None, "weight_count": None}, ...]
    """
    import json
    import subprocess
    import os

    # Get all discovered models (already filtered: MLX + healthy + runtime_compatible + chat)
    all_models = discover_mlx_models_in_user_cache()
    if not all_models:
        return []

    # Get capabilities and size_bytes from mlxk list --json
    env = os.environ.copy()

    try:
        result = subprocess.run(
            [sys.executable, "-m", "mlxk2.cli", "list", "--json"],
            capture_output=True,
            text=True,
            timeout=30,
            env=env
        )

        if result.returncode != 0:
            return []

        # Parse JSON and build vision model data
        data = json.loads(result.stdout)
        models_list = data.get("data", {}).get("models", [])

        # Build map: model_id -> (is_vision, size_bytes)
        model_info = {}
        for m in models_list:
            model_name = m["name"]
            is_vision = "vision" in m.get("capabilities", [])
            size_bytes = m.get("size_bytes", 0)
            model_info[model_name] = (is_vision, size_bytes)

        # Get system memory for vision RAM calculation
        system_memory_bytes = get_system_memory_bytes()

        # Filter to only vision models + recalculate RAM
        vision_models = []
        for model in all_models:
            model_id = model["model_id"]

            # Skip models measured broken on the vision axis. A break on
            # another axis (e.g. a multimodal checkpoint whose text loader
            # fails) does not belong here and must not cost vision coverage.
            if is_known_broken(model_id, "vision"):
                continue

            if model_id in model_info:
                is_vision, size_bytes = model_info[model_id]
                if is_vision:
                    # Recalculate RAM using Vision-specific formula
                    ram_gb = calculate_vision_model_ram_gb(size_bytes, system_memory_bytes)

                    # Create new dict with updated RAM
                    vision_model = model.copy()
                    vision_model["ram_needed_gb"] = ram_gb
                    vision_models.append(vision_model)

        return vision_models

    except Exception:
        return []


def discover_audio_models() -> list[Dict[str, Any]]:
    """Discover audio-capable models only (ADR-020).

    Queries mlxk list --json directly and filters for:
    - model_type == "audio" (STT-only models: Whisper, Voxtral)
    - framework == "MLX" and health == "healthy" and runtime_compatible

    Note: This does NOT use discover_mlx_models_in_user_cache() because
    audio models have model_type="audio", not model_type="chat".

    Returns:
        List of audio model dicts:
        [{"model_id": "...", "ram_needed_gb": X.X, "repo_id": "...", ...}, ...]
    """
    import json
    import subprocess
    import os

    env = os.environ.copy()
    if not env.get("HF_HOME"):
        return []  # Audio discovery requires HF_HOME (see TESTING.md)

    try:
        result = subprocess.run(
            [sys.executable, "-m", "mlxk2.cli", "list", "--json"],
            capture_output=True,
            text=True,
            timeout=30,
            env=env
        )

        if result.returncode != 0:
            return []

        data = json.loads(result.stdout)
        # Apply cache-wins workspace-fallback dedup so workspace-only audio
        # models (e.g. Whisper clones not present in the HF cache) become
        # discoverable (ADR-022 migration-ready).
        models_list = apply_cache_wins_workspace_fallback(
            data.get("data", {}).get("models", [])
        )

        # Get system memory for RAM calculation
        system_memory_bytes = get_system_memory_bytes()

        audio_models = []
        for m in models_list:
            # Filter: MLX + healthy + runtime_compatible + audio model_type
            if (m.get("framework") == "MLX" and
                m.get("health") == "healthy" and
                m.get("runtime_compatible") is True and
                m.get("model_type") == "audio"):

                model_name = m["name"]

                # Skip models measured broken on the audio axis
                if is_known_broken(model_name, "audio"):
                    continue

                # Calculate RAM using vision formula (conservative)
                size_bytes = m.get("size_bytes", 0)
                ram_gb = calculate_vision_model_ram_gb(size_bytes, system_memory_bytes)

                audio_models.append({
                    "model_id": model_name,
                    "repo_id": model_name,
                    "ram_needed_gb": ram_gb,
                    "snapshot_path": None,
                    "weight_count": None,
                })

        return audio_models

    except Exception:
        return []


# =============================================================================
# FALLBACK TEST MODELS - Minimum Required Models for Testing Without HF_HOME
# =============================================================================
# When HF_HOME is not set, Portfolio Discovery returns []. These fallback models
# provide a baseline for testing when the user has these specific models in
# their default cache (~/.cache/huggingface).
#
# These models must be downloaded manually if testing without HF_HOME:
#   mlxk pull mlx-community/gpt-oss-20b-MXFP4-Q8
#   mlxk pull mlx-community/Qwen2.5-0.5B-Instruct-4bit
#   mlxk pull mlx-community/Llama-3.2-3B-Instruct-4bit
#   mlxk pull mlx-community/pixtral-12b-4bit
#   mlxk pull mlx-community/whisper-large-v3-turbo-4bit
# =============================================================================

# Vision fallback model (for tests without HF_HOME)
VISION_TEST_MODELS = {
    "pixtral": {
        "id": "mlx-community/pixtral-12b-4bit",
        "expected_issue": None,
        "description": "Pixtral 12B - general-purpose vision model",
        "ram_needed_gb": 7.0  # 12B 4-bit (~7GB empirical)
    }
}

# Audio fallback model (for tests without HF_HOME)
AUDIO_TEST_MODELS = {
    "whisper": {
        "id": "mlx-community/whisper-large-v3-turbo-4bit",
        "expected_issue": None,
        "description": "Whisper large-v3-turbo - STT baseline",
        "ram_needed_gb": 1.5  # Large-v3 4-bit (~1.5GB)
    }
}

# Embedding test models (ADR-015, alpha-gated — set MLXK2_ENABLE_ALPHA_FEATURES=1).
# Verified-runnable REPRESENTATIVES for 2.0.7 — one per code path / pooling branch. This is a
# test-fixture set, NOT a capability claim and NOT a closed allowlist: the runnable predicate is
# class-level (any model_type qwen3/bert embedder is attempted), so the public "which classes are
# verified" statement lives in docs/MODEL-COVERAGE.md (ADR-023), not here. `id` resolves against
# the portfolio (HF cache + MLXK_WORKSPACE_HOME); tests skip per-model when absent.
EMBED_TEST_MODELS = {
    "qwen3-decoder": {
        "id": "Qwen3-Embedding-0.6B-4bit-DWQ",
        "kind": "decoder",
        "pooling": "last_token",
        "dims": 1024,
        "expected_issue": None,
        "description": "Qwen3-Embedding 0.6B 4bit-DWQ — decoder path (mlx-lm), showcase (workspace)",
    },
    "bge-encoder-cls": {
        "id": "bge-small-en-v1.5-4bit",
        "kind": "encoder",
        "pooling": "cls",
        "dims": 384,
        "expected_issue": None,
        "description": "bge-small-en-v1.5 4bit — encoder CLS pool, 4-bit quantized, WordPiece (cache)",
    },
    "e5-encoder-mean": {
        "id": "multilingual-e5-small-mlx",
        "kind": "encoder",
        "pooling": "mean",
        "dims": 384,
        "expected_issue": None,
        "description": "multilingual-e5-small — encoder mean pool, float16, XLM-R sentencepiece (workspace)",
    },
}


# Re-export for convenience
__all__ = [
    "discover_mlx_models_in_user_cache",
    "discover_text_models",
    "discover_vision_models",
    "discover_audio_models",
    "calculate_text_model_ram_gb",
    "calculate_vision_model_ram_gb",
    "model_size_gb",
    "size_allowance_s",
    "model_timeout",
    "LOAD_ALLOWANCE_S",
    "UNKNOWN_SIZE_ALLOWANCE_S",
    "get_system_memory_bytes",
    "get_safe_ram_budget_gb",
    "get_system_ram_gb",
    "should_skip_model",
    "is_known_broken",
    "KNOWN_BROKEN_MODELS",
    "BROKEN_CAPABILITIES",
    "TEST_MODELS",
    "VISION_TEST_MODELS",
    "AUDIO_TEST_MODELS",
    "EMBED_TEST_MODELS",
    "TEST_PROMPT",
    "MAX_TOKENS",
    "TEST_TEMPERATURE",
]


# Standard test constants (shared across all E2E tests)
TEST_PROMPT = "Write one sentence about cats."
MAX_TOKENS = 50
TEST_TEMPERATURE = 0.0  # Deterministic sampling for reproducible tests
