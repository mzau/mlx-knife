# MLX Knife Testing Guide

## Overview

MLX Knife uses a **3-category test strategy** designed for safety, speed, and reproducibility on Apple Silicon. Most tests run in complete isolation without requiring models or network access.

For current test counts, version-specific details, and complete file listings, see [TESTING-DETAILS.md](TESTING-DETAILS.md).

## Test Philosophy

**Core Principles:**
- **Isolated by default** - User cache stays pristine with sentinel protection
- **Opt-in live tests** - Network/model tests require explicit markers/environment
- **Mock-heavy** - MLX stubs enable fast testing without model downloads
- **Fast feedback** - 300+ tests run in seconds on any Apple Silicon Mac

**Safety First:**
- Tests use temporary caches with `TEST_SENTINEL` protection
- Delete operations fail if not in test cache (`MLXK2_STRICT_TEST_DELETE=1`)
- Live tests never modify user cache without explicit environment variables

## Quick Start

```bash
# Install package + development tools
pip install -e ".[dev,test]"

# Run default test suite (isolated, no live downloads)
pytest -v

# Before committing
ruff check mlxk2/ --fix && mypy mlxk2/ && pytest -v
```

**That's it!** Default tests use isolated caches and MLX stubs - no model downloads required.

## Test Categories

### Category 1: Isolated Cache (Default)
**User cache stays pristine** - Tests use temporary caches with sentinel protection

**What's tested:**
- JSON API contracts (list, show, health)
- Human output formatting
- Model resolution and naming
- Push operations (offline: `--check-only`, `--dry-run`)
- Clone operations (offline: APFS validation, CoW workflow)
- Run command and generation (with MLX stubs)
- Server API endpoints (minimal, no real models)
- Schema validation and spec compliance

**How to run:**
```bash
pytest -v  # Runs all isolated tests
```

**Technical pattern:**
```python
def test_something(isolated_cache):
    # Complete isolation with sentinel protection
    assert_is_test_cache(isolated_cache)
    # Test implementation
```

### Category 2: Live Tests (Opt-in)
**Require explicit environment setup** - Network or user cache dependent

**What's tested:**
- Real HuggingFace push operations
- APFS same-volume clone workflows
- Stop token validation with real models
- Framework detection with private/org models
- Multi-shard model health validation

**Markers:** `live_push`, `live_clone`, `live_list`, `live_stop_tokens`, `live_e2e`, `live_run`, `issue27`

**How to run:**
```bash
# Live stop tokens (requires models in cache or HF_HOME)
pytest -m live_stop_tokens -v

# Live push (requires credentials + workspace)
export MLXK2_ENABLE_ALPHA_FEATURES=1
export MLXK2_LIVE_PUSH=1
export HF_TOKEN=...
export MLXK2_LIVE_REPO=org/model
export MLXK2_LIVE_WORKSPACE=/path/to/workspace
pytest -m live_push -v
```

See [TESTING-DETAILS.md](TESTING-DETAILS.md) for complete environment setup instructions.

### Category 3: Server Tests (Default)
**Basic server functionality** - Lightweight API validation

**What's tested:**
- OpenAI-compatible endpoints
- SSE streaming functionality
- Model loading and error handling
- Token limit enforcement

**How to run:**
```bash
pytest -k server -v  # Optional, included in default suite
```

**Note:** Basic server tests use MLX stubs and run by default. Comprehensive E2E tests with real models are available via `live_e2e` marker (ADR-011).

## Test Structure

```
tests_2.0/
├── conftest.py              # Isolated cache, safety sentinel, core fixtures
├── conftest_runner.py       # Runner-specific fixtures/mocks
├── stubs/                   # Minimal MLX/MLX-LM stubs for unit tests
│   ├── mlx/core.py
│   └── mlx_lm/...
├── spec/                    # JSON API spec/contract validation
│   ├── test_cli_commands_json_flag.py
│   ├── test_spec_version_sync.py
│   └── ...
├── live/                    # Opt-in live tests (markers required)
│   ├── test_push_live.py
│   ├── test_clone_live.py
│   └── test_list_human_live.py
├── test_*.py               # Core test files
└── test_*.py.disabled      # Intentionally disabled (WIP)
```

**Legend:**
- `spec/` - API contract validation (stays in sync with `docs/schema`)
- `live/` - Opt-in tests requiring environment (markers: `live_*`)
- `stubs/` - Lightweight MLX replacements for unit tests
- `conftest.py` - Isolated HF cache (temp), safety sentinel, fixtures

See [TESTING-DETAILS.md](TESTING-DETAILS.md) for complete file listing with descriptions.

## MLX Stubs (Fast Testing Without Model Downloads)

**Purpose:** Unit tests run without loading real models

**How it works:**
- `conftest.py` prepends `tests_2.0/stubs/` to `sys.path`
- `import mlx` / `import mlx_lm` resolve to minimal stubs
- Tests use mock models (~50KB fake files instead of 50GB real models)

**Benefits:**
- Fast test runs (seconds instead of minutes)
- Low RAM usage (16GB sufficient)
- No model downloads required
- Deterministic behavior

**Limitations:**
- Tests requiring real mlx-lm integration use `@requires_mlx_lm` marker (2 tests)
- Production CLI/server still use real packages (stubs not installed)

## Common Test Commands

```bash
# Default suite (isolated, fast)
pytest -v

# Specific categories
pytest -m spec -v              # Only spec/schema tests
pytest -m "not spec" -v        # Exclude spec tests
pytest -k push -v              # Push tests (offline)
pytest -k server -v            # Server tests

# Live tests (opt-in)
pytest -m live_stop_tokens -v  # Stop token validation
pytest -m live_push -v         # Real HF push
pytest -m live_clone -v        # APFS clone workflow

# Development
pytest --durations=10          # Show slowest tests
pytest -k "test_name" -v       # Run specific test
```

## Test Prerequisites

### Required Setup
1. **Apple Silicon Mac (M1/M2/M3)** - Required (MLX uses Metal)
2. **Python 3.9 or newer**
3. **16GB RAM minimum**
4. **~10-20MB disk space** for test temp files
5. **Test dependencies:**
   ```bash
   pip install -e .[test]
   ```

**That's it!** Default tests use mock models - no HF cache or downloads needed.

### Optional Setup (Live Tests)

Live tests require additional environment setup:

**🔍 Show which models would be tested:**
```bash
HF_HOME=/path/to/cache pytest -m show_model_portfolio -s
```
This displays all models that would be used in E2E tests (no actual testing).

**E2E tests** (ADR-011):
```bash
# Full E2E test suite with real models
HF_HOME=/path/to/cache pytest -m live_e2e -v
```

**Stop token validation** (ADR-009):
```bash
# Option A: Portfolio Discovery (recommended)
export HF_HOME=/path/to/cache
pytest -m live_stop_tokens -v

# Option B: Hardcoded models (requires 3 specific models in cache)
# See TESTING-DETAILS.md for model list
```

**Push/Clone tests** (alpha features):
```bash
# See TESTING-DETAILS.md for complete environment setup
```

## Environment & Caches

**User cache** (persistent):
- Real cache for manual operations
- Example: `export HF_HOME="/Volumes/SSD/models"`
- Safe ops: `list`, `health`, `show`

**Test cache** (isolated):
- Ephemeral via fixtures
- Default tests never touch user cache
- Deletion safety: `MLXK2_STRICT_TEST_DELETE=1`

**Best practice:**
- Use isolated tests for development (default `pytest`)
- Use live tests for validation (opt-in with markers)
- Set `HF_HOME` to external SSD for live tests

## Python Version Compatibility

**All tests validated on Python 3.9-3.13**

Multi-version testing:
```bash
# Automated script
./test-multi-python.sh

# Manual verification
python3.9 -m venv test_39
source test_39/bin/activate
pip install -e .[test] && pytest
```

See [TESTING-DETAILS.md](TESTING-DETAILS.md) for version-specific results.

## Code Quality

```bash
# Install tools
pip install -e .[dev]

# Code formatting and linting
ruff check mlxk2/ --fix

# Type checking
mypy mlxk2/

# Complete workflow
ruff check mlxk2/ --fix && mypy mlxk2/ && pytest
```

## Test Markers

MLX Knife uses pytest markers to organize tests by category:

- **Default suite** (`pytest -v`): Unit tests with mocks (fast, offline, no real models)
- **Spec tests** (`-m spec`): API contract/schema validation
- **Live tests** (`-m live_*`): Tests with real models or network (opt-in)

**Common commands:**
```bash
# Default test suite (fast, offline)
pytest -v

# API spec/contract tests only
pytest -m spec -v

# Live tests with real models (examples)
pytest -m live_stop_tokens -v  # Stop token validation (ADR-009)
pytest -m live_e2e -v          # E2E server/HTTP/CLI tests (ADR-011)
```

**For complete marker reference, environment requirements, and detailed usage, see:**
- [TESTING-DETAILS.md → Test Execution Guide](TESTING-DETAILS.md#test-execution-guide)

**Symbol Legend:**
- 🔒 **Marker-required**: Must use `-m marker` (skipped by default `pytest -v`)
- **Skip-unless-env**: Collected but skipped without required environment

## Troubleshooting

**Tests hang forever:**
```bash
pytest --timeout=60
```

**Import errors:**
```bash
pip install -e .[test]
```

**Cache conflicts:**
```bash
export HF_HOME="/tmp/test_cache"
pytest --cache-clear
```

**Debug specific test:**
```bash
pytest path/to/test.py::test_name -v -s
```

## Contributing Tests

When submitting PRs with test changes, please include:

1. **Test environment:**
   - macOS version
   - Apple Silicon chip (M1/M2/M3/M4/M5)
   - Python version

2. **Test results** (example):
   ```
   Platform: macOS 14.5, M2 Pro
   Python: 3.9.6
   Results: 306 passed, 20 skipped
   ```

3. **Any issues encountered** and resolutions

## Development Workflow

Before committing:

```bash
# 1. Code style
ruff check mlxk2/ --fix

# 2. Type checking
mypy mlxk2/

# 3. Run tests
pytest -v

# Or combined
ruff check mlxk2/ --fix && mypy mlxk2/ && pytest -v
```

## Summary

**MLX Knife Testing:**
- ✅ **Isolated by default** - User cache stays pristine
- ✅ **Fast feedback** - 300+ tests run in seconds without model downloads
- ✅ **Low requirements** - 16GB RAM, ~20MB disk, no HF cache needed
- ✅ **Opt-in live tests** - Real models/network when needed
- ✅ **Multi-Python support** - Verified on Python 3.9-3.13

For detailed information including current test counts, complete file structure, version history, and implementation specifics, see [TESTING-DETAILS.md](TESTING-DETAILS.md).

---

*MLX-Knife 2.0 Testing Framework*
