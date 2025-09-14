# MLX Knife Testing Guide

## Current Status

✅ **166/166 tests passing** (September 2025) - **STABLE RELEASE 1.1.1** 🚀  
✅ **Apple Silicon verified** (M1/M2/M3)  
✅ **Python 3.9-3.13 compatible**  
✅ **Stable for development use** - comprehensive testing with real model execution
✅ **Isolated test system** - user cache stays pristine with temp cache isolation
✅ **3-category test strategy** - optimized for performance and safety

## Quick Start

```bash
# Install with test dependencies
pip install -e ".[test]"

# Download test model (optional - most tests use isolated cache)
mlxk pull mlx-community/Phi-3-mini-4k-instruct-4bit

# Run all tests
pytest

# Fast unit tests only
pytest tests/unit/

# Before committing
ruff check mlx_knife/ --fix && mypy mlx_knife/ && pytest
```

## Why Local Testing?

MLX Knife requires **Apple Silicon hardware** and **real MLX models** for comprehensive testing:

- **Hardware Requirement**: MLX framework only runs on Apple Silicon (M1/M2/M3)
- **Model Requirement**: Tests use actual models (4GB+) for realistic validation
- **Industry Standard**: Local testing is normal for MLX projects
- **Quality Assurance**: Real hardware testing ensures actual functionality

This approach ensures our tests reflect real-world usage, not mocked behavior.

## Test Structure

```
tests/
├── conftest.py                         # Shared fixtures and utilities
├── integration/                        # System-level integration tests (78 tests)
│   ├── test_core_functionality.py          # Basic CLI operations (isolated cache)
│   ├── test_health_checks.py               # Model corruption detection (isolated cache)
│   ├── test_lock_cleanup_bug.py            # Issue #23: Lock cleanup (isolated cache)
│   ├── test_process_lifecycle.py           # Process management (isolated cache)
│   ├── test_real_model_lifecycle.py        # Full model lifecycle (isolated cache)
│   ├── test_run_command_advanced.py        # Run command edge cases (isolated cache)
│   ├── test_server_functionality.py        # Server lifecycle tests
│   ├── test_end_token_issue.py             # Issue #20: End-token filtering (@server)
│   ├── test_issue_14.py                    # Issue #14: Chat self-conversation (@server)
│   └── test_issue_15_16.py                 # Issues #15/#16: Dynamic token limits (@server)
└── unit/                              # Module-level unit tests (88 tests)
    ├── test_cache_utils.py                 # Cache management & Issue #21/#23 tests
    ├── test_cli.py                         # CLI argument parsing
    ├── test_health_multishard.py           # Strict multi-shard/index health (Issue #27)
    ├── test_mlx_runner_memory.py           # Memory management tests
    └── test_model_card_detection.py        # Issue #31: README/tokenizer hints for framework/type
```


## 3-Category Test Strategy (MLX Knife 1.1.0+)

MLX Knife uses a **3-category test strategy** to balance test isolation, performance, and user cache protection:

### 🏠 CATEGORY 1: ISOLATED CACHE (Most Tests)
**✅ User cache stays pristine** - Tests use temporary isolated caches with automatic cleanup

**Implemented Tests (78 tests):**
- ✅ `test_real_model_lifecycle.py` - Full model lifecycle with `tiny-random-gpt2` (~12MB download)
- ✅ `test_core_functionality.py` - Basic CLI operations with `patch_model_cache` isolation  
- ✅ `test_process_lifecycle.py` - Process management with isolated cache + MODEL_CACHE patching
- ✅ `test_run_command_advanced.py` - Run command edge cases with `mock_model_cache` in isolation
- ✅ `test_lock_cleanup_bug.py` - Lock cleanup testing with temporary MODEL_CACHE override
- ✅ `test_health_checks.py` - Mock corruption testing with isolated `temp_cache_dir`

**Technical Pattern:**
```python
@pytest.mark.usefixtures("temp_cache_dir")
class TestBasicLifecycle:
    def test_something(self, temp_cache_dir, patch_model_cache):
        with patch_model_cache(temp_cache_dir / "hub"):
            # Test operates in complete isolation
            # User cache never touched, automatic cleanup
```

**Benefits:** 
- ✅ **Clean User Cache**: No test artifacts or broken models ever
- ✅ **Parallel Testing**: No cache conflicts between test runs  
- ✅ **Reproducible**: No dependency on existing models in user cache
- ✅ **Fast CI**: Small models (12MB vs 4GB) for most tests

### 🏥 CATEGORY 2: USER CACHE (Framework Diversity)
**📋 Reserved for future** - Real model diversity that cannot be mocked

**Future Framework Validation Tests:**
- Multiple framework detection (MLX + PyTorch + Tokenizer-only models)
- Health check diversity testing with naturally corrupted models
- Cross-framework model compatibility validation

**Currently**: All health/framework tests use `mock_model_cache` and are Category 1 (isolated)

### 🖥️ CATEGORY 3: SERVER CACHE (Performance Tests)  
**🔒 Large models, user cache expected** - Marked with `@pytest.mark.server`

**Server Tests (Excluded from default `pytest`):**
- 🔒 `test_issue_14.py` - Chat self-conversation regression tests
- 🔒 `test_issue_15_16.py` - Dynamic token limit validation  
- 🔒 `test_end_token_issue.py` - End-token filtering consistency
- 🔒 `test_server_functionality.py` - OpenAI API compliance (basic tests only)

**Technical Pattern:**
```python
@pytest.mark.server  # Excluded from default pytest
def test_server_feature(mlx_server, model_name: str):
    # Uses real models in user cache
    # Requires significant RAM and time
```

**Characteristics:**
- 🔒 **Not run by default** - Must use `pytest -m server`
- 💾 **RAM-aware** - Auto-skip models exceeding available memory
- ⏱️ **Longer execution** - 20-40 minutes for full suite
- 🎯 **Model diversity** - Tests across different model sizes/architectures

## Test Prerequisites

### Required Setup

1. **Apple Silicon Mac** (M1/M2/M3)
2. **Python 3.9 or newer**
3. **Test dependencies installed**:
   ```bash
   pip install -e ".[test]"
   ```

**That's it!** Most tests (Category 1) use isolated caches and download small test models automatically (~12MB).

### Optional Setup (Server Tests Only)

For server tests (`@pytest.mark.server` - **excluded by default**):
```bash
# Medium model for server testing
mlxk pull mlx-community/Phi-3-mini-4k-instruct-4bit

# Different architecture for variety  
mlxk pull mlx-community/Mistral-7B-Instruct-v0.3-4bit
```

**Note**: Server tests are excluded from default `pytest` and require manual execution with `pytest -m server`.

## Test Commands

### Basic Test Execution

```bash
# All tests (recommended before commits)
pytest

# Only integration tests (system-level)
pytest tests/integration/

# Only unit tests (fast)
pytest tests/unit/

# Verbose output
pytest -v

# Show test coverage
pytest --cov=mlx_knife --cov-report=html
```

### Specific Test Categories

```bash
# Process lifecycle tests (critical for production)
pytest tests/integration/test_process_lifecycle.py -v

# Health check robustness (model corruption detection)
pytest tests/integration/test_health_checks.py -v

# Core functionality (basic CLI commands)
pytest tests/integration/test_core_functionality.py -v

# Issue #20: End-token filtering consistency (new in 1.1.0-beta2)
pytest tests/integration/test_end_token_issue.py -v

# Advanced run command tests
pytest tests/integration/test_run_command_advanced.py -v

# Server functionality tests
pytest tests/integration/test_server_functionality.py -v

# Lock cleanup bug tests (Issue #23 - new in 1.1.0-beta3)
pytest tests/integration/test_lock_cleanup_bug.py -v
```

### Test Filtering

```bash
# Run only basic operations tests
pytest -k "TestBasicOperations" -v

# Server tests are excluded by default (marked with @pytest.mark.server)
# Run server tests manually (requires large models in user cache)
pytest -m server -v

# Skip server tests explicitly (default behavior)
pytest -m "not server" -v

# Run only process lifecycle tests
pytest -k "process_lifecycle or zombie" -v

# Run health check tests only
pytest -k "health" -v
```

### Timeout and Performance

```bash
# Set custom timeout (default: 300s, method=thread)
pytest --timeout=60 --timeout-method=thread

# Show slowest tests
pytest --durations=10

# Parallel execution (if pytest-xdist installed)
pytest -n auto
```

### Server Tests (Advanced)

**⚠️ Warning**: Server tests require significant system resources and time.

```bash
# Run comprehensive Issue #20 server tests (48 tests, ~30 minutes)
pytest tests/integration/test_end_token_issue.py -m server -v

# All server-marked tests (includes above + server functionality)
pytest -m server -v

# Quick server functionality test only
pytest tests/integration/test_server_functionality.py -v

# Server tests are RAM-aware - automatically skip models that don't fit
```

**Server Test Requirements:**
- **RAM**: 8GB+ recommended (16GB+ for large models)  
- **Time**: 20-40 minutes for full suite
- **Models**: Multiple 4-bit quantized models (1B-30B parameters)
- **Coverage**: Streaming vs non-streaming consistency, token limits, API compliance

### Memory Gating for Large Models

- The integration tests avoid loading oversized models by estimating RAM usage based on model size and quantization.
- Quantization detection uses common markers in the model name (e.g., `-4bit`, `q4`, `int4`) and, when available, details from `mlxk show <model>`.
- Two estimation maps are used: one for 4‑bit and one conservative for FP16/BF16.
- Safety margin: By default, tests use a RAM safety factor to keep headroom.
  - Configure via `MLXK_TEST_RAM_SAFETY` (float in `0.1..1.0`).
  - Examples:
    - `MLXK_TEST_RAM_SAFETY=0.8` (default in some tests): use ~80% of available RAM.
    - `MLXK_TEST_RAM_SAFETY=1.0`: use up to available RAM (minus 4 GB guard).
  - This allows FP16 models to be included when they truly fit in memory.
  
- Unknown size fallback: tests call `mlxk show <model>` and parse `Size:` and `Quantization:` for more accurate estimates (prevents `unknown → 999GB`).
  
- Advanced tuning (optional):
  - `MLXK_TEST_DISK_TO_RAM_FACTOR`: base factor for converting disk size (GB) to RAM estimate (default: 0.6).
  - `MLXK_TEST_FACTOR_4BIT`: override factor for 4‑bit models (falls back to `MLXK_TEST_DISK_TO_RAM_FACTOR`).
  - `MLXK_TEST_FACTOR_FP16`: override factor for FP16/BF16 models (falls back to `MLXK_TEST_DISK_TO_RAM_FACTOR`).

### Robust Server Process Cleanup

- Server tests install a process guard in their managers (not session-wide) and clean up `mlxk server` processes on Ctrl-C, SIGTERM, or teardown.
- Implementation: `tests/support/process_guard.py`; installed explicitly in server managers.
- Test code registers processes automatically:
  - `MLXKnifeServerManager`/`MLXKnifeServer` call `register_popen(...)` when starting `mlxk server`.
  - The generic `mlx_knife_process` fixture also registers its subprocesses.
- Environment toggles:
  - `MLXK_TEST_DISABLE_PROCESS_GUARD=1` disables guard registration (not recommended).
  - `MLXK_TEST_KILL_ZOMBIES_AT_START=1` sweeps stale servers at session start.
  - `MLXK_TEST_DETACH_PGRP=1` (advanced): detach runner into its own process group to isolate from stray group-kills.

## Python Version Compatibility

### Verification Results (September 2025)

**✅ 166/166 tests passing** - All standard tests validated on Apple Silicon with isolated cache system  
**🆕 1.1.1-beta.3** - MXFP4 quantization support and GPT-OSS reasoning model integration

| Python Version | Status | Tests Passing |
|----------------|--------|---------------|
| 3.9.6 (macOS)  | ✅ Verified | 166/166 |
| 3.10.x         | ✅ Verified | 166/166 |
| 3.11.x         | ✅ Verified | 166/166 |
| 3.12.x         | ✅ Verified | 166/166 |
| 3.13.x         | ✅ Verified | 166/166 |

All versions tested with isolated cache system.
Real MLX execution verified separately with server/run commands.

### Manual Multi-Python Testing

If you have multiple Python versions installed, you can verify compatibility:

```bash
# Run the multi-Python verification script
./test-multi-python.sh

# Or manually test specific versions
python3.9 -m venv test_39
source test_39/bin/activate
pip install -e ".[test]"
pytest
deactivate && rm -rf test_39
```

## Code Quality & Development

### Code Quality Tools

MLX Knife includes comprehensive code quality tools:

```bash
# Install development dependencies  
pip install -e ".[dev]"

# Automatic code formatting and linting
ruff check mlx_knife/ --fix

# Type checking with mypy
mypy mlx_knife/

# Complete development workflow
ruff check mlx_knife/ --fix && mypy mlx_knife/ && pytest
```

### Development Workflow

Before committing changes:

```bash
#!/bin/bash
# pre-commit-check.sh - Run before committing
set -e

echo "🧪 Running MLX Knife pre-commit checks..."

# 1. Code style
echo "Checking code style..."
ruff check mlx_knife/ --fix

# 2. Type checking
echo "Checking types..."
mypy mlx_knife/

# 3. Quick smoke test
echo "Running quick tests..."
pytest tests/unit/ -v

echo "✅ All checks passed. Safe to commit!"
```

## Local Development Testing

### Adding New Tests
1. **Integration tests** go in `tests/integration/`
2. **Unit tests** go in `tests/unit/`
3. Use existing fixtures from `conftest.py`
4. Follow naming: `test_*.py`, `Test*` classes, `test_*` methods

### Test Categories (Markers)
```python
@pytest.mark.integration  # Slower system tests
@pytest.mark.unit         # Fast isolated tests  
@pytest.mark.slow         # Tests >30 seconds
@pytest.mark.requires_model  # Needs actual MLX model
@pytest.mark.network      # Requires internet
@pytest.mark.server       # Requires MLX Knife server (excluded from default pytest)
```

### Mock Utilities
- `mock_model_cache()`: Creates fake model directories
- `mlx_knife_process()`: Manages subprocess lifecycle
- `process_monitor()`: Tracks zombie processes
- `temp_cache_dir()`: Isolated test environment

## Test Philosophy

Following the **"Process Hygiene over Edge-Case Perfection"** principle:

1. **Process Cleanliness**: No zombies, no leaks ✅
2. **Health Checks**: Reliable corruption detection ✅  
3. **Core Operations**: Basic functionality works ✅
4. **Error Handling**: Graceful failures ✅

The test suite validates production readiness with real Apple Silicon hardware and actual MLX models.

## Troubleshooting

### Common Issues

**Tests hang forever:**
```bash
pytest --timeout=60
```

**Import errors:**
```bash
pip install -e ".[test]"
```

**Process cleanup issues:**
```bash
ps aux | grep mlx_knife  # Check for zombies
```

**Cache conflicts:**
```bash
export HF_HOME="/tmp/test_cache"
pytest --cache-clear
```

### Test Environment

```bash
# Clean test run
rm -rf .pytest_cache __pycache__
pytest tests/ -v --cache-clear

# Debug specific test
pytest tests/integration/test_health_checks.py::TestHealthCheckRobustness::test_healthy_model_detection -v -s
```

## Contributing Test Results

When submitting PRs, please include:

1. **Your test environment**:
   - macOS version
   - Apple Silicon chip (M1/M2/M3)
   - Python version
   - Which model(s) you tested with

2. **Test results summary**:
   ```
   Platform: macOS 14.5, M2 Pro
   Python: 3.11.6
   Model: Phi-3-mini-4k-instruct-4bit
   Results: 150/150 tests passed
   ```

3. **Any issues encountered** and how you resolved them

## Summary

**MLX Knife 1.1.0 STABLE + 1.1.1-beta.3 Testing Status:**

✅ **Stable for development use** - 166/166 tests passing  
✅ **Isolated Test System** - User cache stays pristine with temp cache isolation
✅ **3-Category Strategy** - Optimized for performance and safety
✅ **Multi-Python Support** - Python 3.9-3.13 verified  
✅ **Code Quality** - ruff/mypy integration working  
✅ **Real Model Testing** - Server/run commands validated with multiple models
✅ **Memory Management** - Context managers prevent leaks  
✅ **Exception Safety** - Context managers ensure cleanup  
✅ **Cache Directory Fix** - Issue #21: Empty cache crash resolved
✅ **LibreSSL Warning Fix** - Issue #22: macOS Python 3.9 warning suppression
✅ **Lock Cleanup Fix** - Issue #23: Enhanced rm command with lock cleanup

This comprehensive testing framework validates MLX Knife's **stability for development use** through isolated testing with automatic model downloads and separate real MLX validation.

## Server-Based Testing (Advanced)

Some tests require a running MLX Knife server with loaded models. These tests are marked with `@pytest.mark.server` and are **not run by default** with `pytest`.

### Why Separate Server Tests?

- **Test count varies** by loaded models (makes CI reporting inconsistent)
- **Large memory requirements** - need different models for different RAM sizes  
- **Longer execution time** - each model needs to load individually
- **Manual setup required** - need to download appropriate models first
  
Note: If your shell prints a termination message after a successful run (e.g., "Terminated: 15" or "Killed: 9"), this can be caused by a stray SIGTERM/SIGKILL delivered to the test runner at teardown time by the environment. The suite installs a session handler that exits cleanly on SIGTERM to avoid this cosmetic noise. Disable for debugging with `MLXK_TEST_DISABLE_CATCH_TERM=1`.

### Prerequisites for Server Tests

| System RAM | Recommended Models | Commands |
|------------|-------------------|----------|
| **16GB**   | Small models only | `mlxk pull mlx-community/Qwen2.5-0.5B-Instruct-4bit`<br>`mlxk pull mlx-community/Llama-3.2-1B-Instruct-4bit`<br>`mlxk pull mlx-community/Llama-3.2-3B-Instruct-4bit` |
| **32GB**   | + Medium models | `mlxk pull mlx-community/Phi-3-mini-4k-instruct-4bit`<br>`mlxk pull mlx-community/Mistral-7B-Instruct-v0.2-4bit`<br>`mlxk pull mlx-community/Mixtral-8x7B-Instruct-v0.1-4bit` |
| **64GB**   | + Large models | `mlxk pull mlx-community/Mistral-Small-3.2-24B-Instruct-2506-4bit`<br>`mlxk pull mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit`<br>`mlxk pull mlx-community/Llama-3.3-70B-Instruct-4bit` |
| **96GB+**  | + Huge models | `mlxk pull mlx-community/Qwen3-Coder-480B-A35B-Instruct-4bit` |

### Running Server Tests

**Issue #14 Regression Tests** (Chat Self-Conversation Bug):

```bash
# Set environment
export HF_HOME=/path/to/your/cache

# Smoke test first (see which models are available)
python tests/integration/test_issue_14.py

# Run server tests only (excluded from default pytest)
pytest -m server -v

# Run specific Issue #14 tests
pytest tests/integration/test_issue_14.py -m server -v
```

**Expected Output:**
```
🦫 MLX Knife Issue #14 Test - Smoke Test
==================================================
📊 Safe models for this system: 6
💾 System RAM: 64GB total, 40GB available

  🎯 mlx-community/Mistral-7B-Instruct-v0.2-4bit
     └─ Size: 7B, RAM needed: 8GB
  🎯 mlx-community/Llama-3.2-3B-Instruct-4bit  
     └─ Size: 3B, RAM needed: 4GB
  [...]

========== test session starts ==========
tests/integration/test_issue_14.py::test_server_health[mlx_server] PASSED
tests/integration/test_issue_14.py::test_issue_14_self_conversation_regression_original[mlx-community/Mistral-7B-Instruct-v0.2-4bit-7B-8] PASSED
[...6 more model tests...]
========== 7 passed in 45.23s ==========
```

### Additional Server Tests

**Issues #15 & #16** - Dynamic Token Limits (Implemented in 1.1.0-beta1):
```bash
pytest tests/integration/test_issue_15_16.py -v
```

**Issue #20** - End-Token Filtering (Implemented in 1.1.0-beta2):
```bash
pytest tests/integration/test_end_token_issue.py -m server -v
```

### Troubleshooting Server Tests

**Permission warnings are normal:**
```
WARNING: ⚠️  Cannot scan network connections (permission denied)
INFO: 🔧 Falling back to process-based cleanup only
```
This is expected on macOS - the tests continue with process-based cleanup.

**Memory issues:**
- Tests automatically skip models exceeding 80% available RAM
- Use smaller models if you see consistent memory failures  
- Consider external SSD for model cache to reduce memory pressure

**Server startup failures:**
```bash
# Debug server manually
python -m mlx_knife.cli server --port 8000

# Check model health  
mlxk health

# Verify environment
echo $HF_HOME
```

### Adding New Server Tests

When contributing server-based tests:

```python
@pytest.mark.server
def test_new_feature(mlx_server, model_name: str, size_str: str, ram_needed: int):
    """Test new feature with MLX models.""" 
    # Use mlx_server fixture for automatic server management
    # Test implementation here
```

1. **Mark with `@pytest.mark.server`** - excludes from default `pytest`
2. **Use `mlx_server` fixture** - automatic server lifecycle management
3. **Test RAM requirements** - use `get_safe_models_for_system()` helper
4. **Document in TESTING.md** - add to this guide
