# MLX Knife Testing Guide

## Current Status

✅ **114/114 tests passing** (August 2025)  
✅ **Apple Silicon verified** (M1/M2/M3)  
✅ **Python 3.9-3.13 compatible**  
✅ **Production ready** - real model execution validated

## Quick Start

```bash
# Install with test dependencies
pip install -e ".[test]"

# Download test model (required for most tests)
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
├── conftest.py                     # Shared fixtures and utilities
├── integration/                    # System-level integration tests (62 tests)
│   ├── test_core_functionality.py      # Basic CLI operations
│   ├── test_health_checks.py           # Model corruption detection  
│   ├── test_process_lifecycle.py       # Process management & cleanup
│   ├── test_run_command_advanced.py    # Run command edge cases
│   └── test_server_functionality.py    # OpenAI API server tests
└── unit/                          # Module-level unit tests (52 tests)
    ├── test_cache_utils.py            # Cache management functions
    ├── test_cli.py                    # CLI argument parsing
    └── test_mlx_runner_memory.py     # Memory management tests
```

## Test Prerequisites

### Required Setup

1. **Apple Silicon Mac** (M1/M2/M3)
2. **Python 3.9 or newer**
3. **Test dependencies installed**:
   ```bash
   pip install -e ".[test]"
   ```
4. **At least one MLX model**:
   ```bash
   mlxk pull mlx-community/Phi-3-mini-4k-instruct-4bit
   ```

### Optional Setup

For full test coverage, you may want additional models:
```bash
# Smaller model for quick tests
mlxk pull mlx-community/Phi-3-mini-128k-instruct-4bit

# Different architecture for variety
mlxk pull mlx-community/Mistral-7B-Instruct-v0.3-4bit
```

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

# Advanced run command tests
pytest tests/integration/test_run_command_advanced.py -v

# Server functionality tests
pytest tests/integration/test_server_functionality.py -v
```

### Test Filtering

```bash
# Run only basic operations tests
pytest -k "TestBasicOperations" -v

# Skip server tests (faster)
pytest -k "not server" -v

# Skip tests requiring actual models
pytest -k "not requires_model" -v

# Run only process lifecycle tests
pytest -k "process_lifecycle or zombie" -v

# Run health check tests only
pytest -k "health" -v
```

### Timeout and Performance

```bash
# Set custom timeout (default: 300s)
pytest --timeout=60

# Show slowest tests
pytest --durations=10

# Parallel execution (if pytest-xdist installed)
pytest -n auto
```

## Python Version Compatibility

### Verification Results (August 2025)

| Python Version | Status | Tests Passing |
|----------------|--------|---------------|
| 3.9.6 (macOS)  | ✅ Verified | 114/114 |
| 3.10.x         | ✅ Verified | 114/114 |
| 3.11.x         | ✅ Verified | 114/114 |
| 3.12.x         | ✅ Verified | 114/114 |
| 3.13.x         | ✅ Verified | 114/114 |

All versions tested with real MLX model execution (Phi-3-mini-4k-instruct-4bit).

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
   Results: 114/114 tests passed
   ```

3. **Any issues encountered** and how you resolved them

## Summary

**MLX Knife 1.0.3 Testing Status:**

✅ **Production Ready** - 114/114 tests passing  
✅ **Multi-Python Support** - Python 3.9-3.13 verified  
✅ **Code Quality** - ruff/mypy integration working  
✅ **Real Model Testing** - Phi-3-mini execution confirmed  
✅ **Memory Management** - Context managers prevent leaks  
✅ **Exception Safety** - Context managers ensure cleanup  

This comprehensive testing framework validates MLX Knife's **production readiness** through local testing on real Apple Silicon hardware with actual MLX models.