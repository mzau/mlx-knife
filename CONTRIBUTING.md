# Contributing to MLX Knife

First off, thank you for considering contributing to MLX Knife! It's people like you who make MLX Knife such a great tool for the Apple Silicon ML community.

## 🦫 About The BROKE Team

We're a small team passionate about making MLX models accessible and easy to use on Apple Silicon. We welcome contributions from everyone who shares this vision.

## 2.0 Stable – Contributor Notes

- **Code path:** `mlxk2/` (entry points: `mlxk`, `mlxk-json`, `mlxk2`)
- **Default output:** Human-friendly tables/text; pass `--json` for machine-readable JSON API
- **Full feature parity:** All commands available (`list`, `health`, `show`, `pull`, `rm`, `run`, `serve`)
- **Tests:** Primary suite is `tests_2.0/` (see `pytest.ini`)
- **Human output options:**
  - `list`: `--all` (all frameworks), `--health` (add column), `--verbose` (full org/model names)
  - Compact default: MLX-only, compact names (strip `mlx-community/`), no Framework column
- **Cache safety:** Tests use isolated temp caches; read-only ops are safe; coordinate `pull`/`rm` when using a shared user cache
- **Spec discipline:** JSON schema/spec changes require a version bump in `mlxk2/spec.py` (see docs/)


## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues to avoid duplicates. When you create a bug report, include as many details as possible:

- **Use a clear and descriptive title**
- **Describe the exact steps to reproduce the problem**
- **Provide specific examples** (commands, model names, error messages)
- **Describe the behavior you observed and expected**
- **Include your system info** (macOS version, Python version, Apple Silicon chip)

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion:

- **Use a clear and descriptive title**
- **Provide a detailed description** of the suggested enhancement
- **Explain why this enhancement would be useful** to MLX Knife users
- **List some examples** of how it would be used

### Pull Requests

1. Fork the repository and create your branch from `main`
2. If you've added code, add tests that cover your changes
3. Ensure the test suite passes locally: `pytest tests_2.0/ -v`
4. Make sure your code follows the existing style: `ruff check mlxk2/ --fix`
5. Write a clear commit message
6. Open a Pull Request with a clear title and description

## Development Setup

```bash
# Clone your fork
git clone https://github.com/mzau/mlx-knife.git
cd mlx-knife

# Install in development mode
pip install -e .

# Download a test model (required for full test suite)
mlxk pull mlx-community/Phi-3-mini-4k-instruct-4bit

# Run tests (2.0 default)
pytest tests_2.0/ -v

# Check code style (2.0)
ruff check mlxk2/
mypy mlxk2/

# Test with a real model
mlxk run Phi-3-mini "Hello world"
```

## Repository Structure

Understanding what goes where:

```
Repository structure:
├── mlxk2/                       # 2.0 implementation (→ PyPI via mlxk-json)
├── tests_2.0/                   # 2.0 test suite
├── docs/                        # Documentation / ADRs
├── README.md                    # User documentation
├── CONTRIBUTING.md              # This file
├── TESTING.md                   # Testing guide
├── pyproject.toml               # Build configuration (dynamic version, optional test deps)
└── requirements.txt             # Dev/test dependencies
```

**What goes where:**
- **PyPI Package**: Only `mlxk2/` + `pyproject.toml` (optional dependencies excluded from release wheel)
- **GitHub Repository**: Everything else (documentation, tests)
- **Web Interface**: Separate project at [github.com/mzau/broke-nchat](https://github.com/mzau/broke-nchat) (shared across BROKE ecosystem)

This helps ensure contributors commit files to the right place and understand the package vs. repository distinction.

**Note:** The web UI (nChat) is intentionally separate to enable reuse across the BROKE ecosystem (MLX Knife + BROKE Cluster). Do not add web UI code to this repository.

## Testing Requirements

**Important**: MLX Knife requires Apple Silicon hardware for testing. Tests must be run locally on M1/M2/M3 Macs.

### Why Local Testing?

- MLX framework only runs on Apple Silicon
- Tests use real MLX models (4GB+) for realistic validation
- This ensures tests reflect actual usage, not mocked behavior
- Standard practice for MLX projects

### Running Tests

**Prerequisites:**
1. Apple Silicon Mac (M1/M2/M3)
2. Python 3.9 or newer
3. At least one MLX model installed:
   ```bash
   mlxk pull mlx-community/Phi-3-mini-4k-instruct-4bit
   ```

**Test Commands:**
```bash
# Run all tests
pytest
```

For detailed testing options, troubleshooting, and advanced workflows, see **[TESTING.md](TESTING.md)**.

### Before Submitting PRs

Please ensure all tests pass locally:
```bash
# Complete test workflow
ruff check mlxk2/ --fix         # Fix code style
mypy mlxk2/                     # Check types
pytest -v                       # Run all 2.0 tests
```

Since we don't have CI/CD (MLX requires Apple Silicon), we rely on contributors to verify their changes locally. Please mention in your PR:
- Which Python version you tested with
- Which Mac model you tested on (M1/M2/M3)
- Test results summary

## Python Version Requirements

**Minimum**: Python 3.9 (the native macOS version on Apple Silicon)

We prioritize compatibility with:
- **Python 3.9**: Native macOS version - MUST work
- **Newer versions**: Should work, but 3.9 is our baseline

You don't need to test on all Python versions! Just test with what you have:
- If you have native macOS Python 3.9: Perfect! That's our main target
- If you have a newer version: Great, test with that
- Multiple versions installed? Bonus, but not required

Mention your Python version in the PR description.

## Development Workflow

1. **Before starting work:**
   - Check if an issue exists for your change
   - If not, open an issue to discuss the change
   - For major changes, wait for feedback before starting

2. **While working:**
   - Keep changes focused and atomic
   - Write descriptive commit messages
   - Add/update tests as needed
   - Update documentation if needed

3. **Before submitting:**
   - Run the full test suite locally: `pytest -v`
   - Run code quality checks: `ruff check mlxk2/ --fix`
   - Test with YOUR Python version (3.9+ required)
   - Update README.md if you've added features

## Testing

MLX Knife has comprehensive test coverage. For detailed testing documentation including advanced options, test structure, and troubleshooting, see **[TESTING.md](TESTING.md)**.

**When adding new tests**: Please update the test structure documentation in **[TESTING.md](TESTING.md)** if you add new test files or categories.

### Spec Version Discipline (JSON API)

If you change the JSON API spec or schema, bump the spec version and keep code/tests in sync.

- Spec files: `docs/json-api-specification.md`, `docs/json-api-schema.json`
- Version constant: `mlxk2/spec.py` → `JSON_API_SPEC_VERSION`
- Guard script: `scripts/check-spec-bump.sh`

Usage examples:

```bash
# Local check against main
scripts/check-spec-bump.sh origin/main

# Bypass for editorial-only changes
SPEC_BUMP_BYPASS=1 scripts/check-spec-bump.sh origin/main
```

CI suggestion (GitHub Actions step):

```bash
- name: Check JSON API spec bump
  run: |
    git fetch origin main --depth=1
    scripts/check-spec-bump.sh origin/main
```

Bypass tokens (commit message): `[no-spec-bump]` or `[skip-spec-bump]` for formatting-only edits.

## Code Style

- We use `ruff` for formatting and linting
- Type hints are encouraged (checked with `mypy`)
- Follow existing patterns in the codebase
- **IMPORTANT**: Keep Python 3.9 compatibility!
  - Prefer `typing.Optional`/`typing.Union` over `|` syntax
  - Import from `typing` for hints
  - Test with native macOS Python if possible

## Documentation

- Update docstrings for new functions/classes
- Update README.md for user-facing changes
- Keep CLI help text (`--help`) up to date
- Add comments for complex logic

## Recognition

Contributors who submit accepted PRs will be:
- Added to a CONTRIBUTORS.md file (once we have contributors!)
- Mentioned in release notes
- Forever part of MLX Knife history 🦫

## Questions?

Feel free to open an issue with the "question" label or start a discussion. We're here to help!

## License

**Important:** MLX Knife 2.0+ is licensed under the **Apache License, Version 2.0**.

By contributing to MLX Knife, you agree that:
1. Your contributions will be licensed under the Apache License, Version 2.0
2. You have the right to contribute the code under these terms
3. You grant the project maintainers a perpetual, worldwide, non-exclusive, royalty-free license to use, reproduce, modify, and distribute your contributions

**Legacy 1.x versions** (MIT License) are maintained in the `1.x-legacy` branch for reference only. All new contributions go to the main branch (Apache 2.0).

We recommend including a Developer Certificate of Origin (DCO) "Signed-off-by" line in your commits:
```bash
git commit -s -m "Your commit message"
```

---

**Thank you for contributing to MLX Knife!**

Every contribution, no matter how small, makes a difference. Whether it's fixing a typo, adding a test, or implementing a new feature - we appreciate your time and effort.
