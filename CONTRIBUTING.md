# Contributing to MLX Knife

First off, thank you for considering contributing to MLX Knife! It's people like you who make MLX Knife such a great tool for the Apple Silicon ML community.

## 🦫 About The BROKE Team

We're a small team passionate about making MLX models accessible and easy to use on Apple Silicon. We welcome contributions from everyone who shares this vision.

## 2.0 Stable – Contributor Notes

- **Code path:** `mlxk2/` (entry points: `mlxk`, `mlxk2`)
- **Default output:** Human-friendly tables/text; pass `--json` for machine-readable JSON API
- **Commands:** `mlxk --help` lists them
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
├── mlxk2/                       # 2.0 implementation (→ PyPI package mlx-knife)
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
2. Python 3.11 or newer
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

**All tests must pass:**
- ✅ Code quality: `ruff check mlxk2/ --fix && mypy mlxk2/`
- ✅ Unit tests: `pytest tests_2.0/ -v` (always required)
- ✅ Live E2E tests: Required for model/inference changes

**PR requirements:**
- State your Python version + Mac chip in PR description
- For model/inference changes: Document which live tests you ran
- **Important:** Unit tests alone are NOT sufficient - see **[TESTING.md](TESTING.md)** for why and how

## Python Version Requirements

**Supported**: Python 3.11–3.14 (`requires-python = ">=3.11"`).

You don't need to test on all Python versions. Test with what you have and mention it in the PR
description; `bash test-multi-python.sh` runs the suite on every installed version.

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
   - Test with YOUR Python version (3.11+ required)
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

## Documentation

- Update docstrings for new functions/classes
- Update README.md for user-facing changes
- Keep CLI help text (`--help`) up to date
- Add comments for complex logic

### Version numbers in prose

A version is written so that a reader, and `tests_2.0/spec/test_version_prose.py`, can tell it
from a section number, a measurement or a model name:

- **mlx-knife versions:** three parts (`2.0.8`), or named: `MLX Knife 2.0`, `mlx-knife 2.0.8`,
  `mlxk 2.0.8`, `Version 2.0`. Never a bare two-part number. A series is `2.0.x`.
- **Schema versions:** always named: `JSON API 0.2.4`, `spec v0.2.4`, `schema v0.2.2` (the
  benchmark report schema).
- **Nothing unreleased carries a number:** no version after the current release, no schema
  version after the current one. Write the state instead: deferred, not built, a tracking issue.
- **"unreleased"** marks a document that describes the tree ahead of the release
  ("From 2.0.8 → unreleased"). It enters with the first change that describes behavior the
  release does not have, and the release commit removes it. The SERVER-HANDBOOK carries it in
  the header and the migration section together.
- **No "released" before a version** — the number says it.
- **"latest/current release X"** and version echoes (the output shown for `--version`,
  `"cli_version"`) name the newest release; `"json_api_spec_version"` names `mlxk2/spec.py`.
- A name directly before a number makes it someone else's: `LibreSSL 2.8.3`,
  `Apache License, Version 2.0`.

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
