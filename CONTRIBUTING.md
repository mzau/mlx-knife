# Contributing to MLX Knife

First off, thank you for considering contributing to MLX Knife! It's people like you who make MLX Knife such a great tool for the Apple Silicon ML community.

## 🦫 About The BROKE Team

We're a small team passionate about making MLX models accessible and easy to use on Apple Silicon. We welcome contributions from everyone who shares this vision.

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
3. Ensure the test suite passes locally: `pytest tests/`
4. Make sure your code follows the existing style: `ruff check mlx_knife/ --fix`
5. Write a clear commit message
6. Open a Pull Request with a clear title and description

## Development Setup

```bash
# Clone your fork
git clone https://github.com/mzau/mlx-knife.git
cd mlx-knife

# Install in development mode with all dependencies
pip install -e ".[dev,test]"

# Download a test model (required for full test suite)
mlxk pull mlx-community/Phi-3-mini-4k-instruct-4bit

# Run tests
pytest

# Check code style
ruff check mlx_knife/
mypy mlx_knife/

# Test with a real model
mlxk run Phi-3-mini "Hello world"
```

## Repository Structure

Understanding what goes where:

```
Repository structure:
├── mlx_knife/              # Python package (→ PyPI)
├── tests/                  # Test suite
├── simple_chat.html        # Web interface (GitHub only)
├── README.md               # User documentation  
├── CONTRIBUTING.md         # This file
├── TESTING.md              # Testing guide
├── CLAUDE.md               # Development notes
├── pyproject.toml          # Build configuration
└── requirements.txt        # Dependencies
```

**What goes where:**
- **PyPI Package**: Only `mlx_knife/` + build files (`pyproject.toml`, `requirements.txt`)
- **GitHub Repository**: Everything else (documentation, tests, web interface)

This helps ensure contributors commit files to the right place and understand the package vs. repository distinction.

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
ruff check mlx_knife/ --fix    # Fix code style
mypy mlx_knife/                 # Check types
pytest tests/                   # Run all tests
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
   - Run the full test suite locally: `pytest tests/`
   - Run code quality checks: `ruff check mlx_knife/ --fix`
   - Test with YOUR Python version (3.9+ required)
   - Update README.md if you've added features

## Testing

MLX Knife has comprehensive test coverage. For detailed testing documentation including advanced options, test structure, and troubleshooting, see **[TESTING.md](TESTING.md)**.

**When adding new tests**: Please update the test structure documentation in **[TESTING.md](TESTING.md)** if you add new test files or categories.

## Code Style

- We use `ruff` for formatting and linting
- Type hints are encouraged (checked with `mypy`)
- Follow existing patterns in the codebase
- **IMPORTANT**: Keep Python 3.9 compatibility!
  - Use `Union[str, List[str]]` not `str | List[str]`
  - Use `Optional[str]` not `str | None`
  - Import from `typing` module for type hints
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

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to MLX Knife!**

Every contribution, no matter how small, makes a difference. Whether it's fixing a typo, adding a test, or implementing a new feature - we appreciate your time and effort.