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
3. Ensure the test suite passes: `pytest tests/`
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

# Run tests
pytest

# Check code style
ruff check mlx_knife/
mypy mlx_knife/

# Test with a real model
mlxk pull mlx-community/Phi-3-mini-4k-instruct-4bit
mlxk run Phi-3-mini "Hello world"
```

## Python Version Requirements

**Minimum**: Python 3.9 (the native macOS version on Apple Silicon)

We prioritize compatibility with:
- **Python 3.9**: Native macOS version - MUST work
- **Newer versions**: Should work, but 3.9 is our baseline

You don't need to test on all Python versions! Just test with what you have:
- If you have native macOS Python 3.9: Perfect! That's our main target
- If you have a newer version: Great, test with that
- Multiple versions installed? Bonus, but not required

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
   - Run the full test suite: `pytest tests/`
   - Run code quality checks: `ruff check mlx_knife/ --fix`
   - Test with YOUR Python version (3.9+ required)
   - Mention your Python version in the PR description
   - Update README.md if you've added features

## Testing

- **Unit tests**: Fast, isolated tests in `tests/unit/`
- **Integration tests**: System-level tests in `tests/integration/`
- **Real model tests**: Use Phi-3-mini for testing (it's small and fast)

Run specific test categories:
```bash
pytest tests/unit/              # Fast unit tests
pytest tests/integration/        # Integration tests
pytest -k "not requires_model"   # Skip tests requiring models
```

**Note**: Our CI will test multiple Python versions automatically after you submit your PR. You only need to test with your local Python version (3.9+).

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