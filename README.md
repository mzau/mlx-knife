# <img src="https://github.com/mzau/mlx-knife/raw/main/broke-logo.png" alt="BROKE Logo" width="60" style="vertical-align: middle;"> MLX-Knife 2.0.0-alpha

**JSON-First Model Management for Automation & Scripting**

> **🚧 Alpha Development Branch:** This is the `feature/2.0.0-json-only` branch containing MLX-Knife 2.0.0-alpha. For stable production use, see [MLX-Knife 1.1.0](https://github.com/mzau/mlx-knife/tree/main).

[![GitHub Release](https://img.shields.io/badge/version-2.0.0--alpha-orange.svg)](https://github.com/mzau/mlx-knife/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-45%2F45%20passing-brightgreen.svg)](#testing)

## Quick Start

```bash
# Installation (local development)
git clone https://github.com/mzau/mlx-knife.git -b feature/2.0.0-json-only
cd mlx-knife
pip install -e .

# Basic usage - JSON API
mlxk-json list --json | jq '.data.models[].name'
mlxk-json health --json | jq '.data.summary'
mlxk-json show "Phi-3-mini" --json | jq '.data.model_info'
```

**What's New:** JSON-first architecture for automation and scripting  
**What's Missing:** Server mode, run command (use MLX-Knife 1.x for those)

## ⚠️ Alpha Status Disclaimer

MLX-Knife 2.0.0-alpha is **feature-complete for JSON operations** with production-quality reliability:

- ✅ **Core functionality works:** All 5 commands (`list`, `health`, `show`, `pull`, `rm`)
- ✅ **Test status:** 45/45 passing with comprehensive edge case coverage
- ✅ **Production use:** Suitable for broke-cluster integration and automation
- ✅ **Parallel use:** Deploy alongside MLX-Knife 1.x for server functionality

## What 2.0.0-alpha Includes

| Command | Status | Description |
|---------|--------|-------------|
| ✅ `list` | **Complete** | Model discovery with JSON output |
| ✅ `health` | **Complete** | Corruption detection and cache analysis |  
| ✅ `show` | **Complete** | Detailed model information with --files, --config |
| ✅ `pull` | **Complete** | HuggingFace model downloads with corruption detection |
| ✅ `rm` | **Complete** | Model deletion with lock cleanup and fuzzy matching |

## What's Coming Later

| Feature | Target Version | Status |
|---------|----------------|---------|
| 🔄 `server` | 2.0.0-rc | OpenAI-compatible API server |
| 🔄 `run` | 2.0.0-rc | Interactive model execution |
| 🔄 Human-readable output | 2.0.0-rc | CLI formatting layer |
| 🔄 `embed` | TBD | Embedding generation (if merged from 1.x) |

## Installation & Parallel Usage

### Development Installation

```bash
# Install 2.0.0-alpha (this branch)
pip install -e /path/to/mlx-knife

# Verify installation
mlxk-json --version  # → MLX-Knife JSON 2.0.0-alpha
mlxk2 --version      # → MLX-Knife JSON 2.0.0-alpha
```

### Parallel with MLX-Knife 1.x

Both versions can coexist safely:

```bash
# Install stable 1.x for server/run features
pip install mlx-knife

# Commands available:
mlxk list                    # 1.x - Human-readable output
mlxk server --port 8080      # 1.x - Server mode
mlxk run "model" -p "Hello"  # 1.x - Interactive execution

mlxk-json list --json        # 2.0 - JSON API
python -m mlxk2.cli list     # 2.0 - Module invocation
```

**Package Names:**
- MLX-Knife 1.x: `mlx-knife` → `mlxk` command
- MLX-Knife 2.0: `mlxk-json` → `mlxk-json`, `mlxk2` commands

## JSON API Documentation

> **📋 Complete API Specification**: See [docs/json-api-specification.md](docs/json-api-specification.md) for comprehensive JSON schema, error codes, and integration examples.

### Command Structure

All commands follow this JSON response format:

```json
{
    "status": "success|error", 
    "command": "list|health|show|pull|rm",
    "data": { /* command-specific data */ },
    "error": null | { "message": "...", "details": "..." }
}
```

### Examples

#### List Models
```bash
mlxk-json list --json
# Output:
{
    "status": "success",
    "command": "list", 
    "data": {
        "models": [
            {
                "name": "mlx-community/Phi-3-mini-4k-instruct-4bit",
                "hashes": ["e9675aa3def456789abcdef0123456789abcdef0"],
                "cached": true
            }
        ],
        "count": 1
    },
    "error": null
}
```

#### Health Check
```bash
mlxk-json health --json
# Output:
{
    "status": "success",
    "command": "health",
    "data": {
        "healthy": [...],
        "unhealthy": [...],
        "summary": {"total": 5, "healthy_count": 4, "unhealthy_count": 1}
    },
    "error": null
}
```

#### Show Model Details
```bash
mlxk-json show "Phi-3-mini" --json --files
# Output includes file listings, model config, capabilities
```

### Hash Syntax Support

All commands support `@hash` syntax for specific model versions:

```bash
mlxk-json health "Qwen3@e96" --json     # Check specific hash
mlxk-json show "model@3df9bfd" --json   # Short hash matching
mlxk-json rm "Phi-3@e967" --json --force  # Delete specific version
```

## HuggingFace Cache Safety

MLX-Knife 2.0 respects standard HuggingFace cache structure and practices:

### Best Practices for Shared Environments
- **Read operations** (`list`, `health`, `show`) always safe with concurrent processes
- **Write operations** (`pull`, `rm`) coordinate during maintenance windows  
- **Lock cleanup** automatic but avoid during active downloads
- **Your responsibility:** Coordinate with team, use good timing

### Example Safe Workflow
```bash
# Check what's in cache (always safe)
mlxk-json list --json | jq '.data.count'

# Maintenance window - coordinate with team
mlxk-json rm "corrupted-model" --json --force
mlxk-json pull "replacement-model" --json

# Back to normal operations
mlxk-json health --json | jq '.data.summary'
```

## Real-World Examples

> **🔗 Integration Reference**: External projects should implement against [docs/json-api-specification.md](docs/json-api-specification.md) - this alpha phase helps validate that specification matches actual implementation.

### Broke-Cluster Integration
```bash
# Get available model names for scheduling
MODELS=$(mlxk-json list --json | jq -r '.data.models[].name')

# Check cache health before deployment
HEALTH=$(mlxk-json health --json | jq '.data.summary.healthy_count')
if [ "$HEALTH" -eq 0 ]; then
    echo "No healthy models available"
    exit 1
fi

# Download required models
mlxk-json pull "mlx-community/Phi-3-mini-4k-instruct-4bit" --json
```

### CI/CD Pipeline Usage
```bash
# Verify model integrity in CI
mlxk-json health --json | jq -e '.data.summary.unhealthy_count == 0'

# Clean up CI artifacts
mlxk-json rm "test-model-*" --json --force

# Pre-warm cache for deployment
mlxk-json pull "production-model" --json
```

### Model Management Automation
```bash
# Find models by pattern
LARGE_MODELS=$(mlxk-json list --json | jq -r '.data.models[] | select(.name | contains("30B")) | .name')

# Show detailed info for analysis
for model in $LARGE_MODELS; do
    mlxk-json show "$model" --json --config | jq '.data.model_config'
done
```

## Testing

The 2.0 test suite runs by default (pytest discovery points to `tests_2.0/`):

```bash
# Run 2.0 tests (default)
pytest -v

# Explicitly run legacy 1.x tests (not maintained on this branch)
pytest tests/ -v

# Test categories (2.0 example):
# - ADR-002 edge cases
# - Integration scenarios  
# - Model naming logic
# - Robustness testing

# Current status: 45/45 passing ✅
```

**Revolutionary Test Architecture:**
- **Isolated Cache System** - Zero risk to user data
- **Atomic Context Switching** - Production/test cache separation
- **Comprehensive Mock Models** - Realistic test scenarios
- **Edge Case Coverage** - All documented failure modes tested

## Known Issues & Limitations

### Critical Issues
- **Health Check False Positive**: Health check may report incomplete downloads as healthy during model pull operations (affects both 1.1.0 and 2.0.0-alpha)

### Alpha Limitations
- No interactive prompts (use `--force` flag for rm operations)
- JSON output only (no human-readable formatting)
- Limited error message user experience (coming in beta)

### GitHub Issues
- **Issue #18**: Server signal handling limitation (known, will fix in 2.0.0-rc)
- **Issue #24**: Lock cleanup command (planned for future release)

## Development Status

### Version Roadmap
- **2.0.0-alpha** ← You are here (JSON API core complete)
- **2.0.0-beta**: 6-8 weeks robust testing, production validation  
- **2.0.0-rc**: Server/run features, full 1.x parity
- **2.0.0-stable**: Community validated, enterprise ready

### Architecture Decisions
- **JSON-First**: All output structured for scripting and automation
- **Cache Safety**: Respects HuggingFace standards, no custom formats
- **Atomic Operations**: Clean separation between test and production contexts
- **Backward Compatibility**: Parallel deployment with 1.x maintained

## Contributing

This branch follows the established MLX-Knife development patterns:

```bash
# Run quality checks
python test-multi-python.sh  # Tests across Python 3.9-3.13
./run_linting.sh             # Code quality validation

# Key files:
mlxk2/                       # 2.0.0 implementation
tests_2.0/                   # Alpha test suite  
docs/ADR/                    # Architecture decision records
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## Support & Feedback

- **Issues**: [GitHub Issues](https://github.com/mzau/mlx-knife/issues)
- **Discussions**: [GitHub Discussions](https://github.com/mzau/mlx-knife/discussions)
- **API Specification**: [docs/json-api-specification.md](docs/json-api-specification.md) - Complete JSON schema
- **Documentation**: See `docs/` directory for technical details

**For production use**: Consider MLX-Knife 1.1.0 until 2.0.0-beta is available.

### Alpha Testing Goals
- ✅ Validate JSON API specification matches implementation
- ✅ Real-world integration feedback from external projects  
- ✅ Edge case discovery through broke-cluster usage
- ✅ API stability testing before beta release

---

*MLX-Knife 2.0.0-alpha - Built for automation, tested for reliability, designed for the future.*

## Local Safety Setup (Optional)

To keep local coordination files out of Git and avoid accidental pushes during development:

- Ignore locally (branch-independent): add to `.git/info/exclude`
  - `AGENTS.md`
  - `CLAUDE.md`
- Local hooks (not versioned):
  - `.git/hooks/pre-commit` blocks commits including `AGENTS.md`/`CLAUDE.md`.
  - `.git/hooks/pre-push` blocks pushes of the current branch. Override once with `ALLOW_PUSH=1 git push`.

Minimal pre-commit example:
```bash
#!/usr/bin/env bash
set -euo pipefail
staged=$(git diff --cached --name-only || true)
for f in AGENTS.md CLAUDE.md; do
  echo "$staged" | grep -qx "$f" && { echo "Commit blocked: $f" >&2; exit 1; }
done
```

Minimal pre-push example:
```bash
#!/usr/bin/env bash
set -euo pipefail
[ "${ALLOW_PUSH:-}" = "1" ] && exit 0
br=$(git rev-parse --abbrev-ref HEAD)
while read -r l _ r _; do [ "$l" = "refs/heads/$br" ] && { echo "Push blocked: $br" >&2; exit 1; }; done
exit 0
```
