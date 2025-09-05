# <img src="https://github.com/mzau/mlx-knife/raw/main/broke-logo.png" alt="BROKE Logo" width="60" style="vertical-align: middle;"> MLX-Knife 2.0.0-alpha.2

<p align="center">
  <img src="https://github.com/mzau/mlx-knife/raw/main/mlxk-demo.gif" alt="MLX Knife Demo" width="1000">
</p>

## New: JSON-First Model Management for Automation & Scripting

> **🚧 Alpha Development:** Server and run are not included yet in 2.0.0-alpha.2. Use [MLX-Knife 1.1.0](https://github.com/mzau/mlx-knife/tree/main) for those features.

**Stable Version: 1.1.0**

[![GitHub Release](https://img.shields.io/badge/version-2.0.0--alpha.2-orange.svg)](https://github.com/mzau/mlx-knife/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-M1%2FM2%2FM3-green.svg)](https://support.apple.com/en-us/HT211814)
[![MLX](https://img.shields.io/badge/MLX-Latest-orange.svg)](https://github.com/ml-explore/mlx)
[![Sponsor mlx-knife](https://img.shields.io/badge/Sponsor-mlx--knife-ff69b4?logo=github-sponsors&logoColor=white)](https://github.com/sponsors/mzau)

[![Tests](https://img.shields.io/badge/tests-45%2F45%20passing-brightgreen.svg)](#testing)

## Features

### Core Functionality
- **List & Manage Models**: Browse your HuggingFace cache with MLX-specific filtering
- **Model Information**: Detailed model metadata including quantization info
- **Download Models**: Pull models from HuggingFace with progress tracking
- **Run Models**: Native MLX execution with streaming and chat modes (version 1.0.0 stable only)
- **Health Checks**: Verify model integrity and completeness
- **Cache Management**: Clean up and organize your model storage

### Requirements
- macOS with Apple Silicon (M1/M2/M3)
- Python 3.9+ (native macOS version or newer)
- 8GB+ RAM recommended + RAM to run LLM

### Python Compatibility
MLX Knife has been comprehensively tested and verified on:

✅ **Python 3.9.6** (native macOS) - Primary target  
✅ **Python 3.10-3.13** - Fully compatible  



## Quick Start

```bash
# Installation (local development)
git clone https://github.com/mzau/mlx-knife.git
cd mlx-knife
pip install -e .
```
# Install with development tools (ruff, mypy, tests)
pip install -e ".[dev,test]"
```

## Human output (default)
mlxk2 list
mlxk2 list --health
mlxk2 list --all --verbose
mlxk2 health
mlxk2 show "mlx-community/Phi-3-mini-4k-instruct-4bit"

## JSON API
mlxk2 list --json | jq '.data.models[].name'
mlxk2 health --json | jq '.data.summary'
mlxk2 show "Phi-3-mini" --json | jq '.data.model'
```

## Differences vs 1.0.0

- CLI: new entry points `mlxk2` and `mlxk-json` (1.0.0 used `mlxk`).
- Output: human output by default; add `--json` for machine-readable responses (new vs 1.0.0).
- List formatting: improved compact table with relative times in the Modified column (e.g., 3h ago) and a new Type column; compact MLX-only view by default.
- Flags (human-only): `--all` (all frameworks), `--health` (add Health column), `--verbose` (show full `org/model`).
- JSON API: current spec v0.1.3; CLI accepts `--json` after subcommands.
- Missing features (compared to 1.0.0): server and run are not included in 2.0 alpha.2 (use `mlxk` 1.x).

## ⚠️ Alpha Status Disclaimer

This is an alpha because:
- Not feature-complete vs 1.0.0 (server and run pending).
- Major internal refactor to a JSON-first CLI (new package `mlxk2`).

Status:
- ✅ Core commands: `list`, `health`, `show`, `pull`, `rm`.
- ✅ JSON outputs stable and schema-aligned; human output available by default.
- ✅ Suitable for automation/integration; can run alongside 1.x for server/run.

## What 2.0.0-alpha Includes

| Command | Status | Description |
|---------|--------|-------------|
| ✅ `list` | **Complete** | Model discovery with JSON output |
| ✅ `health` | **Complete** | Corruption detection and cache analysis |  
| ✅ `show` | **Complete** | Detailed model information with --files, --config |
| ✅ `pull` | **Complete** | HuggingFace model downloads with corruption detection |
| ✅ `rm` | **Complete** | Model deletion with lock cleanup and fuzzy matching |
| 🧪 `push` | **Experimental (alpha)** | Upload-only; quiet JSON; supports `--check-only` and `--dry-run` |

## What's Coming Later

| Feature | Target Version | Status |
|---------|----------------|---------|
| 🔄 `server` | 2.0.0-rc | OpenAI-compatible API server |
| 🔄 `run` | 2.0.0-rc | Interactive model execution |
| ✅ Human-readable output | 2.0.0-alpha.2 | CLI formatting layer |
| 🔄 `embed` | TBD | Embedding generation (if merged from 1.x) |

## Experimental: `push` (upload only)

`mlxk2 push` is experimental (M0). It uploads a local folder to a Hugging Face model repository using `huggingface_hub/upload_folder`.

- Requires `HF_TOKEN` (write-enabled).
- Default branch: `main` (explicitly override with `--branch`).
- Alpha safety: `--private` is required to avoid accidental public uploads.
- No validation or manifests. Basic hard excludes are applied by default: `.git/**`, `.DS_Store`, `__pycache__/`, common virtualenv folders (`.venv/`, `venv/`), and `*.pyc`.
- `.hfignore` (gitignore-like) in the workspace is supported and merged with the defaults.
- Repo creation: use `--create` if the target repo does not exist; harmless on existing repos. Missing branches are created during upload.
- JSON-first: output includes `commit_sha`, `commit_url`, `no_changes`, `uploaded_files_count` (when available), `local_files_count` (approx), `change_summary` and a short `message`.
- Quiet JSON by default: with `--json` (without `--verbose`) progress bars/console logs are suppressed; hub logs are still captured in `data.hf_logs`.
- Human output: derived from JSON; add `--verbose` to include extras such as the commit URL or a short message variant. JSON schema is unchanged.
- Local workspace check: use `--check-only` to validate a workspace without uploading. Produces `workspace_health` in JSON (no token/network required).
- Dry-run planning: use `--dry-run` to compute a plan vs remote without uploading. Returns `dry_run: true`, `dry_run_summary {added, modified:null, deleted}`, and sample `added_files`/`deleted_files`.
- Testing: see TESTING.md ("Push Testing (2.0)") for offline tests and opt-in live checks with markers/env.
- Intended for early testers only. Carefully review the result on the Hub after pushing.
- Responsibility: You are responsible for complying with Hugging Face Hub policies and applicable laws (e.g., copyright/licensing) for any uploaded content.

Example:
```bash
mlxk2 push --private ./workspace org/model --create --commit "init"
```

This feature is not final and may change or be removed.

## Installation & Parallel Usage

### Development Installation

```bash
# Install 2.0.0-alpha (this branch)
pip install -e /path/to/mlx-knife

# Verify installation
mlxk-json --version  # → mlxk2 2.0.0-alpha.2
mlxk2 --version      # → mlxk2 2.0.0-alpha.2
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

> **📋 Complete API Specification**: See the JSON API spec for comprehensive schema, error codes, and examples: [JSON API Specification](docs/json-api-specification.md)

### Command Structure

All commands follow this JSON response format:

```json
{
    "status": "success|error", 
    "command": "list|health|show|pull|rm|push",
    "data": { /* command-specific data */ },
    "error": null | { "message": "...", "details": "..." }
}
```

### Examples

For full, up-to-date examples for every command, refer to the spec: [JSON API Specification](docs/json-api-specification.md)

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
        "hash": "a5339a41b2e3abcdefgh1234567890ab12345678",
        "size_bytes": 4613734656,
        "last_modified": "2024-10-15T08:23:41Z",
        "framework": "MLX",
        "model_type": "chat",
        "capabilities": ["text-generation", "chat"],
        "health": "healthy",
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
    "healthy": [
      { "name": "mlx-community/Phi-3-mini-4k-instruct-4bit", "status": "healthy", "reason": "Model is healthy" }
    ],
    "unhealthy": [],
    "summary": { "total": 1, "healthy_count": 1, "unhealthy_count": 0 }
  },
  "error": null
}
```

#### Show Model Details
```bash
mlxk-json show "Phi-3-mini" --json --files
# Output (simplified):
{
  "status": "success",
  "command": "show",
  "data": {
    "model": {
      "name": "mlx-community/Phi-3-mini-4k-instruct-4bit",
      "hash": "a5339a41b2e3abcdefgh1234567890ab12345678",
      "size_bytes": 4613734656,
      "framework": "MLX",
      "model_type": "chat",
      "capabilities": ["text-generation", "chat"],
      "last_modified": "2024-10-15T08:23:41Z",
      "health": "healthy",
      "cached": true
    },
    "files": [
      {"name": "config.json", "size": "1.2KB", "type": "config"},
      {"name": "model.safetensors", "size": "2.3GB", "type": "weights"}
    ],
    "metadata": null
  },
  "error": null
}
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

> **🔗 Integration Reference**: External projects should implement against the JSON API spec — this alpha phase validates that implementation matches documentation: [JSON API Specification](docs/json-api-specification.md)

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

# Current status: all current 2.0 tests pass (some optional schema tests may be skipped without extras)
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
- Server and run not included (use 1.x)
- Limited error message UX in some paths (to be refined)

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
- **API Specification**: [JSON API Specification](docs/json-api-specification.md)
- **Documentation**: See `docs/` directory for technical details

**For production use**: Consider MLX-Knife 1.1.0 until 2.0.0-beta is available.

### Alpha Testing Goals
- ✅ Validate JSON API specification matches implementation
- ✅ Real-world integration feedback from external projects  
- ✅ Edge case discovery through broke-cluster usage
- ✅ API stability testing before beta release

---

*MLX-Knife 2.0.0-alpha - Built for automation, tested for reliability, designed for the future.*

## Sponsors

<div align="left" style="display: flex; flex-wrap: wrap; gap: 8px; align-items: center;">
  <a href="https://github.com/tileslauncher" title="Tiles Launcher">
    <img src="https://github.com/tileslauncher.png" alt="Tiles Launcher" width="48" style="width:48px; height:auto; max-width:100%;">
  </a>
</div>

Special thanks to early supporters and users providing feedback during the 2.0 alpha.


## Acknowledgments

- Built for Apple Silicon using the [MLX framework](https://github.com/ml-explore/mlx)
- Models hosted by the [MLX Community](https://huggingface.co/mlx-community) on HuggingFace
- Inspired by [ollama](https://ollama.ai)'s user experience

---

<p align="center">
  <b>Made with ❤️ by The BROKE team <img src="broke-logo.png" alt="BROKE Logo" width="30" style="vertical-align: middle;"></b><br>
  <i>Version 2.0.0-alpha.2 | September 2025</i><br>
  <a href="https://github.com/mzau/broke-cluster">🔮 Next: BROKE Cluster for multi-node deployments</a>
</p> 
