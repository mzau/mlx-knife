# MLX-Knife 2.0 JSON API Specification

**Specification Version:** 0.1.4
**Status:** Alpha - Subject to change  
**Target:** MLX-Knife 2.0.0

> Based on [GitHub Issue #8](https://github.com/mzau/mlx-knife/issues/8) - Comprehensive JSON output support for all commands

## Motivation

MLX Knife is promoted as a "scriptable" tool, but formatted terminal output makes automation difficult. JSON output enables robust scripting integration and broke-cluster compatibility.

## CLI Usage

All commands require the `--json` flag for JSON output:

```bash
mlxk2 list --json                      # JSON output
mlxk2 list                             # Human-readable output
```

### Version Reporting

- CLI version (human):
  - `mlxk2 --version`
- CLI version (JSON):
  - `mlxk2 --version --json`

JSON output example:
```json
{
  "status": "success",
  "command": "version",
  "data": {
    "cli_version": "2.0.0-alpha",
    "json_api_spec_version": "0.1.2"
  },
  "error": null
}
```

Notes:
- Regular command responses (e.g., `list`, `show`) do not include a separate protocol tag; the spec version is reported by the `version` command in `data.json_api_spec_version`.

## Commands Overview

All commands support consistent JSON output with standardized error handling and exit codes.

### Core Schema Pattern

```jsonc
{
  "status": "success" | "error",
  "command": "list" | "show" | "health" | "pull" | "rm" | "clone" | "version" | "push" | "run" | "server",
  "data": { /* command-specific data */ },
  "error": null | { "type": "string", "message": "string" }
}
```

## Common Model Object

All commands that return model information use the same minimal model object.

- `name`: Full HF name `org/model`.
- `hash`: 40-char snapshot commit of the selected snapshot, or `null`.
- `size_bytes`: Total size in bytes of files under the selected path (snapshot preferred, else model root).
- `last_modified`: ISO-8601 UTC timestamp (with `Z`) of the selected path.
- `framework`: "MLX" | "GGUF" | "PyTorch" | "Unknown".
- `model_type`: "chat" | "embedding" | "base" | "unknown".
- `capabilities`: e.g., ["text-generation", "chat"] or ["embeddings"].
- `health`: "healthy" | "unhealthy".
- `cached`: true.

Notes:
- No human-readable `size` field; only `size_bytes`.
- No human-readable "modified" field; `last_modified` is authoritative.
- No absolute filesystem paths are exposed.

### Supported Commands

| Command | Description | JSON-Only in 2.0 | Alpha Feature |
|---------|-------------|------------------|---------------|
| `list` | List models with metadata and hash codes | ✅ | - |
| `show` | Detailed model inspection with files/config | ✅ | - |
| `health` | Check model integrity and corruption | ✅ | - |
| `pull` | Download models from HuggingFace | ✅ | - |
| `rm` | Delete models from cache | ✅ | - |
| `clone` | Clone models to workspace directory | ✅ | `MLXK2_ENABLE_ALPHA_FEATURES=1` |
| `push` | Upload a local folder to Hugging Face (experimental) | ✅ | `MLXK2_ENABLE_ALPHA_FEATURES=1` |
| `run` | Execute model inference | ✅ | - |
| `serve`/`server` | OpenAI-compatible API server | ✅ | - |

**Note:** Commands marked with Alpha Feature require `MLXK2_ENABLE_ALPHA_FEATURES=1` environment variable to be available.

## Model Discovery & Metadata

### Model Type & Capabilities

**Model Types:**
- `"chat"` - Language models with chat/instruction capability
- `"embedding"` - Embedding models for vector representations
- `"completion"` - Base models for text completion (no chat template)
- `"unknown"` - Cannot determine model type from config

**Capabilities Array:**
- `"text-generation"` - Can generate text
- `"chat"` - Supports chat template/instruction format
- `"embeddings"` - Can generate embeddings
- `"completion"` - Text completion without chat format

### `mlxk-json list [pattern] --json`

**Basic Usage:**
```bash
mlxk-json list --json                        # All models with health status
mlxk-json list "mlx-community" --json        # Filter by pattern  
mlxk-json list "Llama" --json                # Fuzzy matching
```

**Behavior:**
- Equivalent to 1.1.0 columns (NAME/ID/SIZE/MODIFIED/FRAMEWORK/HEALTH) with JSON mapping:
  - NAME → `name`
  - ID → `hash`
  - SIZE → `size_bytes` (bytes, integer)
  - MODIFIED → `last_modified` (ISO-8601 UTC)
  - FRAMEWORK → `framework`
  - HEALTH → `health`
- Health status is always included.
- Pattern filter is a case-insensitive substring match on `name`.

**JSON Schema:**
```json
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
      },
      {
        "name": "mlx-community/mxbai-embed-large-v1",
        "hash": "b5679a5f90abcdef1234567890abcdef12345678",
        "size_bytes": 1200000000,
        "last_modified": "2024-10-20T10:30:15Z",
        "framework": "MLX",
        "model_type": "embedding",
        "capabilities": ["embeddings"],
        "health": "healthy",
        "cached": true
      },
      {
        "name": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        "hash": "e96c7a5f90abcdef1234567890abcdef12345678",
        "size_bytes": 16900000000,
        "last_modified": "2024-09-20T14:15:22Z",
        "framework": "GGUF",
        "model_type": "chat",
        "capabilities": ["text-generation", "chat"],
        "health": "unhealthy",
        "cached": true
      }
    ],
    "count": 12
  },
  "error": null
}
```

**Empty Cache:**
```json
{
  "status": "success",
  "command": "list",
  "data": {
    "models": [],
    "count": 0
  },
  "error": null
}
```

### `mlxk-json health [pattern] --json`

**Usage:**
```bash
mlxk-json health --json                      # Check all models
mlxk-json health "Phi-3" --json              # Check specific pattern
mlxk-json health "Qwen3@e96" --json          # Check specific hash
```

**Healthy Models:**
```json
{
  "status": "success",
  "command": "health",
  "data": {
    "healthy": [
      {
        "name": "mlx-community/Phi-3-mini-4k-instruct-4bit",
        "status": "healthy",
        "reason": "Model is healthy"
      }
    ],
    "unhealthy": [],
    "summary": {
      "total": 1,
      "healthy_count": 1,
      "unhealthy_count": 0
    }
  },
  "error": null
}
```

**Unhealthy Models (Real Scenario):**
```json
{
  "status": "success", 
  "command": "health",
  "data": {
    "healthy": [],
    "unhealthy": [
      {
        "name": "mlx-community/Phi-3-mini-4k-instruct-4bit",
        "status": "unhealthy",
        "reason": "config.json missing"
      },
      {
        "name": "corrupted/model", 
        "status": "unhealthy",
        "reason": "LFS pointers instead of files: model.safetensors"
      }
    ],
    "summary": {
      "total": 2,
      "healthy_count": 0,
      "unhealthy_count": 2
    }
  },
  "error": null
}
```

**Ambiguous Pattern:**
```json
{
  "status": "error",
  "command": "health", 
  "data": null,
  "error": {
    "type": "ambiguous_match",
    "message": "Multiple models match 'Llama'",
    "matches": [
      "mlx-community/Llama-3.2-1B-Instruct-4bit",
      "mlx-community/Llama-3.2-3B-Instruct-4bit"
    ]
  }
}
```

### `mlxk-json show <model> --json`

**Usage:**
```bash
mlxk-json show "Phi-3-mini" --json               # Short name expansion
mlxk-json show "mlx-community/Phi-3-mini" --json # Full name
mlxk-json show "Qwen3@e96" --json                # Specific hash
mlxk-json show "Phi-3-mini" --files --json       # Include file listing
mlxk-json show "Phi-3-mini" --config --json      # Include config.json content
```

**Basic Model Information:**
```json
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
    "metadata": {
      "model_type": "phi3",
      "quantization": "4bit",
      "context_length": 4096,
      "vocab_size": 32064,
      "hidden_size": 3072,
      "num_attention_heads": 32,
      "num_hidden_layers": 32
    }
  },
  "error": null
}
```

**With Files Listing (--files):**
```json
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
      {"name": "model.safetensors", "size": "2.3GB", "type": "weights"},
      {"name": "model-00001-of-00002.safetensors", "size": "1.8GB", "type": "weights"},
      {"name": "model-00002-of-00002.safetensors", "size": "200MB", "type": "weights"},
      {"name": "tokenizer.json", "size": "2.1MB", "type": "tokenizer"},
      {"name": "tokenizer_config.json", "size": "3.4KB", "type": "config"},
      {"name": "special_tokens_map.json", "size": "588B", "type": "config"}
    ],
    "metadata": null
  },
  "error": null
}
```

**With Config Content (--config):**
```json
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
    "config": {
      "architectures": ["Phi3ForCausalLM"],
      "model_type": "phi3",
      "vocab_size": 32064,
      "hidden_size": 3072,
      "intermediate_size": 8192,
      "num_hidden_layers": 32,
      "num_attention_heads": 32,
      "max_position_embeddings": 4096,
      "rope_theta": 10000.0,
      "quantization": {
        "bits": 4,
        "group_size": 64
      }
    },
    "metadata": null
  },
  "error": null
}
```

**Model Not Found:**
```json
{
  "status": "error",
  "command": "show",
  "data": null,
  "error": {
    "type": "model_not_found",
    "message": "No model found matching 'nonexistent-model'"
  }
}
```

**Ambiguous Match:**
```json
{
  "status": "error",
  "command": "show",
  "data": null,
  "error": {
    "type": "ambiguous_match",
    "message": "Multiple models match 'Llama'",
    "matches": [
      "mlx-community/Llama-3.2-1B-Instruct-4bit",
      "mlx-community/Llama-3.2-3B-Instruct-4bit"
    ]
  }
}
```

## Changes in 0.1.2 (Alpha)

- Introduced a common minimal Model Object for consistency across commands.
- Replaced human-readable `size` with machine-friendly `size_bytes`.
- Removed human-readable `modified`; `last_modified` (ISO-8601 UTC) is authoritative.

## Operations

### `mlxk-json pull <model> --json`

**Usage:**
```bash
mlxk-json pull "Phi-3-mini" --json               # Short name expansion
mlxk-json pull "mlx-community/Phi-3-mini" --json # Full name
mlxk-json pull "microsoft/DialoGPT-small" --json # Non-MLX model
```

**Successful Download:**
```json
{
  "status": "success",
  "command": "pull",
  "data": {
    "model": "mlx-community/Phi-3-mini-4k-instruct-4bit",
    "download_status": "success",
    "message": "Successfully downloaded model",
    "expanded_name": "mlx-community/Phi-3-mini-4k-instruct-4bit"
  },
  "error": null
}
```

**Already Exists (Bug - doesn't detect corruption):**
```json
{
  "status": "success",
  "command": "pull",
  "data": {
    "model": "mlx-community/Phi-3-mini-4k-instruct-4bit",
    "download_status": "already_exists",
    "message": "Model mlx-community/Phi-3-mini-4k-instruct-4bit already exists in cache",
    "expanded_name": null
  },
  "error": null
}
```

**Download Failed:**
```json
{
  "status": "error",
  "command": "pull",
  "data": {
    "model": "nonexistent/model",
    "download_status": "failed",
    "message": "",
    "expanded_name": null
  },
  "error": {
    "type": "download_failed",
    "message": "Repository not found for url: https://huggingface.co/api/models/nonexistent/model"
  }
}
```

**Validation Error:**
```json
{
  "status": "error",
  "command": "pull", 
  "data": {
    "model": null,
    "download_status": "error",
    "message": "",
    "expanded_name": null
  },
  "error": {
    "type": "ValidationError",
    "message": "Model name too long: 105/96 characters"
  }
}
```

**Ambiguous Match:**
```json
{
  "status": "error",
  "command": "pull",
  "data": {
    "model": null,
    "download_status": "unknown",
    "message": "",
    "expanded_name": null
  },
  "error": {
    "type": "ambiguous_match",
    "message": "Multiple models match 'Llama'",
    "matches": [
      "mlx-community/Llama-3.2-1B-Instruct-4bit",
      "mlx-community/Llama-3.2-3B-Instruct-4bit"
    ]
  }
}
```

### `mlxk-json rm <model> [--force] --json`

**Usage:**
```bash
mlxk-json rm "Phi-3-mini" --json                 # Direct deletion (no locks)
mlxk-json rm "Phi-3-mini" --force --json         # Force deletion (ignores locks)
mlxk-json rm "locked-model" --json               # Error: requires --force due to locks
```

**Successful Deletion:**
```json
{
  "status": "success", 
  "command": "rm",
  "data": {
    "model": "mlx-community/Phi-3-mini-4k-instruct-4bit",
    "action": "deleted",
    "message": "Successfully deleted mlx-community/Phi-3-mini-4k-instruct-4bit"
  },
  "error": null
}
```

**Model has Active Locks (requires --force):**
```json
{
  "status": "error",
  "command": "rm", 
  "data": {
    "model": "mlx-community/Phi-3-mini-4k-instruct-4bit",
    "locks_detected": true,
    "lock_files": [".locks/model-lock-12345.lock"]
  },
  "error": {
    "type": "locks_present",
    "message": "Model has active locks. Use --force to override."
  }
}
```

**Model Not Found:**
```json
{
  "status": "error",
  "command": "rm",
  "data": null,
  "error": {
    "type": "model_not_found",
    "message": "No models found matching 'nonexistent-model'"
  }
}
```

**Ambiguous Pattern:**
```json
{
  "status": "error",
  "command": "rm",
  "data": {
    "matches": [
      "mlx-community/Llama-3.2-1B-Instruct-4bit",
      "mlx-community/Llama-3.2-3B-Instruct-4bit"
    ]
  },
  "error": {
    "type": "ambiguous_match", 
    "message": "Multiple models match 'Llama'. Please specify which model to delete."
  }
}
```

**Permission Error:**
```json
{
  "status": "error",
  "command": "rm",
  "data": {
    "model": "mlx-community/Phi-3-mini-4k-instruct-4bit"
  },
  "error": {
    "type": "PermissionError",
    "message": "Permission denied: Cannot delete read-only files"
  }
}
```

### `mlxk-json clone <model> <target_dir> --json`

**Requires:** `MLXK2_ENABLE_ALPHA_FEATURES=1`

**Usage:**
```bash
mlxk-json clone "Phi-3-mini" ./workspace --json              # Clone to workspace directory
mlxk-json clone "mlx-community/Phi-3-mini" ./my-model --json # Full name to custom directory
mlxk-json clone "microsoft/DialoGPT-small" ./workspace --json # Non-MLX model
```

**Successful Clone:**
```json
{
  "status": "success",
  "command": "clone",
  "data": {
    "model": "mlx-community/Phi-3-mini-4k-instruct-4bit",
    "clone_status": "success",
    "message": "Cloned to ./workspace",
    "target_dir": "./workspace",
    "expanded_name": "mlx-community/Phi-3-mini-4k-instruct-4bit"
  },
  "error": null
}
```

**Target Directory Not Empty:**
```json
{
  "status": "error",
  "command": "clone",
  "data": {
    "model": null,
    "clone_status": "error",
    "target_dir": "./workspace"
  },
  "error": {
    "type": "ValidationError",
    "message": "Target directory './workspace' already exists and is not empty"
  }
}
```

**Clone Failed:**
```json
{
  "status": "error",
  "command": "clone",
  "data": {
    "model": "nonexistent/model",
    "clone_status": "failed",
    "target_dir": "./workspace"
  },
  "error": {
    "type": "clone_failed",
    "message": "Repository not found for url: https://huggingface.co/api/models/nonexistent/model"
  }
}
```

**Access Denied:**
```json
{
  "status": "error",
  "command": "clone",
  "data": {
    "model": "gated/model",
    "clone_status": "access_denied",
    "target_dir": "./workspace"
  },
  "error": {
    "type": "access_denied",
    "message": "Access denied: gated/private model 'gated/model'. Accept terms and set HF_TOKEN."
  }
}
```

**APFS Filesystem Error:**
```json
{
  "status": "error",
  "command": "clone",
  "data": {
    "model": "org/model",
    "clone_status": "filesystem_error",
    "target_dir": "./workspace"
  },
  "error": {
    "type": "FilesystemError",
    "message": "APFS required for clone operations."
  }
}
```

### `mlxk-json push <dir> <org/model> [--create] [--private] [--branch <b>] [--commit "..."] [--verbose] [--check-only] --json`

**Requires:** `MLXK2_ENABLE_ALPHA_FEATURES=1`

Behavior:
- Requires `HF_TOKEN` env.
- Default branch: `main` (subject to change).
- Fails if repo missing unless `--create` is provided.
- Sends folder as-is to the specified branch using `huggingface_hub.upload_folder`.
 - `--verbose` affects only human output; JSON remains unchanged in structure.
 - `--check-only` performs a local, content-oriented workspace validation and does not contact the Hub (no token required). Results are included under `data.workspace_health`.

Successful Upload (with changes):
```json
{
  "status": "success",
  "command": "push",
  "data": {
    "repo_id": "org/model",
    "branch": "main",
    "commit_sha": "abcdef1234567890abcdef1234567890abcdef12",
    "commit_url": "https://huggingface.co/org/model/commit/abcdef1",
    "repo_url": "https://huggingface.co/org/model",
    "uploaded_files_count": 3,
    "local_files_count": 11,
    "no_changes": false,
    "created_repo": false,
    "change_summary": {"added": 1, "modified": 2, "deleted": 0},
    "message": "Push successful. Clone operations require APFS filesystem.",
    "experimental": true,
    "disclaimer": "Experimental feature (M0: upload only). No validation/filters; review on the Hub."
  },
  "error": null
}
```

No Changes (no-op commit avoided):
```json
{
  "status": "success",
  "command": "push",
  "data": {
    "repo_id": "org/model",
    "branch": "main",
    "commit_sha": null,
    "commit_url": null,
    "repo_url": "https://huggingface.co/org/model",
    "uploaded_files_count": 0,
    "local_files_count": 11,
    "no_changes": true,
    "created_repo": false,
    "message": "No files changed; skipped empty commit.",
    "experimental": true,
    "disclaimer": "Experimental feature (M0: upload only). No validation/filters; review on the Hub.",
    "hf_logs": ["No files have been modified since last commit. Skipping to prevent empty commit."]
  },
  "error": null
}
```

Check-only (no network):
```json
{
  "status": "success",
  "command": "push",
  "data": {
    "repo_id": "org/model",
    "branch": "main",
    "commit_sha": null,
    "commit_url": null,
    "repo_url": "https://huggingface.co/org/model",
    "local_files_count": 11,
    "no_changes": null,
    "created_repo": false,
    "message": "Check-only: no upload performed.",
    "workspace_health": {
      "files_count": 11,
      "total_bytes": 289612345,
      "config": {"exists": true, "valid_json": true, "path": "/path/to/config.json"},
      "weights": {"count": 3, "formats": ["safetensors"], "index": {"has_index": true, "missing": []}, "pattern_complete": true},
      "anomalies": [],
      "healthy": true
    },
    "experimental": true,
    "disclaimer": "Experimental feature (M0: upload only). No validation/filters; review on the Hub."
  },
  "error": null
}
```

Missing Token:
```json
{
  "status": "error",
  "command": "push",
  "data": {
    "repo_id": "org/model",
    "branch": "main",
    "repo_url": "https://huggingface.co/org/model",
    "uploaded_files_count": null,
    "local_files_count": null,
    "no_changes": null,
    "created_repo": false,
    "experimental": true,
    "disclaimer": "Experimental feature (M0: upload only). No validation/filters; review on the Hub."
  },
  "error": {
    "type": "auth_error",
    "message": "HF_TOKEN not set"
  }
}
```

## Error Handling

**All errors follow consistent format with detailed error types:**

### Error Types

**Validation Errors:**
- `ValidationError` - Invalid input (96 char limit, empty names)
- `ambiguous_match` - Multiple models match pattern
- `model_not_found` - No models match pattern

**Network Errors:**
- `download_failed` - HuggingFace API errors, network timeouts
- `NetworkError` - Connection issues

**System Errors:**
- `PermissionError` - File system permission denied
- `OperationError` - Cache corruption, disk full
- `InternalError` - Unexpected system errors

**Error Response Schema:**
```json
{
  "status": "error",
  "command": "pull",
  "data": { /* partial data if available */ },
  "error": {
    "type": "ValidationError",
    "message": "Repository name exceeds HuggingFace Hub limit: 105/96 characters"
  }
}
```

### Real-World Error Examples

**Cache Corruption (Health Check Bug):**
```json
{
  "status": "success",
  "command": "health", 
  "data": {
    "healthy": [],
    "unhealthy": [{
      "name": "mlx-community/Phi-3-mini-4k-instruct-4bit",
      "status": "unhealthy",
      "reason": "config.json missing"
    }],
    "summary": {
      "total": 1,
      "healthy_count": 0,
      "unhealthy_count": 1
    }
  },
  "error": null
}
```

**Pull Refuses Corrupted Model (Bug):**
```json
{
  "status": "success",
  "command": "pull",
  "data": {
    "download_status": "already_exists",
    "message": "Model already exists in cache"
  },
  "error": null
}
```

## Agent Integration Examples

**Model Management Automation:**
```bash
# List all MLX models with hashes
mlxk-json list --json | jq -r '.data.models[] | select(.framework=="MLX") | "\(.name)@\(.hash)"'

# Get model hashes for pattern matching
mlxk-json list "Qwen" --json | jq -r '.data.models[] | .hash'

# Count models by framework
mlxk-json list --json | jq '.data.models | group_by(.framework) | map({framework: .[0].framework, count: length})'

# Health summary
mlxk-json health --json | jq '.data.summary'

# Find unhealthy models
mlxk-json health --json | jq -r '.data.unhealthy[].name'

# Filter by pattern
mlxk-json list "Llama" --json | jq '.data.count'

# Model sizes with hashes
mlxk-json list --json | jq -r '.data.models[] | "\(.name)@\(.hash): \(.size_bytes)"'

# Get detailed model info
mlxk-json show "Phi-3-mini" --json | jq '.data.model'

# List all files in a model
mlxk-json show "Phi-3-mini" --files --json | jq -r '.data.files[] | "\(.name): \(.size)"'

# Extract model config
mlxk-json show "Phi-3-mini" --config --json | jq '.data.config.quantization'
```

**Automated Health Monitoring:**
```bash
#!/bin/bash
# Check if any models are unhealthy
unhealthy_count=$(mlxk-json health --json | jq '.data.summary.unhealthy_count')
if [ "$unhealthy_count" -gt 0 ]; then
  echo "Warning: $unhealthy_count unhealthy models found"
  mlxk-json health --json | jq -r '.data.unhealthy[] | "UNHEALTHY: \(.name) - \(.reason)"'
fi
```

**Batch Operations:**
```bash
# Pull multiple models
for model in "Phi-3-mini" "Llama-3.2-1B"; do
  echo "Pulling $model..."
  mlxk-json pull "$model" --json | jq '.data.download_status'
done

# Clean up old models
mlxk-json list --json | jq -r '.data.models[] | select(.size | test("GB")) | .name' | while read model; do
  echo "Found large model: $model"
done
```

## Design Principles

- **No implementation details:** No cache paths, internal directories, or implementation specifics
- **No user-specific data:** No usernames in paths or environment-dependent information  
- **Consistent schema:** All commands follow same `status/command/data/error` structure
- **Scriptable output:** Rich structured data optimized for `jq` and automation
- **Backward compatible:** Exit codes remain unchanged for script compatibility

## Exit Codes

All commands use consistent exit codes for scripting:

- `0` - Success
- `1` - General error (validation, not found, etc.)
- `2` - Network/download error
- `3` - Permission/filesystem error

## Version History

- **2.0.0-alpha:** JSON-only implementation with `mlxk-json --json`
- **2.0.0-alphha.1:** Full implementation with both JSON and human-readable output
- **2.0.0-alphha.2:** Push function protocol extension (json-0.1.3)
