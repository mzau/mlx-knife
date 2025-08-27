# MLX-Knife 2.0 JSON API Specification

**Specification Version:** 0.1.1 
**Status:** Alpha - Subject to change  
**Target:** MLX-Knife 2.0.0

> Based on [GitHub Issue #8](https://github.com/mzau/mlx-knife/issues/8) - Comprehensive JSON output support for all commands

## Motivation

MLX Knife is promoted as a "scriptable" tool, but formatted terminal output makes automation difficult. JSON output enables robust scripting integration and broke-cluster compatibility.

## CLI Usage

All commands require the `--json` flag for JSON output:

```bash
mlxk-json list --json                  # JSON output (2.0.0-alpha+)
mlxk list --json                       # JSON output (2.0.0+)
mlxk list                              # Human-readable output (2.0.0+)
```

**Version Support:**
- **2.0.0-alpha:** Only `mlxk-json --json` available (JSON-only implementation)
- **2.0.0+:** Both `mlxk --json` and `mlxk-json --json` for JSON output
- **2.0.0+:** `mlxk` without `--json` for human-readable output

## Commands Overview

All commands support consistent JSON output with standardized error handling and exit codes.

### Core Schema Pattern

```json
{
  "status": "success" | "error",
  "command": "list" | "show" | "health" | "pull" | "rm",
  "data": { /* command-specific data */ },
  "error": null | { "type": "string", "message": "string" }
}
```

### Supported Commands

| Command | Description | JSON-Only in 2.0 |
|---------|-------------|------------------|
| `list` | List models with metadata and hash codes | ✅ |
| `show` | Detailed model inspection with files/config | ✅ |
| `health` | Check model integrity and corruption | ✅ |
| `pull` | Download models from HuggingFace | ✅ |
| `rm` | Delete models from cache | ✅ |
| `run` | Execute model inference | ❌ Not in 2.0 |
| `server` | OpenAI-compatible API server | ❌ Not in 2.0 |

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
mlxk-json list --json                        # All models
mlxk-json list "mlx-community" --json        # Filter by pattern
mlxk-json list "Llama" --json                # Fuzzy matching
```

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
        "size": "4.3GB",
        "framework": "MLX",
        "model_type": "chat",
        "capabilities": ["text-generation", "chat"],
        "cached": true,
        "last_modified": "2024-10-15T08:23:41Z"
      },
      {
        "name": "mlx-community/mxbai-embed-large-v1",
        "hash": "b5679a5f90abcdef1234567890abcdef12345678",
        "size": "1.2GB",
        "framework": "MLX",
        "model_type": "embedding",
        "capabilities": ["embeddings"],
        "cached": true,
        "last_modified": "2024-10-20T10:30:15Z"
      },
      {
        "name": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        "hash": "e96c7a5f90abcdef1234567890abcdef12345678",
        "size": "16.9GB", 
        "framework": "GGUF",
        "model_type": "chat",
        "capabilities": ["text-generation", "chat"],
        "cached": true,
        "last_modified": "2024-09-20T14:15:22Z"
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
      "size": "4.3GB",
      "framework": "MLX",
      "model_type": "chat",
      "capabilities": ["text-generation", "chat"],
      "last_modified": "2024-10-15T08:23:41Z",
      "health": "healthy",
      "files_count": 15,
      "total_size_bytes": 4613734656
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
      "size": "4.3GB",
      "framework": "MLX",
      "model_type": "chat",
      "capabilities": ["text-generation", "chat"]
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
      "size": "4.3GB",
      "framework": "MLX",
      "model_type": "chat",
      "capabilities": ["text-generation", "chat"]
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
    "unhealthy": [{
      "name": "mlx-community/Phi-3-mini-4k-instruct-4bit",
      "status": "unhealthy",
      "reason": "config.json missing"
    }]
  }
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
  }
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
mlxk-json list --json | jq -r '.data.models[] | "\(.name)@\(.hash): \(.size)"'

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
- **2.0.0:** Full implementation with both JSON and human-readable output