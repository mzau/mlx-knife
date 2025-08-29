# Model Capabilities Extension Plan

## Problem Statement
With the addition of embedding functionality (Issue #26), MLX-Knife will support both **chat/generative** and **embedding** models. Users need to distinguish between model types to understand which commands work with which models:

- **Chat Models**: Work with `mlxk run` and `/v1/chat/completions`
- **Embedding Models**: Work with `mlxk embed` and `/v1/embeddings` 
- **Mixed Models**: Some models may support both capabilities

## Current State Analysis

### Existing `list` Command Structure:
```bash
mlxk list                    # Shows MLX models only (default)
mlxk list --all             # Shows all frameworks + FRAMEWORK column
mlxk list --health          # Shows health status
mlxk list --verbose         # Shows full model names (keeps mlx-community/ prefix)
```

### Existing `show` Command Structure:
```bash
mlxk show <model>           # Basic model info
mlxk show <model> --files   # Include file listing  
mlxk show <model> --config  # Show config.json content
```

## Solution Design

### Model Capability Detection Logic

#### From config.json Analysis:
```python
def detect_model_capabilities(model_path):
    """Detect model capabilities from config.json"""
    config_path = model_path / "config.json"
    
    if not config_path.exists():
        return ["unknown"]
    
    try:
        with open(config_path) as f:
            config = json.load(f)
        
        capabilities = []
        
        # Chat/Generative Detection
        chat_architectures = [
            "LlamaForCausalLM", "QwenForCausalLM", "Phi3ForCausalLM",
            "MistralForCausalLM", "DeepseekForCausalLM"
        ]
        
        # Embedding Detection  
        embed_architectures = [
            "BertModel", "BertForSequenceClassification",
            "XLMRobertaModel", "NomicBertModel"
        ]
        
        architectures = config.get("architectures", [])
        model_type = config.get("model_type", "").lower()
        
        # Check for chat capability
        if (any(arch in architectures for arch in chat_architectures) or 
            "causal" in model_type or "llama" in model_type):
            capabilities.append("chat")
        
        # Check for embedding capability
        if (any(arch in architectures for arch in embed_architectures) or 
            "bert" in model_type or "embed" in config.get("model_name", "").lower()):
            capabilities.append("embed")
        
        # Special cases based on model name patterns
        model_name = config.get("_name_or_path", "").lower()
        if "embed" in model_name or "nomic" in model_name or "mxbai" in model_name:
            capabilities.append("embed")
            
        return capabilities if capabilities else ["unknown"]
        
    except (json.JSONDecodeError, KeyError):
        return ["unknown"]
```

### Enhanced `list` Command

#### New Column in --verbose Mode:
```bash
mlxk list --verbose
# Output:
NAME                                     ID         SIZE       MODIFIED        CAPABILITIES
mlx-community/Phi-3-mini-4k-instruct    a1b2c3d4   2.1 GB     2 days ago      chat
mlx-community/mxbai-embed-large-v1       e5f6g7h8   1.2 GB     1 week ago      embed  
mlx-community/Qwen2.5-0.5B-Instruct     i9j0k1l2   512 MB     3 days ago      chat
nomic-embed-text-v1                      m3n4o5p6   256 MB     1 day ago       embed
```

#### Backwards Compatibility:
```bash
mlxk list              # Same as before - no changes
mlxk list --all        # Add CAPABILITIES column only if --verbose also used
```

### Enhanced `show` Command

#### Basic Info Extension:
```bash
mlxk show "Phi-3-mini"
Model: mlx-community/Phi-3-mini-4k-instruct-4bit
Path: ~/.cache/huggingface/models--mlx-community--Phi-3-mini-4k-instruct-4bit/snapshots/abc123
Snapshot: abc123def456
Size: 2.1 GB
Framework: MLX
Capabilities: chat                    # 👈 NEW
Compatible Commands: mlxk run         # 👈 NEW
Health: [OK]

mlxk show "mxbai-embed"
Model: mlx-community/mxbai-embed-large-v1  
Path: ~/.cache/huggingface/models--mlx-community--mxbai-embed-large-v1/snapshots/def789
Snapshot: def789ghi012
Size: 1.2 GB
Framework: MLX
Capabilities: embed                   # 👈 NEW
Compatible Commands: mlxk embed       # 👈 NEW
Health: [OK]
```

## Implementation Plan

### Phase 1: Core Detection Logic
```python
# File: mlx_knife/model_capabilities.py (new)

def detect_model_capabilities(model_path):
    """Core capability detection logic"""
    
def get_compatible_commands(capabilities):
    """Map capabilities to available commands"""
    command_map = {
        "chat": ["mlxk run", "POST /v1/chat/completions"],
        "embed": ["mlxk embed", "POST /v1/embeddings"],
        "unknown": ["Try mlxk health <model> for details"]
    }
    
def format_capabilities_display(capabilities, compact=False):
    """Format capabilities for different display contexts"""
```

### Phase 2: Integration Points

#### cache_utils.py Extensions:
```python
def list_models(..., show_capabilities=False):
    # Add capabilities column when show_capabilities=True
    # Called from CLI with --verbose flag
    
def show_model(...):
    # Add capabilities and compatible commands section
    capabilities = detect_model_capabilities(model_path)
    print(f"Capabilities: {', '.join(capabilities)}")
    print(f"Compatible Commands: {get_compatible_commands(capabilities)}")
```

#### CLI Argument Extensions:
```python
# mlx_knife/cli.py
list_p.add_argument("--verbose", ..., help="Show full names and model capabilities")
# No new arguments needed for show - always display capabilities
```

### Phase 3: Testing & Validation

#### Test Coverage:
- Unit tests for `detect_model_capabilities()` with various config.json examples
- Integration tests for `list --verbose` output format
- Integration tests for `show` command capability display
- Real model testing with known embedding and chat models

#### Edge Cases:
- Missing config.json files
- Malformed JSON
- Models with both chat and embedding capabilities  
- Unknown/unsupported model types

## User Experience Benefits

### Clear Model Distinction:
```bash
# User wants to do embeddings
mlxk list --verbose | grep embed
mxbai-embed-large-v1    e5f6g7h8   1.2 GB   1 week ago   embed
nomic-embed-text-v1     m3n4o5p6   256 MB   1 day ago    embed

# User wants to do chat
mlxk list --verbose | grep chat  
Phi-3-mini-4k-instruct  a1b2c3d4   2.1 GB   2 days ago   chat
Qwen2.5-0.5B-Instruct   i9j0k1l2   512 MB   3 days ago   chat
```

### Error Prevention:
```bash
mlxk run "mxbai-embed-large-v1"
# Error: Model mxbai-embed-large-v1 is an embedding model, not compatible with run command.
# Use: mlxk embed -m "mxbai-embed-large-v1" -c "your text"

mlxk embed -m "Phi-3-mini" -c "test"  
# Error: Model Phi-3-mini is a chat model, not compatible with embed command.
# Use: mlxk run "Phi-3-mini" --prompt "your text"
```

### Discovery & Education:
```bash
mlxk show "mysterious-model"
# Shows capabilities and exactly which commands work
```

## Implementation Complexity

### Low Complexity:
- ✅ Detection logic using config.json (existing patterns)
- ✅ CLI argument integration (--verbose already exists)
- ✅ Display formatting (follow existing column patterns)

### Medium Complexity:
- 📝 Architecture pattern matching (research needed)
- 📝 Edge case handling for unknown models
- 📝 Comprehensive testing across model types

### High Impact:
- 🎯 Prevents user confusion between model types
- 🎯 Makes embedding models discoverable  
- 🎯 Provides clear usage guidance
- 🎯 Maintains backwards compatibility

## Success Criteria

### Functional:
- [ ] `mlxk list --verbose` shows capabilities column
- [ ] `mlxk show <model>` displays capabilities and compatible commands
- [ ] Detection works for common embedding models (mxbai, nomic)
- [ ] Detection works for common chat models (Phi, Llama, Qwen)
- [ ] Error messages guide users to correct commands

### Quality:
- [ ] Backwards compatibility maintained (no breaking changes)
- [ ] Comprehensive test coverage for detection logic
- [ ] Performance impact negligible (caching config.json reads)
- [ ] Clear, helpful error messages

### User Experience:
- [ ] Users can easily find embedding-capable models
- [ ] Users understand which commands work with which models
- [ ] Discovery of new model types is intuitive
- [ ] Migration path clear for users learning new commands

**Estimated Implementation Time**: 2-3 hours (building on existing patterns)