# Issue #26 Summary: Embeddings Endpoint Implementation

## Issue Overview
**Title**: Add `/v1/embeddings` endpoint for OpenAI-compatible embedding generation  
**Type**: Feature Request  
**Status**: Open  
**Complexity**: Medium (4-6 hours estimated)

## Original Issue Description

### Core Requirements
Add a new `/v1/embeddings` endpoint to MLX-Knife's server that provides stateless embedding generation for previously pulled MLX models.

### Key Design Principles
- **Stateless Operation**: No vector database, no memory, no intelligent model auto-selection
- **OpenAI Compatibility**: Standard JSON response format matching OpenAI embeddings API
- **Context-Free Server**: Simple load-model-and-return-vectors operation
- **User Responsibility**: Client manages model selection, vector storage, and reindexing

### Endpoint Specification
```
POST /v1/embeddings
```

#### Request Parameters
- `model` (required): Name of the embedding model to use
- `input` (required): String or array of strings to embed
- `encoding_format` (optional): Response format - "float" or "base64" 
- `normalize` (optional): Whether to normalize embeddings (default: true)
- `max_length` (optional): Maximum input length limit

#### Response Format
Standard OpenAI-compatible JSON structure:
```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "index": 0,
      "embedding": [0.1, 0.2, 0.3, ...]
    }
  ],
  "model": "model-name",
  "usage": {
    "prompt_tokens": 10,
    "total_tokens": 10
  }
}
```

### Use Cases
- **Agent Frameworks**: Integration with AI agent systems requiring embeddings
- **RAG Pipelines**: Retrieval-Augmented Generation implementations  
- **External Clients**: Third-party tools needing embedding generation
- **Semantic Search**: Applications requiring text similarity matching

### Boundaries & Limitations
- **No Persistence**: Server doesn't store or remember embeddings
- **No Auto-Selection**: User must specify exact model name
- **No Quality Assurance**: User responsible for model appropriateness
- **Single Response**: Always returns complete JSON (non-streaming)

## Follow-Up Comment: CLI Integration

### Additional CLI Requirement
The original author added a follow-up comment requesting a complementary CLI subcommand alongside the server endpoint:

```bash
mlxk embed <MODEL> --input "text content"
```

### CLI Specifications
- **Non-Streaming**: Always returns complete JSON response
- **Input Options**: Support both `--input "text"` and `--input-file path/to/file`
- **OpenAI-Compatible Output**: Same JSON structure as server endpoint
- **Separation of Concerns**: Keep `mlxk run` command for generative models only

### CLI Use Cases
- **Development Testing**: Quick embedding generation during development
- **Batch Processing**: File-based embedding generation
- **Scripting**: Integration with shell scripts and automation
- **Local Processing**: Offline embedding generation without server

## Technical Implementation Strategy

### Architecture Pattern
Follow the existing `run` command architecture:
- **Shared Core**: `embed_model_core()` function used by both CLI and server
- **CLI Wrapper**: `embed_model()` in `cache_utils.py` (similar to `run_model()`)
- **Server Endpoint**: `/v1/embeddings` route (similar to `/v1/chat/completions`)

### Reusable Components
- `resolve_single_model()` for model path resolution
- `detect_framework()` for MLX compatibility checking
- `get_or_load_model()` for server-side model caching
- Existing error handling and response patterns

### File Structure
- `mlx_knife/embedding_utils.py` - Core embedding logic
- `mlx_knife/cache_utils.py` - CLI wrapper function  
- `mlx_knife/cli.py` - CLI command definitions
- `mlx_knife/server.py` - REST endpoint implementation

## Expected Benefits

### For Users
- **Unified Interface**: Consistent embedding access via CLI and API
- **OpenAI Compatibility**: Drop-in replacement for OpenAI embedding API
- **Local Processing**: No external API dependencies for embedding generation
- **Model Flexibility**: Use any compatible MLX embedding model

### For Ecosystem
- **Integration Ready**: Standard API for external tool integration
- **Development Friendly**: Easy testing and experimentation via CLI
- **Stateless Design**: Scalable and predictable behavior
- **Performance**: Direct MLX backend without additional abstraction layers

## Compatibility Considerations

### MLX Framework
- Requires MLX-compatible embedding models
- Leverages existing MLX model loading infrastructure
- Benefits from MLX performance optimizations

### OpenAI API
- Request/response format matches OpenAI embeddings API
- Parameter names and behavior consistent with OpenAI
- Easy migration from OpenAI to local MLX-Knife

### Existing Codebase  
- Follows established architectural patterns
- Reuses existing model resolution and error handling
- Maintains separation between generative (`run`) and embedding functionality

## Implementation Priority
**Medium Priority** - Valuable feature that extends MLX-Knife's capabilities without disrupting existing functionality. The stateless design and reuse of existing patterns makes this a relatively low-risk addition with clear user benefits.