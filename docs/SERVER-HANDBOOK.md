# MLX Knife Server Handbook

**Version:** 2.0.4-beta.1 (WIP)
**Status:** ⚠️ **WORK IN PROGRESS** - This document will evolve until 2.1 stable release
**Last Updated:** 2025-12-15

> **Audience:** Server operators, DevOps, API consumers
> **Not for:** Developers (see `ARCHITECTURE.md` and ADRs instead)

---

## Quick Start

```bash
# Basic server
mlxk serve --port 8000

# JSON logging (production)
mlxk serve --port 8000 --log-json

# Custom host
mlxk serve --host 0.0.0.0 --port 8000
```

**Requirements:**
- Python 3.9+ (Text models)
- Python 3.10+ (Vision models)
- mlx-lm 0.28.4+
- mlx-vlm 0.3.9+ (optional, for vision)

---

## API Endpoints

### POST /v1/chat/completions

**OpenAI-compatible chat completion endpoint.**

**Request:**
```json
{
  "model": "mlx-community/Llama-3.2-3B-Instruct-4bit",
  "messages": [
    {"role": "user", "content": "Hello!"}
  ],
  "max_tokens": null,
  "temperature": 0.7,
  "stream": false
}
```

**Vision Request (Base64 Images):**
```json
{
  "model": "mlx-community/Llama-3.2-11B-Vision-Instruct-4bit",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "What's in this image?"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
      ]
    }
  ],
  "max_tokens": 2048,
  "temperature": 0.4
}
```

**Response:**
```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1702345678,
  "model": "mlx-community/Llama-3.2-3B-Instruct-4bit",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! How can I help you?"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 12,
    "completion_tokens": 8,
    "total_tokens": 20
  }
}
```

---

### POST /v1/completions

**Legacy completion endpoint (text-only, no chat template).**

**Request:**
```json
{
  "model": "mlx-community/Llama-3.2-3B-Instruct-4bit",
  "prompt": "Once upon a time",
  "max_tokens": 100,
  "temperature": 0.7
}
```

---

### GET /v1/models

**List available models.**

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "mlx-community/Llama-3.2-3B-Instruct-4bit",
      "object": "model",
      "created": 1702345678,
      "owned_by": "mlx-community"
    }
  ]
}
```

---

### GET /health

**Server health check (200 OK if server is running).**

---

## Features & Capabilities

### Vision Support (2.0.4-beta.1)

See `examples/vision_pipe.sh` for a practical Vision→Text pipeline example (CLI).

**Supported:**
- ✅ Base64 data URLs (`data:image/jpeg;base64,...`)
- ✅ Multiple images (up to 5 per request)
- ✅ Formats: JPEG, PNG

**Limits:**
- **Per-image:** 20 MB max
- **Total:** 50 MB max per request
- **Count:** 5 images max per request

**Important Characteristics:**

- **Stateless Server:** No server-side state required
- **Sequential Images:** Only images from the **last user message** are processed (OpenAI API compliant)
- **Each request is independent:** No "shift-window" context like text models (Metal memory limitations)

### Stable Image IDs (History-Based)

**Problem:** How to maintain stable "Image 1, 2, 3..." numbering across multiple requests?

**Solution:** The conversation history IS the session.

The server scans the full `messages[]` array (which clients send with each request per OpenAI API) and assigns IDs chronologically based on content hash:

```
Request 1: beach.jpg (hash: 5c691ddb) → Image 1
Request 2: beach.jpg + mountain.jpg in history → Image 1, Image 2
Request 3: Re-upload beach.jpg → Still Image 1 (hash match)
```

**Properties:**
- ✅ **100% OpenAI API compatible** — standard messages[] format, no custom headers
- ✅ **Stateless server** — no registry, no TTL, no cleanup
- ✅ **Content-hash deduplication** — same image always gets same ID
- ✅ **Cross-model workflows** — "Image 1" stable across Vision↔Text model switches

**Client Responsibility:**
- Maintain full conversation history in `messages[]` array
- Same content = same ID (content-hash based)

**Python Version:**
- ✅ Python 3.10+ required (mlx-vlm dependency)
- ❌ Python 3.9: Vision requests → HTTP 501

---

### Token Limits: Vision vs Text Models

**Critical Difference:** Vision and text models use different `max_tokens` strategies.

#### Text Models (MLXRunner)

**Strategy:** Shift-window context management
- Conversation history maintained in context buffer
- Server reserves space for history

**Defaults:**
- **Server:** `context_length / 2` (reserve half for history, half for generation)
- **CLI:** `context_length` (full context, no reservation)

**Example:**
- Llama-3.2-3B (128K context) → Server default: 64K max_tokens

#### Vision Models (VisionRunner)

**Strategy:** Stateless processing
- Each request is independent (no conversation history in context)
- Metal limitations prevent context preservation

**Defaults:**
- **Server/CLI:** `2048` tokens (conservative, works for all models)

**Rationale:**
- No need for `/2` division (no history to reserve)
- Vision inference is slow → 2048 adequate for image descriptions
- Prevents accidentally generating 64K+ tokens

**Override:**
```json
{
  "model": "mlx-community/Llama-3.2-11B-Vision-Instruct-4bit",
  "messages": [...],
  "max_tokens": 4096  // Explicit override
}
```

---

### Memory-Aware Loading (ADR-016)

**Pre-load memory checks prevent OOM crashes.**

#### Vision Models
- **Threshold:** 70% system RAM
- **Behavior:** Model size > 70% → HTTP 507 (Insufficient Storage)
- **Rationale:** Vision Encoder has unpredictable per-image overhead

**Example (64GB system):**
- Llama-3.2-11B-Vision (5.6GB) → ✅ Loads (8.75% of RAM)
- Llama-3.2-90B-Vision (46.4GB) → ❌ HTTP 507 (72.5% of RAM)

#### Text Models
- **Threshold:** 70% system RAM
- **Behavior:** Model size > 70% → **Warning only** (backwards compatible)
- **Rationale:** Text models swap gracefully, no hard memory spikes

---

### Streaming (SSE - Server-Sent Events)

#### Text Models
- ✅ **True streaming:** Tokens streamed as generated
- **Format:** SSE (`data: {...}\n\n`)
- **Completion:** `data: [DONE]\n\n`

#### Vision Models
- ⚠️ **Graceful degradation:** SSE emulation (batch result split into chunks)
- **Reason:** mlx-vlm doesn't guarantee streaming support
- **Behavior:** Returns full result via SSE format for client compatibility

**Request:**
```json
{
  "model": "mlx-community/Llama-3.2-3B-Instruct-4bit",
  "messages": [...],
  "stream": true
}
```

**Response (SSE stream):**
```
data: {"id":"chatcmpl-...","choices":[{"delta":{"content":"Hello"},"index":0}],...}

data: {"id":"chatcmpl-...","choices":[{"delta":{"content":" there"},"index":0}],...}

data: [DONE]
```

---

## Configuration

### Environment Variables

```bash
# Server binding
MLXK2_HOST=0.0.0.0
MLXK2_PORT=8000

# Logging
MLXK2_LOG_JSON=1          # JSON logs (production)
MLXK2_LOG_LEVEL=info      # debug|info|warning|error

# Feature gates (beta features)
MLXK2_ENABLE_PIPES=1      # Unix pipe integration (2.0.4-beta.1)
```

### Supervised Mode (Default)

**Behavior:**
- Server auto-restarts on crashes
- Logs go to stderr
- `--log-json` produces 100% JSON output

**Start:**
```bash
mlxk serve --port 8000 --log-json
```

### Direct Mode (Development)

**Behavior:**
- No auto-restart
- Direct uvicorn process

**Start:**
```bash
python -m mlxk2.core.server_base
```

---

## HTTP Status Codes

### Success
- **200 OK:** Request successful
- **201 Created:** Resource created (future)

### Client Errors (4xx)
- **400 Bad Request:** Invalid input (e.g., missing images for vision model, too many images)
- **404 Not Found:** Model not found in cache

### Server Errors (5xx)
- **500 Internal Server Error:** Unexpected backend failure
- **501 Not Implemented:** Feature not supported (e.g., vision on Python 3.9)
- **503 Service Unavailable:** Server shutting down
- **507 Insufficient Storage:** Memory constraints violated (vision model >70% RAM)

---

## Performance Characteristics

### Model Loading
- **Time:** ~5-10 seconds (first request only)
- **Caching:** Model stays loaded until server restart or model switch
- **Memory:** Held in RAM until explicitly unloaded

### Inference Speed

**Text Models:**
- **Typical:** 20-50 tokens/sec (depends on model size, hardware)
- **Streaming:** Real-time token output

**Vision Models:**
- **Slower than text:** Vision Encoder adds overhead
- **Per-image:** ~2-5 seconds baseline + generation time
- **Multiple images:** Linear scaling (no batching in 2.0.4-beta.1)

### Concurrent Requests
- **Current:** Sequential processing (one request at a time)
- **Reason:** Metal backend, single GPU
- **Future:** May add request queuing

---

## Troubleshooting

### Vision Model Fails on Python 3.9

**Symptom:** HTTP 501 "Vision models require Python 3.10+"

**Solution:**
```bash
# Upgrade Python
pyenv install 3.10
pyenv local 3.10
pip install mlx-lm mlx-vlm
```

### Memory Constraint Errors (HTTP 507)

**Symptom:** `Model requires XGB but only YGB available (70% of system RAM)`

**Solutions:**
1. Use smaller quantized model (e.g., 4-bit instead of 8-bit)
2. Add more system RAM
3. Try different model architecture

### Vision Responses Too Short

**Symptom:** Responses truncated mid-sentence

**Cause:** Default `max_tokens: 2048` might be too low for complex descriptions

**Solution:**
```json
{
  "model": "mlx-community/Llama-3.2-11B-Vision-Instruct-4bit",
  "messages": [...],
  "max_tokens": 4096  // Increase limit
}
```

### Image Upload Fails (HTTP 400)

**Common causes:**
- Image size > 20MB per image
- Total size > 50MB
- More than 5 images
- External URLs (not supported, use Base64)
- Invalid Base64 encoding

**Solution:** Resize images, reduce count, or check encoding

---

## Limits Summary

| Resource | Limit | Reason |
|----------|-------|--------|
| Images per request | 5 | Metal OOM prevention |
| Image size | 20 MB | Metal OOM prevention |
| Total image size | 50 MB | Metal OOM prevention |
| Vision model RAM | 70% system | Metal OOM prevention |
| Text model RAM | 70% (warning) | Swap tolerance |
| Vision max_tokens | 2048 (default) | Stateless, slow inference |
| Text max_tokens | context_length/2 | Shift-window reservation |

---

## Migration Guide

### From 2.0.3 → 2.0.4-beta.1

**New Features:**
- ✅ Vision support (Python 3.10+)
- ✅ Memory pre-load checks (HTTP 507)
- ✅ Unix pipe integration (`MLXK2_ENABLE_PIPES=1`)

**Breaking Changes:**
- ⚠️ Vision models: `max_tokens` default changed from 1024 → 2048
- ⚠️ Memory checks: Vision models >70% RAM now blocked (was: no check)

**Recommendations:**
- Update clients expecting vision `max_tokens: 1024` to handle 2048
- Monitor for HTTP 507 errors (memory constraints)
- Test vision workflows on Python 3.10+

---

## References

- **API Schema:** `docs/json-api-specification.md`
- **Architecture Principles:** `docs/ARCHITECTURE.md`
- **Testing Details:** `TESTING-DETAILS.md`
- **ADR-012:** Vision Support (development decisions)
- **ADR-016:** Memory-Aware Loading (development decisions)

---

## Appendix: Client Requirements

> **Audience:** Client developers integrating with MLX Knife server

### OpenAI API Compliance

Clients MUST follow the OpenAI Chat Completions API format. MLX Knife is designed to work with any OpenAI-compatible client.

### Conversation History

**Clients MUST send the full conversation history** with each request:

```json
{
  "model": "mlx-community/Llama-3.2-11B-Vision-Instruct-4bit",
  "messages": [
    {"role": "user", "content": [...]},
    {"role": "assistant", "content": "..."},
    {"role": "user", "content": [...]}
  ]
}
```

**Why:** The server reconstructs stable image IDs from the history. Without full history, image numbering restarts at 1 with each request.

### Vision: Stateless Prompt, History-Based IDs

**Important architectural distinction for Vision requests:**

| Aspect | Behavior | Reason |
|--------|----------|--------|
| **Prompt to model** | Only last user message | Prevents pattern reproduction (model copying old mappings) |
| **Image ID assignment** | Full history scanned | Consistent numbering across session (Image 1, 2, 3...) |

**What this means:**
- The Vision model does NOT see previous assistant responses
- But image numbering remains stable across the conversation
- Follow-up questions about image descriptions should use a **Text model** (which has full history)

**Recommended workflow:**
```
1. Vision model: User sends beach.jpg → "Image 1 shows a beach..."
2. Vision model: User sends mountain.jpg → "Image 2 shows a mountain..."
3. Text model: User asks "Compare these two locations" → Full context available
```

**Rationale:**
- Vision models can't "see" previous images anyway (Metal memory limitations)
- Sending history caused pattern reproduction (model hallucinating mappings)
- Clean separation: Vision=describe, Text=discuss

### Vision Messages Format

**Multimodal content** uses the OpenAI array format:

```json
{
  "role": "user",
  "content": [
    {"type": "text", "text": "What's in this image?"},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
  ]
}
```

**Image URLs:**
- ✅ **Base64 Data URLs:** `data:image/jpeg;base64,/9j/4AAQ...`
- ❌ **HTTP URLs:** Not supported (no external fetching)

**Supported formats:** JPEG, PNG, GIF, WebP

### Cross-Model Workflows (Vision → Text)

When switching from Vision to Text model mid-conversation:

1. **Client:** Continue sending full history (including previous image_url content)
2. **Server:** Automatically filters images for text models, preserves text context
3. **Result:** Text model sees `[Image 1: beach...]` placeholders instead of binary data

**Example workflow:**
```
1. Vision model: User sends beach.jpg → "Image 1 shows a beach"
2. Vision model: User sends mountain.jpg → "Image 2 shows a mountain"
3. Switch to Text model: User asks "Which is better for vacation?"
4. Text model: Can reference "Image 1" and "Image 2" from context
```

### Image Deduplication

Same image content = same ID (content-hash based).

**Client behavior:**
- Re-uploading the same image → Server assigns same ID
- No client-side deduplication needed

### Image ID Persistence (100% OpenAI-Compatible)

**Problem:** How do Image IDs remain stable across Vision→Text→Vision workflows when clients drop Base64 data from history (storage optimization)?

**Solution:** The server **reads its own filename mapping tables** from assistant responses.

**Workflow:**

1. **Request 1 (Vision):** Client sends beach.jpg
   ```json
   {"role": "user", "content": [
     {"type": "text", "text": "describe"},
     {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
   ]}
   ```

2. **Server Response:** Includes filename mapping table
   ```
   A sandy beach with blue water.

   <!-- mlxk:filenames -->
   | Image | Filename |
   |-------|----------|
   | 1 | image_5733332c.jpeg |
   ```

3. **Client Storage Optimization:** Client can **drop Base64 from history**, keep only:
   ```json
   {"role": "user", "content": "describe"}
   {"role": "assistant", "content": "A sandy beach...\n\n<!-- mlxk:filenames -->\n..."}
   ```

4. **Request 3 (Vision after Text):** Client sends mountain.jpg with text-only history
   ```json
   {
     "messages": [
       {"role": "user", "content": "describe"},  // No Base64!
       {"role": "assistant", "content": "Beach...\n\n| 1 | image_5733332c.jpeg |"},
       {"role": "user", "content": "What color?"},
       {"role": "assistant", "content": "Blue."},
       {"role": "user", "content": [
         {"type": "text", "text": "new picture"},
         {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
       ]}
     ]
   }
   ```

5. **Server Reconstruction:** Server scans history:
   - Finds `<!-- mlxk:filenames -->` marker in assistant response
   - Parses: `image_5733332c.jpeg` → Image ID 1
   - Assigns: mountain.jpg → Image ID 2 ✅

**Benefits:**
- ✅ **Zero client changes** - Works with standard OpenAI API
- ✅ **Storage optimization** - Client can drop large Base64 data (2 MB → 2 KB)
- ✅ **100% OpenAI-compatible** - No protocol extensions needed
- ✅ **Stateless server** - No server-side session state required
- ✅ **Scales to 100+ images** - Clients only store small text mappings

**Client Recommendations:**
- **After first Vision request:** Drop Base64 image_url from history, keep text + assistant response
- **Store locally:** Small thumbnails for UI (~20 KB/image via IndexedDB)
- **History format:** Text-only user messages + full assistant responses (with mapping tables)

**Example client storage (100 images):**
- ❌ **Before:** 100 images × 2 MB Base64 = 200 MB (exceeds browser limits)
- ✅ **After:** 100 thumbnails × 20 KB + text history = ~2 MB (fits in IndexedDB)

---

## Changelog

- **2025-12-15:** 2.0.4-beta.1 WIP
  - Vision support: Base64 images, multiple images, limits
  - History-based stable image IDs (stateless, OpenAI-compatible)
  - **NEW:** Server reads mapping tables from assistant responses (Image ID persistence without Base64)
  - Vision: Stateless prompt + history-based IDs (pattern reproduction fix)
  - Vision: temperature=0.0 (greedy sampling, reduces hallucinations)
  - Vision vs Text max_tokens strategy
  - Memory-aware loading (HTTP 507)
  - Feature gates and troubleshooting

---

**📝 Note:** This handbook will be updated continuously until 2.1 stable release. Check version header for freshness.
