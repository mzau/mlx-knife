# MLX Knife Server Handbook

**Version:** 2.0.4-beta.8 (WIP)
**Status:** ⚠️ **WORK IN PROGRESS** - This document will evolve until 2.1 stable release
**Last Updated:** 2026-01-20

> **Audience:** Server operators, DevOps, API consumers
> **For implementation details:** See `ARCHITECTURE.md` and `docs/ADR/` (developer documentation)

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
- Python 3.10+ (Vision and Audio models)
- mlx-lm 0.28.4+
- mlx-vlm ≥0.3.10 (required for audio; currently GitHub-only, not yet on PyPI)

---

## OpenAI API Compatibility

MLX Knife implements a **subset** of the OpenAI API with documented behavioral differences.

### Supported Endpoints

| Endpoint | Status | Notes |
|----------|--------|-------|
| `/v1/chat/completions` | ✅ Supported | Text, Vision (`image_url`), Audio (`input_audio`) |
| `/v1/completions` | ✅ Supported | Legacy text completion |
| `/v1/models` | ✅ Supported | Extended with `context_length` field |
| `/health` | ✅ Custom | MLX Knife extension |

### Not Implemented

| Endpoint | Status |
|----------|--------|
| `/v1/embeddings` | ❌ Planned (ADR-015) |
| `/v1/audio/*` | ❌ Not planned (use `input_audio` in chat) |
| `/v1/files` | ❌ Not planned |
| `/v1/moderations` | ❌ Not planned |
| `/v1/responses` | ❌ Not planned |

### Authentication

MLX Knife **ignores** authentication headers. The server accepts but does not validate:
- `Authorization: Bearer ...`
- Any API key

**Note:** For production deployments requiring authentication, use a reverse proxy (nginx, Caddy).

### Request Headers

```
Content-Type: application/json  (required)
Authorization: Bearer ...       (optional, ignored)
```

### Behavioral Deviations from OpenAI

These are intentional design choices, not bugs:

| Behavior | OpenAI | MLX Knife | Reason |
|----------|--------|-----------|--------|
| Vision history | Full history to model | Only last user message | Prevents pattern reproduction (hallucinations) |
| Image URLs | HTTP URLs + Base64 + File IDs | Base64 data URLs only | No external fetching |
| Audio+Vision | Both processed | Audio silently ignored | mlx-vlm limitation |
| Multi-audio | Supported | 1 per request | mlx-vlm limitation |
| Error format | `{"error": {"message", "type", "code"}}` | ADR-004 envelope (see below) | Richer error context |
| `max_completion_tokens` | Preferred | Not supported (use `max_tokens`) | Legacy compatibility |
| HTTP 507 | Not used | Memory constraint | Explicit OOM prevention |

### Error Response Format

MLX Knife uses an extended error envelope (ADR-004), not the OpenAI format:

```json
{
  "status": "error",
  "error": {
    "type": "validation_error",
    "message": "No user message found",
    "retryable": false
  },
  "request_id": "abc123..."
}
```

**Error types:** `validation_error`, `model_not_found`, `internal_error`, `server_shutdown`, `insufficient_memory`, `access_denied`

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
  "temperature": 0.4,
  "chunk": 1
}
```

**Audio Request (OpenAI `input_audio` format):**
```json
{
  "model": "mlx-community/gemma-3n-E2B-it-4bit",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Transcribe what is spoken in this audio"},
        {
          "type": "input_audio",
          "input_audio": {
            "data": "<base64-encoded>",
            "format": "wav"
          }
        }
      ]
    }
  ],
  "max_tokens": 2048,
  "temperature": 0.0
}
```

**Supported audio formats:** `wav`, `mp3` (or `mpeg` alias)

**mlx-knife Extension Parameters:**
- `chunk` (integer, optional): Batch size for vision processing (default: 1). Controls how many images are processed per inference session. Higher values may trigger OOM on resource-constrained systems. Maximum: 5 (enforced by server).

**Default chunk size:**
1. Request parameter `chunk` (highest priority)
2. Server startup: `mlxk serve --chunk N`
3. Environment: `MLXK2_VISION_CHUNK_SIZE=N`
4. Default: 1 (maximum safety)

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

Returns all cached models that are healthy and runtime-compatible.
Models are sorted with preloaded model first (if any), then alphabetically.

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "mlx-community/Llama-3.2-3B-Instruct-4bit",
      "object": "model",
      "owned_by": "mlx-knife-2.0",
      "permission": [],
      "context_length": 8192
    }
  ]
}
```

**Fields:**
- `id`: Model identifier (HuggingFace name or workspace path)
- `object`: Always `"model"` (OpenAI-compatible)
- `owned_by`: `"mlx-knife-2.0"` for cached models, `"workspace"` for local directories
- `permission`: Empty array (OpenAI legacy field)
- `context_length`: Maximum context window in tokens (may be `null` if unavailable)

**Why context_length matters:**

MLX Knife uses **client-side context management** (unlike OpenAI's server-side history):
- **Vision models:** Fully stateless - client holds entire conversation history
- **Text models:** Shift-window (context_length / 2 reserved for history on server)
- **Clients need this** to manage conversation pruning and token budgets
- **Load balancing:** BROKE Cluster and similar tools use this for scheduling decisions

Note: LM Studio provides similar field as `max_context_length`.

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
- ✅ Formats: JPEG, PNG, GIF, WebP

**Limits:**
- **Per-image:** 20 MB max
- **Count:** 5 images max per request

**Important Characteristics:**

- **Stateless Server:** No server-side state required
- **Sequential Images:** Only images from the **last user message** are processed (OpenAI API compliant)
- **Each request is independent:** No "shift-window" context like text models (Metal memory limitations)

#### Stable Image IDs (History-Based)

**Problem:** How to maintain stable "Image 1, 2, 3..." numbering across multiple requests?

**Solution:** The conversation history IS the session.

The server scans the full `messages[]` array (which clients send with each request per OpenAI API) and assigns IDs chronologically based on content hash:

```
Request 1: beach.jpg (hash: 5c691ddb) → Image 1
Request 2: beach.jpg + mountain.jpg in history → Image 1, Image 2
Request 3: Re-upload beach.jpg → Still Image 1 (hash match)
```

**Properties:**
- ✅ **Standard messages[] format** — no custom headers or protocol extensions
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

### Audio Support (2.0.4-beta.8)

**Native audio input** for audio-capable models (Gemma-3n).

**Supported:**
- ✅ OpenAI `input_audio` format (Base64-encoded)
- ✅ Formats: WAV, MP3
- ✅ Temperature 0.0 (greedy sampling for transcription consistency)

**Limits:**
- **Per-audio:** 5 MB max (~2-3 minutes at 16kHz mono)
- **Count:** 1 audio per request (multi-audio blocked)

**Important Characteristics:**

- **Stateless Server:** Same as Vision — no server-side state
- **Single Audio:** Only one audio file per request (mlx-vlm limitation)
- **Audio+Vision:** When both present, audio is silently ignored (mlx-vlm behavior)
- **Temperature:** Fixed at 0.0 for transcription consistency (CLI default: 0.2)

**Audio-Capable Models:**
- `gemma-3n` (Google): Vision + Audio + Text
- Qwen3-Omni: Not supported (mlx-lm architecture missing)

**History Handling:**

When switching from Audio to Text model mid-conversation:
- Server filters `input_audio` content blocks
- Text model sees `[n audio(s) were attached]` placeholder

**Python Version:**
- ✅ Python 3.10+ required (same as Vision)
- ❌ Python 3.9: Audio requests → HTTP 501

---

### Token Limits: Text vs Multimodal Models

**Critical Difference:** Text and multimodal (Vision/Audio) models use different `max_tokens` strategies.

#### Text Models (MLXRunner)

**Strategy:** Shift-window context management
- Conversation history maintained in context buffer
- Server reserves space for history

**Defaults:**
- **Server:** `context_length / 2` (reserve half for history, half for generation)
- **CLI:** `context_length` (full context, no reservation)

**Example:**
- Llama-3.2-3B (128K context) → Server default: 64K max_tokens

#### Vision/Audio Models (VisionRunner)

**Strategy:** Stateless processing
- Each request is independent (no conversation history in context)
- Metal limitations prevent context preservation

**Defaults:**
- **Server/CLI:** `2048` tokens (conservative, works for all models)

**Rationale:**
- No need for `/2` division (no history to reserve)
- Multimodal inference is slow → 2048 adequate for descriptions/transcriptions
- Prevents accidentally generating 64K+ tokens

**Override:**
```json
{
  "model": "mlx-community/gemma-3n-E2B-it-4bit",
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
- ✅ **Per-chunk streaming:** Real SSE events as each image chunk completes (2.0.4-beta.7+)
- **Multiple images:** Each chunk (1-5 images) streams as it finishes processing
- **Single image:** Behaves like batch mode (one SSE event)
- **Format:** OpenAI-compatible SSE with per-chunk deltas

#### Audio Models
- ⚠️ **Batch mode only:** Single SSE event with complete response
- **Reason:** Single audio per request, no chunking needed
- **Format:** Same as Vision single-image mode

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
data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1702345678,"model":"mlx-community/Llama-3.2-3B-Instruct-4bit","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1702345678,"model":"mlx-community/Llama-3.2-3B-Instruct-4bit","choices":[{"index":0,"delta":{"content":" there"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1702345678,"model":"mlx-community/Llama-3.2-3B-Instruct-4bit","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

**Note:** `stream_options.include_usage` is not supported.

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
- Handles Ctrl-C gracefully (clean shutdown with 5s timeout)
- Runs server in subprocess for improved signal handling
- Logs go to stderr
- `--log-json` produces 100% JSON output
- **Note:** No auto-restart on crashes (use systemd/supervisor for production)

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
- **400 Bad Request:** Invalid input (e.g., too many images, invalid format, validation failures)
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
- **Multiple images:** Processed in chunks (default: 1, max: 5 via `--chunk`)
- **Streaming:** Each chunk delivers results immediately (see Streaming section above)

### Concurrent Requests
- **Current:** Sequential processing (one request at a time)
- **Reason:** Metal backend, single GPU
- **Future:** May add request queuing

---

## Troubleshooting

### Multimodal Request Fails on Python 3.9

**Symptom:** HTTP 501 "Vision/Audio models require Python 3.10+"

**Solution:**
```bash
# Upgrade Python
pyenv install 3.10
pyenv local 3.10
pip install mlx-lm mlx-vlm

# Until mlx-vlm 0.3.10 on PyPI (Vision + Audio support)
pip install mlx-lm "mlx-vlm @ git+https://github.com/Blaizzy/mlx-vlm.git@58122703b0bba7c574d23c9c751f01cf60485d4f"
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
- Image size > 20 MB per image
- More than 5 images per request
- Unsupported format (use JPEG, PNG, GIF, WebP)
- External URLs (not supported, use Base64 data URLs)
- Invalid Base64 encoding

**Solution:** Resize images, reduce count, or check encoding

### Audio Errors

#### Audio Request Fails (HTTP 400)

**Common causes:**
- Audio size > 5 MB
- More than 1 audio per request (multi-audio not supported)
- Unsupported format (use WAV or MP3)
- Invalid Base64 encoding

**Solution:** Compress audio, ensure single audio per request, use supported format

#### Audio Model Not Found

**Symptom:** `Model does not support audio input`

**Cause:** Model lacks audio capability

**Solution:** Use an audio-capable model:
```bash
mlxk list | grep +audio
```

**Note:** Some HuggingFace models may require `mlxk convert --repair-index` before use.

#### Audio Output is Garbled/Multilingual

**Symptom:** Transcription includes unexpected languages (Arabic, Hindi, etc.)

**Cause:** Temperature too high (default text temperature 0.7 causes drift)

**Solution:** Use temperature 0.0 for audio:
```json
{
  "temperature": 0.0
}
```

---

## Limits Summary

| Resource | Limit | Reason |
|----------|-------|--------|
| Images per request | 5 | Metal OOM prevention |
| Image size | 20 MB | Metal OOM prevention |
| Total image size | 50 MB | Metal OOM prevention |
| **Audio per request** | **1** | **mlx-vlm limitation** |
| **Audio size** | **5 MB** | **Token count constraint** |
| Vision model RAM | 70% system | Metal OOM prevention |
| Text model RAM | 70% (warning) | Swap tolerance |
| Vision max_tokens | 2048 (default) | Stateless, slow inference |
| Audio max_tokens | 2048 (default) | Stateless, like Vision |
| Text max_tokens | context_length/2 | Shift-window reservation |

---

## Migration Guide

### From 2.0.4-beta.7 → 2.0.4-beta.8

**New Features:**
- ✅ Audio input support via OpenAI `input_audio` format
- ✅ Supported audio formats: WAV, MP3
- ✅ Audio history filtering: `[n audio(s) were attached]`

**Breaking Changes:**
- None (audio is additive)

**Recommendations:**
- Update mlx-vlm to ≥0.3.10 (GitHub install required, not yet on PyPI)
- Use temperature 0.0 for audio transcription requests
- Test with Gemma-3n or other audio-capable models

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

**Clients MUST send the full message list** with each request:

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

**What "full history" means:**
- ✅ All messages with correct roles (`user`, `assistant`, `system`)
- ✅ Complete assistant responses (including `<!-- mlxk:filenames -->` markers)
- ⚠️ Media payloads (Base64) can be dropped after first Vision request (see [Image ID Persistence](#image-id-persistence-stateless))

**Note:** For Vision models, the server only forwards the last user message to the model (stateless prompt), but still scans the full history for image ID reconstruction.

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

### Image Deduplication

Same image content = same ID (content-hash based).

**Client behavior:**
- Re-uploading the same image → Server assigns same ID
- No client-side deduplication needed

### Image ID Persistence (Stateless)

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

2. **Server Response:** Includes filename mapping table (wrapped in `<details>`)
   ```html
   <details>
   <summary>📸 Image Metadata (1 image)</summary>

   <!-- mlxk:filenames -->
   | Image | Filename | Original | Location | Date | Camera |
   |-------|----------|----------|----------|------|--------|
   | 1 | image_5733332c.jpeg | beach.jpg | 📍 34.0522°N, 118.2437°W | 📅 2024-06-15 | iPhone 14 |

   </details>

   A sandy beach with blue water.
   ```

   **Note:** EXIF columns (Original, Location, Date, Camera) are enabled by default.
   Disable with `MLXK2_EXIF_METADATA=0` for minimal output (Image, Filename only).

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
- ✅ **Zero client changes** - Works with standard OpenAI message format
- ✅ **Storage optimization** - Client can drop large Base64 data (2 MB → 2 KB)
- ✅ **No protocol extensions** - Standard messages[] array, no custom headers
- ✅ **Stateless server** - No server-side session state required
- ✅ **Scales to 100+ images** - Clients only store small text mappings

**Client Recommendations:**
- **After first Vision request:** Drop Base64 image_url from history, keep text + assistant response
- **Store locally:** Small thumbnails for UI (~20 KB/image via IndexedDB)
- **History format:** Text-only user messages + full assistant responses (with mapping tables)
- **⚠️ Preserve verbatim:** Do not sanitize or strip HTML comments from assistant responses — the `<!-- mlxk:filenames -->` markers are required for ID reconstruction

**Example client storage (100 images):**
- ❌ **Before:** 100 images × 2 MB Base64 = 200 MB (exceeds browser limits)
- ✅ **After:** 100 thumbnails × 20 KB + text history = ~2 MB (fits in IndexedDB)

### Audio Messages Format

**Audio content** uses the OpenAI `input_audio` format:

```json
{
  "role": "user",
  "content": [
    {"type": "text", "text": "Transcribe this audio"},
    {
      "type": "input_audio",
      "input_audio": {
        "data": "<base64-encoded>",
        "format": "wav"
      }
    }
  ]
}
```

**Supported formats:** `wav`, `mp3` (or `mpeg` alias)

**Limitations:**
- ❌ Only 1 audio per request (multi-audio causes mlx-vlm token mismatch)
- ❌ Audio + Vision combined: audio is silently ignored

### Cross-Model Workflows (Vision/Audio → Text)

When switching from Vision or Audio to Text model mid-conversation:

1. **Client:** Continue sending full message list (media payloads can be stripped if mapping tables exist)
2. **Server:** Automatically filters any remaining media for text models, replaces with placeholders
3. **Result:** Text model sees `[n image(s) were attached]` or `[n audio(s) were attached]`

**Example workflow:**
```
1. Vision model: User sends 2 images → Model describes both
2. Vision model: User asks "What's different?" → Model compares
3. Switch to Text model: User asks "Which is better for vacation?"
4. Text model: Sees "[2 image(s) were attached]" in history, can reference the conversation
```

**Storage optimization:** After the first Vision request, clients can drop Base64 payloads from history while preserving assistant responses with `<!-- mlxk:filenames -->` markers. The server reconstructs image IDs from these markers.

---

## Changelog

- **2026-01-20:** 2.0.4-beta.8
  - **NEW:** Audio input support via OpenAI `input_audio` format
  - Supported formats: WAV, MP3
  - Audio-capable models: Gemma-3n (others as available)
  - Limits: 5 MB per audio, 1 audio per request
  - Temperature: 0.0 for transcription consistency
  - History filter: `input_audio` → `[n audio(s) were attached]`

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
