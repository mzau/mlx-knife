# MLX Knife Server Handbook

**Version:** 2.0.6 stable (PyPI) · 2.0.7 dev: experimental `/v1/embeddings` backend (`embed-serve`) + `serve --embed-backend` proxy
**Status:** ⚠️ **WORK IN PROGRESS** - This document will evolve until 2.1 stable release
**Last Updated:** 2026-06-17

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

# Embeddings (experimental, 2.0.7) — embed-serve backend + serve gateway = one OpenAI surface
MLXK2_ENABLE_ALPHA_FEATURES=1 mlxk embed-serve bge-small-en-v1.5 --port 8002                 # internal embedding backend
MLXK2_ENABLE_ALPHA_FEATURES=1 mlxk serve --port 8000 --embed-backend http://127.0.0.1:8002   # gateway: /v1/embeddings + /v1/chat/completions both on :8000
```

**Requirements (2.0.7):**
- Python 3.10-3.13 (Text + Vision); 3.10-3.12 only for Audio (miniaudio wheel missing on 3.13/macOS-ARM)
- `mlx-lm==0.31.3` (text backend)
- `mlx-vlm==0.6.2` (vision + multimodal audio; `pip install mlx-knife[vision]`)
- `mlx-audio==0.4.4` (Whisper / Voxtral STT; `pip install mlx-knife[audio]`)
- `transformers==5.5.4` (driven by mlx-audio 0.4.4)
- `torch>=2.0`, `torchvision>=0.15` — temporary base deps for Pixtral / Llama-Vision / Mistral-Small-3.1 (`sunset-by mlx-vlm#1011`, see ADR-023 Workaround-Sunset Policy)

Pins are exact per ADR-023: every upstream minor bump goes through an explicit mlx-knife release with re-verified integration. Do not loosen on `pip install`.

---

## OpenAI API Compatibility

MLX Knife implements a **subset** of the OpenAI API with documented behavioral differences.

### Supported Endpoints

| Endpoint | Status | Notes |
|----------|--------|-------|
| `/v1/chat/completions` | ✅ Supported | Text, Vision (`image_url`), Audio (`input_audio`) |
| `/v1/completions` | ✅ Supported | Legacy text completion |
| `/v1/audio/transcriptions` | ✅ Supported | OpenAI Whisper API (beta.9+) |
| `/v1/embeddings` | ✅ Supported (2.0.7, experimental) | OpenAI Embeddings API. Served by the separate `embed-serve` backend; `serve` proxies it via `--embed-backend`. Returns **501** on a plain `serve` started without `--embed-backend` (embeddings not enabled). See [Embeddings Backend](#embeddings-backend-embed-serve) |
| `/v1/models` | ✅ Supported | HF cache + workspace models (ADR-022); extended with `context_length` field. Does **not** list embedders in 2.0.7 (discovery merge → 2.1) |
| `/health` | ✅ Custom | MLX Knife extension |

### Authentication

MLX Knife **ignores** authentication headers. The server accepts but does not validate:
- `Authorization: Bearer ...`
- Any API key

**Note:** For production deployments requiring authentication, use a reverse proxy (nginx, Caddy).

**⚠️ Client Implementers:** When adding reverse proxy authentication, ensure your client sends authentication headers to **all** endpoints, including:
- `/v1/chat/completions`
- `/v1/completions`
- `/v1/audio/transcriptions` (file upload endpoint)
- `/v1/embeddings` (only active when `--embed-backend` is configured)
- `/v1/models`

A common mistake is implementing auth for JSON endpoints but forgetting `multipart/form-data` endpoints like audio transcription.

### Request Headers

```
Content-Type: application/json  (required)
Authorization: Bearer ...       (optional, ignored)
```

### Response Headers

```
X-Request-ID: <unique-id>       (all responses, MLX Knife extension)
```

**X-Request-ID** (MLX Knife extension):
- Present on **every response** (success and error)
- Same ID appears in error response body as `"request_id"`
- Use for request correlation and distributed tracing (e.g., Broke-Cluster log aggregation)

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

**Error types** (full `ErrorType` enum, `mlxk2/errors.py`):

| Type | HTTP | Meaning |
|------|------|---------|
| `validation_error` | 400 | Invalid request payload (e.g. too many images, malformed audio) |
| `access_denied` | 403 | File / cache permission denied |
| `model_not_found` | 404 | Model spec does not resolve to a cached / workspace model |
| `ambiguous_match` | 400 | Model spec matches multiple cached models — disambiguate |
| `download_failed` | 503 | HF download failed mid-stream |
| `push_operation_failed` | 500 | `/v1/push`-style operation failed (CLI-only path; not user-reachable on the server today) |
| `server_shutdown` | 503 | Lifespan shutdown in progress; new requests are rejected |
| `insufficient_memory` | 507 | Model exceeds the memory threshold (ADR-016) |
| `not_implemented` | 501 | Known capability class, but the feature is not yet implemented (e.g. STT quantization) |
| `unsupported_multimodal` | 501 | Model uses a multimodal class outside the verified-multimodal list (ADR-023) |
| `bad_gateway` | 502 | Embed backend (`serve --embed-backend`) unreachable / connection failed / connect-timeout (retryable; ADR-015 D2) |
| `gateway_timeout` | 504 | Embed backend read-timeout on a slow / large batch (retryable; ADR-015 D2) |
| `internal_error` | 500 | Unexpected backend failure |

(`bad_gateway` / `gateway_timeout` are raised only by the `serve --embed-backend` proxy; a backend's own `4xx`/`5xx` envelopes otherwise pass through verbatim.)

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

**Also honored** (standard OpenAI sampling fields): `top_p` (default `0.9`), `stop` (string or
list of strings), and `repetition_penalty` (default `1.1`, an mlx-knife-leaning default) in
addition to `temperature` and `max_tokens`.

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

### POST /v1/audio/transcriptions

**OpenAI Whisper API compatible audio transcription (beta.9+).**

Use this endpoint for **direct file upload** transcription with STT models (Whisper, Voxtral).

**Request (multipart/form-data):**
```bash
curl -X POST http://localhost:8080/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "model=whisper-large" \
  -F "language=en" \
  -F "response_format=json"
```

**Form Fields:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | File | ✅ | Audio file (WAV, MP3, M4A, FLAC, OGG) |
| `model` | String | ✅ | Model ID (e.g., `whisper-large`, `mlx-community/whisper-large-v3-turbo-4bit`) |
| `language` | String | ❌ | Language code (e.g., `en`, `de`). Auto-detect if omitted. |
| `prompt` | String | ❌ | Optional context to guide transcription |
| `response_format` | String | ❌ | `json` (default), `text`, `verbose_json` |
| `temperature` | Float | ❌ | Sampling temperature (default: 0.0 for greedy) |

**Response (JSON - default):**
```json
{
  "text": "A man said to the universe, Sir, I exist."
}
```

**Response (text):**
```
A man said to the universe, Sir, I exist.
```

**Response (verbose_json):**
```json
{
  "task": "transcribe",
  "language": "en",
  "duration": 0.57,
  "text": "A man said to the universe, Sir, I exist."
}
```

> **Field semantics (differ from OpenAI):** `duration` is the **server-side processing
> wall-time** in seconds, not the audio clip length. `language` echoes the requested `language`
> form field, or the literal `"auto"` when none was supplied — it is never a server-detected
> language code.

**Supported Models:**
- Whisper: `whisper-large`, `mlx-community/whisper-large-v3-turbo-4bit`
- Voxtral: `mlx-community/Voxtral-Mini-3B-2507-bf16` (upstream tokenizer issues)

**Note:** This endpoint requires `mlx-audio` (`pip install mlx-knife[audio]`).

**Translation (2.0.7+):** audio-to-English translation is available via the **CLI**
(`mlxk run --audio FILE --translate`, multilingual non-turbo Whisper only). An OpenAI-compatible
`POST /v1/audio/translations` **server** endpoint is planned but **not yet implemented** in 2.0.7.

**vs. `/v1/chat/completions` with `input_audio`:**

| Feature | `/v1/audio/transcriptions` | `/v1/chat/completions` |
|---------|---------------------------|------------------------|
| Format | Multipart file upload | Base64 in JSON |
| Models | STT only (Whisper, Voxtral) | Multimodal (Gemma-3n) |
| Use case | Pure transcription | Chat with audio context |
| OpenAI API | Whisper API | Chat Completions API |

---

### POST /v1/embeddings

**OpenAI Embeddings API compatible text embeddings (2.0.7, experimental).**

Served by the **`embed-serve`** backend — a separate, single-model process (see
[Embeddings Backend](#embeddings-backend-embed-serve) for the topology and why). A client may
call the backend port directly, or — preferred — go through `serve`'s proxy so one base URL
serves both `/v1/embeddings` and `/v1/chat/completions`.

**Request:**
```bash
# Through the serve gateway (preferred — one base URL for chat + embeddings):
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "bge-small-en-v1.5", "input": "machine learning tutorial", "encoding_format": "float"}'
# Standalone, talking to the embed-serve backend directly, swap in its port: http://localhost:8002
```

**Body Fields:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model` | String | ✅ | Model ID. The backend serves a **single** model, so this is informational (the loaded model answers regardless) and is echoed back as the canonical `org/name` in the response. |
| `input` | String or String[] | ✅ | One text, or a batch (one vector per item, in order). |
| `encoding_format` | String | ❌ | `base64` (**default** — little-endian float32, what the OpenAI SDK decodes) or `float` (raw JSON array, handy for `curl`). |
| `dimensions` | Integer | ❌ | Accepted only if equal to the model's native width; any other value → **400** (no Matryoshka truncation in 2.0.7). |
| `user` | String | ❌ | Accepted and ignored (OpenAI passthrough). |
| `input_type` | String | ❌ | **mlxk extension** (RAG): `document` (default) or `query` (applies the model's query-instruction prefix). Ignored by standard OpenAI clients. |
| `instruct` | String | ❌ | **mlxk extension**: overrides the query task instruction; implies `input_type: query`. |

**Response (`encoding_format: float`):**
```json
{
  "object": "list",
  "data": [
    {"object": "embedding", "index": 0, "embedding": [0.0123, -0.0456, "..."]}
  ],
  "model": "mlx-community/bge-small-en-v1.5",
  "usage": {"prompt_tokens": 4, "total_tokens": 4}
}
```

With `encoding_format: base64` (the default) each `embedding` is a base64 string of
little-endian float32 bytes — the OpenAI Python client decodes it transparently:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="-")   # serve gateway (or :8002 for the backend directly)
q    = client.embeddings.create(model="bge-small-en-v1.5", input="how does X work?").data[0].embedding
docs = client.embeddings.create(model="bge-small-en-v1.5", input=corpus_chunks).data
```

**Notes:**
- Vectors are **L2-normalized**. `usage` token counts are best-effort.
- **Same-model / same-device rule:** a single `embed-serve` backend *is* one model on one device by
  construction (it loads one model at startup; the GPU/`--cpu` choice is fixed for the process), so
  every vector it returns shares one space — that is why the OpenAI response carries no per-vector
  `content_hash`/`device` stamp (unlike the CLI JSONL): it needs none. Send a store's query and
  corpus to the **same backend/gateway**; never mix vectors across backends/models/devices (CPU vs
  GPU diverge ~0.98 cosine on a 4-bit model), which silently breaks cosine/dedup logic.
- **Supported models:** the verified embedders (`docs/MODEL-COVERAGE.md`) — decoder
  (`Qwen3-Embedding-*`, via `mlx-lm`) and encoder (`bge-*`, `*-e5-*`, `mxbai-*`; `model_type: bert`).
  A declared-but-not-vendored embedder (e.g. `xlm-roberta`/`modernbert`) is rejected at backend
  **startup**, never silently.
- **Experimental:** the backend requires `MLXK2_ENABLE_ALPHA_FEATURES=1` (2.0.7).

---

### GET /v1/models

**List available models.**

Returns the runnable models — healthy and runtime-compatible — from both the
HF cache and the workspace home (`MLXK_WORKSPACE_HOME`, ADR-022). This is the
same set of models as the default human `mlxk list` view (without `--all`);
a model preloaded from outside the workspace home is included as well.
The preloaded model (if any) appears exactly once, sorted first; all other
models follow alphabetically.

> **Embedders are excluded in 2.0.7.** Embedding models (e.g. `bge-*`,
> `Qwen3-Embedding-*`) are **not** listed here — they are served by the separate
> `embed-serve` backend, and the discovery merge is deferred to 2.1 (ADR-015). This is the
> one case where `/v1/models` differs from `mlxk list`, which *does* show embedders.

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "Mistral-Small-3.1-24B-Instruct-2503-4bit",
      "object": "model",
      "owned_by": "workspace",
      "permission": [],
      "context_length": 131072
    },
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
- `id`: Model identifier — HuggingFace name for cache models; directory
  basename for workspace models (resolves workspace-first at request time,
  so clients can use it directly as `model` in requests). A model preloaded
  from an explicit path outside the workspace home keeps its absolute path.
- `object`: Always `"model"` (OpenAI-compatible)
- `owned_by`: `"mlx-knife-2.0"` for cached models, `"workspace"` for workspace models
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

```json
{ "status": "healthy", "service": "mlx-knife-server-2.0" }
```

(The `embed-serve` backend has its own `/health` — see [Embeddings Backend](#embeddings-backend-embed-serve) — which returns `{"status": "ok", "model": ...}` and `503` until its model is loaded.)

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

### Audio Support (2.0.4-beta.9)

**Two methods** for audio transcription:

#### Method 1: `/v1/audio/transcriptions` (Whisper API)

**Direct file upload** for STT models (Whisper, Voxtral). Recommended for pure transcription.

```bash
curl -X POST http://localhost:8080/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "model=whisper-large"
```

**Supported:**
- ✅ File upload (multipart/form-data)
- ✅ Formats: WAV, MP3, M4A, FLAC, OGG
- ✅ Response formats: `json`, `text`, `verbose_json`
- ✅ Language detection or explicit `language` parameter

**Models:** Whisper, Voxtral (requires `pip install mlx-knife[audio]`)

#### Method 2: `/v1/chat/completions` with `input_audio`

**Base64-encoded audio** in chat messages for multimodal models (Gemma-3n).

```json
{
  "model": "gemma-3n-E2B-it-4bit",
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "Transcribe this audio"},
      {"type": "input_audio", "input_audio": {"data": "<base64>", "format": "wav"}}
    ]
  }]
}
```

**Supported:**
- ✅ OpenAI `input_audio` format (Base64-encoded)
- ✅ Formats: WAV, MP3
- ✅ Temperature 0.0 (greedy sampling for transcription consistency)

**Limits (both methods):**
- **Per-audio:** 50 MB max (`MAX_AUDIO_SIZE_BYTES` in `mlxk2/tools/vision_adapter.py`; same limit on both endpoints)
- **Count:** 1 audio per request

> **Caveat — multimodal chat audio.** STT-dedicated models (Whisper, Voxtral) have natural stop tokens and process long audio reliably; the `/v1/audio/transcriptions` endpoint is robust against runaway inference. Multimodal chat audio (Gemma-3n in `/v1/chat/completions` with `input_audio`) lacks robust EOS-detection and can hallucinate without converging — `max_tokens` (default 2048) is currently the only inference bound. Keep chat audio short (a few seconds) for now; model-specific bounds are an open engineering item.

**Models:** Gemma-3n (Vision + Audio + Text)

**Important Characteristics:**

- **Stateless Server:** Same as Vision — no server-side state
- **Single Audio:** Only one audio file per request
- **Audio+Vision:** When both present in chat, audio is silently ignored (mlx-vlm behavior)
- **Temperature:** Fixed at 0.0 for transcription consistency

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

### Embeddings Backend (embed-serve)

**Experimental (2.0.7).** Text embeddings run in a **separate process**, `mlxk embed-serve` —
not inside `mlxk serve`. This keeps the main server's memory gates (8 GB vision / 4 GB audio)
intact: an embedding model is never loaded into serve's address space. The backend exposes two
routes: `POST /v1/embeddings` (the OpenAI surface) and `GET /health` (liveness — `200` once the
model is loaded, `503` before).

**Topology — one OpenAI surface:**
```bash
# Embedding backend — separate process, owns the model, localhost-internal
MLXK2_ENABLE_ALPHA_FEATURES=1 mlxk embed-serve bge-small-en-v1.5 --port 8002

# Main server — proxies /v1/embeddings to the backend; clients use ONE base URL
MLXK2_ENABLE_ALPHA_FEATURES=1 mlxk serve --model chat-model --embed-backend http://127.0.0.1:8002
```
A RAG client points at `serve` (or, in a cluster, broke's gateway) for both `/v1/embeddings`
and `/v1/chat/completions` — it never talks to `embed-serve` directly. In standalone use you may
also call the backend port directly.

> **2.0.7 rollout:** both the `embed-serve` backend (Slice D1) and the `serve --embed-backend`
> proxy (Slice D2) ship in 2.0.7 (experimental, alpha-gated). `GET /v1/models` on `serve` does
> **not** advertise the backend's embedders in 2.0.7 (discovery merge deferred to 2.1) — embeddings
> still work; only "list → embed" auto-discovery is incomplete.

**Proxy behavior (`serve --embed-backend`):** serve forwards the request body to the backend
**byte-for-byte** and returns the backend's response verbatim — the embed model is never loaded
into serve's process. The backend is **not** probed at startup, so it may be started after `serve`;
connection problems surface per request, not at boot.

| Condition | serve returns |
|-----------|---------------|
| No `--embed-backend` configured | `501` (`not_implemented`) |
| Backend unreachable / refused / connect-timeout | `502` (`bad_gateway`, retryable) |
| Backend read-timeout (slow / large batch) | `504` (`gateway_timeout`, retryable) |
| Backend returns `4xx`/`5xx` | passed through **verbatim** (status + body) |

Timeouts: connect 3 s (fail fast when the backend is down), read 120 s (large batches).

**Flags:** `mlxk embed-serve <model> [--port 8002] [--host 127.0.0.1] [--cpu] [--log-level info] [--log-json] [--json] [--verbose]`
(`--json` prints startup info as JSON; `--verbose` shows detailed output.)

**Device:** GPU by default. When co-resident with a GPU-bound `serve`, run the backend with
`--cpu` — embeddings are 5–50 ms, and CPU keeps the single Metal GPU free for latency-critical
chat (on unified memory this trades GPU contention, not RAM).

**Memory:** an embedding model is small (~300 MB–1 GB) and visible in Activity Monitor. On
RAM-constrained machines, simply don't start `embed-serve` (explicit choice, not implicit
degradation).

**Logging:** `--log-json` produces JSON logs on the backend's own stderr (same schema as
`serve`); each process logs independently.

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

# Feature gates
MLXK2_ENABLE_PIPES=1              # Unix pipe integration (beta, 2.0.4-beta.1)
MLXK2_ENABLE_ALPHA_FEATURES=1     # Alpha (2.0.7): embed, embed-serve, serve --embed-backend

# Embeddings proxy (ADR-015 D2) — normally set for you by `serve --embed-backend URL`,
# but can be set directly. When unset, POST /v1/embeddings on serve returns 501.
MLXK2_EMBED_BACKEND=http://127.0.0.1:8002
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

### Client Errors (4xx)
- **400 Bad Request:** Invalid input (e.g., too many images, invalid format, validation failures, ambiguous model spec; for `/v1/embeddings`: empty or non-string `input` (incl. empty array items), unsupported `encoding_format` or `input_type`, or a non-native `dimensions` value)
- **403 Forbidden:** File or cache permission denied (`access_denied`)
- **404 Not Found:** Model not found in cache or workspace

### Server Errors (5xx)
- **500 Internal Server Error:** Unexpected backend failure
- **501 Not Implemented:** Feature not supported. Sub-causes:
  - `not_implemented` — known capability class, feature missing (e.g. STT quantization); also returned by `POST /v1/embeddings` when `serve` has no `--embed-backend` configured (ADR-015 D2)
  - `unsupported_multimodal` — model uses a multimodal class outside the verified-multimodal list (ADR-023)
- **502 Bad Gateway:** Embed backend unreachable / connection refused / connect-timeout (`bad_gateway`, **retryable**; `serve --embed-backend` proxy, ADR-015 D2)
- **503 Service Unavailable:** Server shutting down (`server_shutdown`) or HF download failed (`download_failed`)
- **504 Gateway Timeout:** Embed backend read-timeout on a slow / large batch (`gateway_timeout`, **retryable**; `serve --embed-backend` proxy, ADR-015 D2)
- **507 Insufficient Storage:** Memory constraints violated (vision/audio model >70% RAM, ADR-016)

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

### Vision/Audio Require Python 3.10+

mlx-knife itself requires Python 3.10+ (`requires-python >=3.10`), so a normal `pip install`
cannot land on 3.9. This 501 only appears when running from a source checkout on an unsupported
interpreter.

**Symptom:** HTTP 501 "Vision/Audio models require Python 3.10+"

**Solution:**
```bash
# Upgrade Python (3.10-3.12 required)
pyenv install 3.10
pyenv local 3.10

# Install with Vision support
pip install mlx-knife[vision]

# Install with Audio STT support (Whisper)
pip install mlx-knife[audio]

# Install with everything
pip install mlx-knife[all]
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
- Audio size > 50 MB (hard limit, both endpoints — `MAX_AUDIO_SIZE_BYTES`)
- More than 1 audio per request (multi-audio not supported)
- Unsupported format (use WAV or MP3 for chat `input_audio`; WAV/MP3/M4A/FLAC/OGG for `/v1/audio/transcriptions`)
- Invalid Base64 encoding (chat endpoint only)

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

#### Transcription Endpoint Returns Wrong Model Error

**Symptom:** `Model 'xxx' is not an audio transcription model`

**Cause:** `/v1/audio/transcriptions` only works with STT models (Whisper, Voxtral)

**Solution:** Use the correct model type:
```bash
# For transcription endpoint: STT models
curl -X POST http://localhost:8080/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "model=whisper-large"

# For multimodal chat: Gemma-3n (use chat/completions instead)
# See "Audio Messages Format" in Appendix
```

#### mlx-audio Not Installed

**Symptom:** `STT models require mlx-audio`

**Solution:**
```bash
pip install mlx-knife[audio]
```

### Embeddings Errors (experimental, 2.0.7)

**Symptom:** `embed` / `embed-serve` / `serve --embed-backend` rejected with
"requires MLXK2_ENABLE_ALPHA_FEATURES=1".
**Cause/Fix:** the embeddings surface is alpha-gated — export `MLXK2_ENABLE_ALPHA_FEATURES=1`
before starting `embed-serve` *and* `serve --embed-backend`.

**Symptom:** `POST /v1/embeddings` on `serve` returns **501** (`not_implemented`,
"Embeddings are not enabled on this server").
**Cause/Fix:** `serve` was started without `--embed-backend`. Start `mlxk embed-serve <model>`
and pass `--embed-backend http://<host>:<port>` to `serve`.

**Symptom:** `POST /v1/embeddings` returns **502** (`bad_gateway`, retryable).
**Cause/Fix:** the embed-serve backend is unreachable / refused / didn't accept the connection
within 3 s. Verify the backend process is up and the `--embed-backend` URL/port is correct.
`serve` does not probe the backend at startup, so this only surfaces per request.

**Symptom:** `POST /v1/embeddings` returns **504** (`gateway_timeout`, retryable).
**Cause/Fix:** the backend didn't respond within the 120 s read budget — usually a very large
batch. Reduce the batch size or retry.

---

## Limits Summary

| Resource | Limit | Reason |
|----------|-------|--------|
| Images per request | 5 | Metal OOM prevention |
| Image size | 20 MB | Metal OOM prevention |
| Total image size | 50 MB | Metal OOM prevention |
| **Audio per request (chat)** | **1** | **mlx-vlm limitation** |
| **Audio size (both endpoints)** | **50 MB** | **`MAX_AUDIO_SIZE_BYTES` — measured in raw bytes, codec-agnostic. WAV @ 16 kHz mono caps at ~26 min; compressed formats fit much more (verified: 55 min MP3 transcription via Whisper stays under the limit). Whisper handles long audio robustly; multimodal chat audio is bounded by `max_tokens` only (see Audio Support caveat).** |
| Vision model RAM | 70% system | Metal OOM prevention |
| Text model RAM | 70% (warning) | Swap tolerance |
| Vision max_tokens | 2048 (default) | Stateless, slow inference |
| Audio max_tokens | 2048 (default) | Stateless, like Vision |
| Text max_tokens | context_length/2 | Shift-window reservation |

---

## Migration Guide

### From 2.0.3 → 2.0.4

**New Features:**

| Feature | Endpoint | Requirements |
|---------|----------|--------------|
| Vision (images) | `/v1/chat/completions` | `pip install mlx-knife[vision]` |
| Audio Chat (Gemma-3n) | `/v1/chat/completions` | `pip install mlx-knife[vision]` |
| Audio STT (Whisper) | `/v1/audio/transcriptions` | `pip install mlx-knife[audio]` |
| Memory pre-load checks | All endpoints | Built-in (HTTP 507) |
| Server audio preload | `mlxk serve --model whisper-large` | Built-in |

**Breaking Changes:**

| Change | Before | After | Impact |
|--------|--------|-------|--------|
| Python version | 3.9+ | 3.10-3.12 | Upgrade required |
| Vision `max_tokens` default | 1024 | 2048 | Longer responses |
| Memory checks (Vision) | None | 70% RAM limit | HTTP 507 possible |

**New Dependencies (auto-installed):**
- `mlx-vlm>=0.3.10` (Vision + Gemma-3n audio)
- `mlx-audio>=0.3.1` (Whisper STT)
- `python-multipart>=0.0.9` (file uploads)

**Client Updates Required:**
- Handle HTTP 507 (Insufficient Storage) for large Vision models
- Update clients expecting `max_tokens: 1024` to handle 2048
- Use `temperature: 0.0` for audio transcription consistency

**Recommendations:**
- Pure transcription: Use `/v1/audio/transcriptions` with Whisper
- Multimodal chat: Use `/v1/chat/completions` with `input_audio`
- Test Vision/Audio workflows on Python 3.10+

### From 2.0.4 → 2.0.5

**Endpoint surface:** unchanged.

**Dependency bumps (auto-installed):**

| Package | 2.0.4 | 2.0.5 |
|---------|-------|-------|
| `mlx-lm` | `>=0.30.5` | `>=0.31.1,<0.32` |
| `mlx-vlm` | `==0.3.10` | `>=0.3.10,<0.4` |
| `mlx-audio` | `==0.3.1` | `>=0.4.1,<0.5` |
| `transformers` | (transitive only) | `==5.0.0` (now explicit) |

**Behavior changes:**

| Change | Effect on operators |
|--------|---------------------|
| ADR-023 Text-First + Verified Multimodal | Multimodal models outside the verified list now reject with HTTP 501 `unsupported_multimodal`. Previously ran with risk of silent fallback. |
| Workspace model spec (CLI) | `--model <path-to-workspace-dir>` accepted by `mlxk serve --model` (path-based; transparent to API consumers). |

**Client-visible:**
- Models that previously partially-worked may now return 501. Clients should surface the new `unsupported_multimodal` type from the error envelope.

### From 2.0.5 → 2.0.6

**Endpoint surface:** unchanged.

**Dependency bumps (auto-installed):**

| Package | 2.0.5 | 2.0.6 |
|---------|-------|-------|
| `mlx-lm` | `>=0.31.1,<0.32` | `==0.31.3` |
| `mlx-vlm` | `>=0.3.10,<0.4` | `==0.4.4` |
| `mlx-audio` | `>=0.4.1,<0.5` | `==0.4.3` |
| `transformers` | `==5.0.0` | `==5.5.4` |
| `torch` | (optional via mlx-vlm) | `>=2.0` (**new base dep**) |
| `torchvision` | (optional via mlx-vlm) | `>=0.15` (**new base dep**) |

**Why `torch` + `torchvision` as base deps:** `transformers >=5.5` made the torchvision-backed Fast image processor the default for Pixtral / Llama-Vision / Mistral-Small-3.1. Without these, `mlx-vlm`'s `AutoProcessor.from_pretrained(..., use_fast=True)` fails with `requires_backends ImportError`. Marked `sunset-by mlx-vlm#1011` (ADR-023 Workaround-Sunset Policy) — drops on `mlx-vlm` providing a `use_fast=False` fallback.

**Install size impact:** `torch>=2.0` + `torchvision>=0.15` add ~1 GB to the base install. Operators on size-constrained images (containers, embedded systems) should plan for this.

**Behavior changes:**

| Change | Effect on operators |
|--------|---------------------|
| `gemma4` vision convert | New `mlxk convert --quantize` target (CLI; no server-side change). |
| Capability label fixes | `/v1/models` listing is more accurate for STT-only and Gemma 4 models — no API contract change, may shift which models clients see. |

**Client updates required:** none.

---

### From 2.0.6 → 2.0.7

**Endpoint surface:** adds `POST /v1/embeddings` (experimental, gated by
`MLXK2_ENABLE_ALPHA_FEATURES=1`). It is served by the separate `mlxk embed-serve` backend and
exposed on `serve` only when started with `--embed-backend URL` (otherwise `POST /v1/embeddings`
returns **501**). The embed model is never loaded into serve's process. `GET /v1/models` does
**not** advertise embedders in 2.0.7 (discovery merge deferred to 2.1).

**New error codes:** the proxy adds **502** `bad_gateway` (backend unreachable) and **504**
`gateway_timeout` (backend read-timeout) — both **retryable** (ADR-015 D2). Backend `4xx/5xx`
envelopes pass through verbatim.

**Dependency bumps (auto-installed):**

| Package | 2.0.6 | 2.0.7 |
|---------|-------|-------|
| `mlx-vlm` | `==0.4.4` | `==0.6.2` |
| `mlx-audio` | `==0.4.3` | `==0.4.4` |

(`mlx-lm==0.31.3`, `transformers==5.5.4`, `torch`/`torchvision` base deps unchanged.)

**Behavior changes:**

| Change | Effect on operators |
|--------|---------------------|
| Audio translation | New CLI flag `mlxk run --audio FILE --translate` (Whisper, multilingual non-turbo). The server endpoint `POST /v1/audio/translations` is **not yet implemented** in 2.0.7. |

**Client updates required:** none for existing chat/audio/models clients. A RAG client can now
point at one base URL (serve's port) for both `/v1/chat/completions` and `/v1/embeddings` once
`serve --embed-backend` is configured.

---

## References

- **Architecture Principles:** `docs/ARCHITECTURE.md`
- **Testing Details:** `docs/TESTING-DETAILS.md`
- **Verified Multimodal Coverage:** `docs/MODEL-COVERAGE.md` (per-release operation × model_type matrix)

### ADRs (development decisions)

- **ADR-004:** Enhanced Error Logging (the error-type taxonomy that `bad_gateway`/`gateway_timeout` extend)
- **ADR-012:** Vision Support
- **ADR-015:** Embeddings API (`/v1/embeddings` backend, `embed-serve`, `serve --embed-backend` proxy; 502/504 gateway codes)
- **ADR-016:** Memory-Aware Loading (HTTP 507 rationale)
- **ADR-020:** Audio Backend Architecture (STT routing, MLX_AUDIO vs MLX_VLM)
- **ADR-022:** Workspace-First Paradigm (background; surface-transparent on the server)
- **ADR-023:** Text-First + Verified Multimodal (HTTP 501 `unsupported_multimodal` policy + Workaround-Sunset Policy for `torch` / `torchvision`)
- **ADR-024:** Pre-Execution Capability-Mismatch Reject (Class A — CLI-side; surface-transparent on the server today)
- **ADR-025:** content_hash v2 (background; surface-transparent on the server)

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

### Audio Transcriptions (File Upload)

For direct STT transcription with dedicated models (Whisper, Voxtral), use the `/v1/audio/transcriptions` endpoint:

**Request (multipart/form-data):**
```bash
curl -X POST http://localhost:8080/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "model=whisper-large" \
  -F "language=en" \
  -F "response_format=json"
```

**Form Fields:**

| Field | Required | Description |
|-------|----------|-------------|
| `file` | ✅ | Audio file (WAV, MP3, M4A, FLAC, OGG) |
| `model` | ✅ | Model ID (e.g., `whisper-large`, full HF path) |
| `language` | ❌ | Language code (`en`, `de`, etc.). Auto-detect if omitted. |
| `response_format` | ❌ | `json` (default), `text`, `verbose_json` |
| `temperature` | ❌ | Sampling temperature (default: 0.0) |

**Response Formats:**

```json
// json (default)
{"text": "Hello world."}

// verbose_json
{"task": "transcribe", "language": "en", "duration": 2.5, "text": "Hello world."}

// text
Hello world.
```

**When to use which endpoint:**

| Use Case | Endpoint | Model Type | Format |
|----------|----------|------------|--------|
| Pure transcription | `/v1/audio/transcriptions` | STT (Whisper, Voxtral) | File upload |
| Chat with audio context | `/v1/chat/completions` | Multimodal (Gemma-3n) | Base64 JSON |
| Long audio (>30s) | `/v1/audio/transcriptions` | STT (Whisper) | File upload |

**Client Implementation Notes:**
- Use `multipart/form-data` content type (not `application/json`)
- File field name must be `file`
- Maximum file size: 50 MB (~15 min @ 16kHz mono)
- Requires `mlx-audio` on server (`pip install mlx-knife[audio]`)

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

- **2026-06-17:** 2.0.7 (in progress) — embeddings + audio translation
  - **NEW:** `/v1/embeddings` (OpenAI Embeddings API), served by the new `mlxk embed-serve`
    backend (separate single-model process; ADR-015). Experimental — requires
    `MLXK2_ENABLE_ALPHA_FEATURES=1`.
  - `encoding_format`: `base64` (default, SDK-compatible) and `float`; batch `input`; L2-normalized
    vectors; mlxk extensions `input_type` / `instruct` for RAG query embedding.
  - Topology: `serve --embed-backend URL` proxies `/v1/embeddings` to the backend (Slice D2, same
    release). Proxy errors: **501** (no `--embed-backend` configured), **502** `bad_gateway`
    (backend unreachable), **504** `gateway_timeout` (read-timeout) — 502/504 retryable; backend
    `4xx/5xx` pass through verbatim. `GET /v1/models` does not yet advertise embedders (merge → 2.1).
  - **NEW:** `mlxk run --audio FILE --translate` — Whisper audio-to-English translation (CLI;
    multilingual non-turbo). Server endpoint `POST /v1/audio/translations` still pending.
  - Dep-wave: `mlx-vlm 0.4.4 → 0.6.2`, `mlx-audio 0.4.3 → 0.4.4` (`mlx-lm`/`transformers` unchanged).
  - Endpoint surface otherwise unchanged.

- **2026-05-12:** 2.0.6 stable (handbook sync)
  - Endpoint surface unchanged.
  - Dep-wave: `mlx-lm==0.31.3`, `mlx-vlm==0.4.4`, `mlx-audio==0.4.3`, `transformers==5.5.4`.
  - **NEW base deps** `torch>=2.0`, `torchvision>=0.15` (Pixtral / Llama-Vision / Mistral-Small-3.1; `sunset-by mlx-vlm#1011`, ADR-023 Workaround-Sunset Policy). Adds ~1 GB to base install.
  - `/v1/models` listing accuracy improved for STT-only and Gemma 4 (capability label fixes; no schema change).
  - Documentation: full Error-Type table; HTTP 501 split into `not_implemented` vs `unsupported_multimodal` (ADR-023); migration blocks 2.0.4 → 2.0.5 → 2.0.6; audio-size limit corrected to 50 MB unified (both endpoints).

- **2026-04-18:** 2.0.5 stable
  - Endpoint surface unchanged.
  - ADR-023 enforced: multimodal models outside the verified list reject with HTTP 501 `unsupported_multimodal` (was: silent fallback risk).
  - Dep-wave: `mlx-lm==0.31.1`, `mlx-audio==0.4.1`, `transformers>=5.0.0,<5.5.0`.
  - CLI: workspace clone (`mlxk clone`) introduced; workspace paths accepted by `mlxk serve --model <path>` (transparent to API consumers).

- **2026-01-31:** 2.0.4-beta.9
  - **NEW:** `/v1/audio/transcriptions` endpoint (OpenAI Whisper API compatible)
  - Direct file upload for STT models (Whisper, Voxtral)
  - Server preload support for audio models
  - Response formats: `json`, `text`, `verbose_json`
  - Supported audio formats: WAV, MP3, M4A, FLAC, OGG

- **2026-01-20:** 2.0.4-beta.8
  - **NEW:** Audio input support via OpenAI `input_audio` format (chat completions)
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
