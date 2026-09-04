# MLX Knife Server Handbook

**Version:** 2.0.7 plus the unreleased tree state — the 2.0.8 dependency wave, the `serve`
signal/teardown fix, and the generation-budget rule (default `max_tokens`, `finish_reason:
"length"`, HTTP 400 `context_length_exceeded`). Endpoint surface and request/response shapes are
those of released 2.0.7; the [Migration Guide](#migration-guide) records what differs.
**Scope:** what the server does today. Planned work, deferred features and target releases are
deliberately absent — this is a contract, not a roadmap.
**Last Updated:** 2026-09-02

> **Audience:** Server operators, DevOps, API consumers
> **For implementation details:** See `ARCHITECTURE.md` and `docs/ADR/` (developer documentation)

> **Which server does this describe?** Write your client against *this* document rather than a
> release-pinned copy of it. **The server does not report its own version** — `GET /health`
> returns the family string `mlx-knife-server-2.0` — and there is no capability negotiation, so
> a client cannot select a version-specific contract at runtime even if one existed.
>
> The surface described here is current as of **2.0.7** plus the unreleased 2.0.8 tree state.
> Older 2.0.x releases predate parts of it — the Changelog at the end records when each endpoint appeared. Where behaviour genuinely
> varies it is anchored inline rather than left to the reader, including the one case a client
> cannot probe: container audio formats depend on tooling installed on the server host (see
> [Audio Errors](#audio-errors)).

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

**Requirements (2.0.8 pin set):**
- Python 3.10–3.12. macOS/ARM has no 3.13 wheel for `miniaudio`, and `mlx-audio` is a **base** dependency — there is no audio-free install variant, so 3.13 fails at install time.
- `mlx>=0.30.0,<0.32.1`
- `mlx-lm==0.31.3` (text backend)
- `mlx-vlm==0.6.10` (vision + multimodal audio)
- `mlx-audio==0.4.8` (Whisper / Voxtral STT)
- `transformers==5.14.1` (required by `mlx-vlm >=0.6.5`)
- **no `torch` / `torchvision`** — the verified vision set loads torch-free from `mlx-vlm 0.6.4` onwards (mlx-vlm #1011)

Pins are exact per ADR-023: every upstream minor bump goes through an explicit mlx-knife release with re-verified integration. Do not loosen on `pip install`.

> **If you are on released 2.0.7 (PyPI):** you have the previous pin set — `mlx-vlm==0.6.2`, `transformers==5.5.4`, plus `torch`/`torchvision` as base deps. Endpoints and request/response shapes are identical; what differs is the pin set, how `serve` shuts down, and the generation budget — the default `max_tokens`, `finish_reason: "length"` on a cut answer, and a 400 for a prompt that fills the context window. See *From 2.0.7 → 2.0.8* in the [Migration Guide](#migration-guide).

---

## OpenAI API Compatibility

MLX Knife implements a **subset** of the OpenAI API with documented behavioral differences.

### Supported Endpoints

| Endpoint | Status | Notes |
|----------|--------|-------|
| `/v1/chat/completions` | ✅ Supported | Text, Vision (`image_url`), Audio (`input_audio`) |
| `/v1/completions` | ✅ Supported | Legacy text completion |
| `/v1/audio/transcriptions` | ✅ Supported | OpenAI Whisper API (beta.9+) |
| `/v1/audio/translations` | ✅ Supported (2.0.7+) | OpenAI Whisper translations API — speech→English (multilingual non-turbo Whisper; non-capable models → 400/422). See [POST /v1/audio/translations](#post-v1audiotranslations) |
| `/v1/embeddings` | ✅ Supported (2.0.7, experimental) | OpenAI Embeddings API. Served by the separate `embed-serve` backend; `serve` proxies it via `--embed-backend`. Returns **501** on a plain `serve` started without `--embed-backend` (embeddings not enabled). See [Embeddings Backend](#embeddings-backend-embed-serve) |
| `/v1/models` | ✅ Supported | HF cache + workspace models (ADR-022); extended with `context_length` field. Does **not** list embedders — they belong to the separate `embed-serve` backend, whose model list is not merged in |
| `/health` | ✅ Custom | MLX Knife extension — liveness probe, no backend state |

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

**Browser clients:** if a fronting proxy enforces auth, send `Authorization: Bearer <key>` — it passes the CORS preflight (see [CORS](#cors-browser-clients)). `serve` / `embed-serve` themselves accept and ignore it.

### CORS (Browser Clients)

Both `serve` and the `embed-serve` backend send permissive CORS headers, so the
OpenAI surface (`/v1/chat/completions`, `/v1/embeddings`, `/v1/models`,
`/v1/audio/*`) is callable **directly from a browser**. Behavior verified against
`serve` and `embed-serve` (mlx-knife 2.0.7):

Preflight (`OPTIONS`) → `200`:
```
Access-Control-Allow-Origin:      <the request's Origin, reflected>
Access-Control-Allow-Methods:     DELETE, GET, HEAD, OPTIONS, PATCH, POST, PUT
Access-Control-Allow-Headers:     <reflects Access-Control-Request-Headers>
Access-Control-Allow-Credentials: true
Access-Control-Max-Age:           600
Vary:                             Origin
```
Actual responses carry `Access-Control-Allow-Origin: <reflected Origin>`,
`Access-Control-Allow-Credentials: true`, `Vary: Origin`.

Notes for client implementers:
- **The Origin is reflected, not `*`.** The server is configured to allow all
  origins, but because credentials are enabled it echoes the caller's `Origin`
  (and sets `Vary: Origin`) instead of a literal `*`. Effectively any origin is
  accepted — **including `null`** (pages opened via `file://`) — and you may use
  `credentials: 'include'`.
- **`Content-Type` and `Authorization` pass preflight** (request headers are
  reflected), so a proxy-enforced bearer token works at the transport layer (mlxk
  itself ignores it — see [Authentication](#authentication)).
- **Through the proxy**, a browser talks only to `serve`'s port; the
  `serve → embed-serve` hop is server-to-server (no browser CORS). Talking
  directly to the `embed-serve` port uses its own, identical policy.
- This is a wide-open, **local / trusted-network** posture — origins are not
  restricted and not configurable. For exposure to untrusted origins, front the
  server with a reverse proxy.

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
| `max_completion_tokens` | Preferred | Silently ignored — the request falls to `max_tokens`, else the default ceiling | Unknown request fields are dropped, not rejected |
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

**Error types** (the taxonomy is shared with the CLI; rows marked *CLI only* have no server path and never appear in a response):

| Type | HTTP | Meaning |
|------|------|---------|
| `validation_error` | 400 | Invalid request payload (e.g. an image over 20 MB or more than 50 MB of images in one request, malformed audio, `max_tokens` below 1) |
| `context_length_exceeded` | 400 | The prompt fills the model's context window; nothing is left to generate. The message names prompt tokens and window, `detail` carries both as `{"prompt_tokens", "context_length"}`; never retryable |
| `access_denied` | 403 | File / cache permission denied — *CLI only*; no server path raises 403 |
| `model_not_found` | 404 | Model spec does not resolve to a cached / workspace model |
| `not_found` | 404 | No endpoint matches the request path — most often a base URL that already ends in `/v1` |
| `method_not_allowed` | 405 | The endpoint exists, but not for this method; the response carries an `Allow` header |
| `ambiguous_match` | 400 | Model spec matches multiple cached models — *CLI only*; the server answers such a spec with 404 `model_not_found` |
| `payload_too_large` | 413 | Audio upload above the 50 MB limit, on either audio endpoint |
| `capability_not_supported` | 422 | The request is well-formed and the feature exists, but *this* model cannot serve it — a request carrying images or audio while the loaded model is text-only, or `/v1/audio/translations` against a model that cannot translate |
| `download_failed` | 503 | HF download failed mid-stream — *CLI only*; the server never downloads |
| `push_operation_failed` | 500 | `mlxk push` failed — *CLI only* |
| `server_shutdown` | 503 | Lifespan shutdown in progress; new requests are rejected |
| `insufficient_memory` | 507 | Model exceeds the memory threshold (ADR-016) |
| `not_implemented` | 501 | The server cannot run this: a missing dependency, Python below 3.10 for a vision model, an audio model of unknown backend, a checkpoint the runtime reports incompatible, or `/v1/embeddings` without `--embed-backend` |
| `unsupported_multimodal` | 501 | Model uses a multimodal class outside the verified-multimodal list (ADR-023) — *CLI only* (`convert --quantize`); a server response never carries it |
| `bad_gateway` | 502 | Embed backend (`serve --embed-backend`) unreachable / connection failed / connect-timeout (retryable; ADR-015) |
| `gateway_timeout` | 504 | Embed backend read-timeout on a slow / large batch (retryable; ADR-015) |
| `internal_error` | 500 | Unexpected backend failure |

(`bad_gateway` / `gateway_timeout` are raised only by the `serve --embed-backend` proxy; a backend's own `4xx`/`5xx` envelopes otherwise pass through verbatim.)

**Status and type always agree.** A fixed mapping binds the two, so routing on either one gives the
same answer; the type is the more specific of the two where a status carries two meanings — 400
(`validation_error`, `context_length_exceeded`) and 404 (`model_not_found`, `not_found`). Every other
status maps to exactly one type.

Three types describe a request the server declines rather than fails, and they say different things.
`not_implemented` — the feature does not exist here. `unsupported_multimodal` — the model's class is
outside the verified-multimodal list. `capability_not_supported` — the feature exists and the request
is fine, but the model you named cannot serve it; `POST /v1/audio/translations` against a
whisper-turbo or `.en` variant is the case you will meet (see
[the reject matrix](#post-v1audiotranslations)). All three are deliberate rejects, and none of them is
`retryable`.

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

**Also honored** (standard OpenAI sampling fields): `top_p` (default `0.9`) and
`repetition_penalty` (default `1.1`, an mlx-knife-leaning default), in addition to `temperature`
and `max_tokens`. `temperature` defaults to **0.7**, and to **0.0** against an audio model on any
surface — transcription is not a creative task. An explicit value always wins, except on the vision
paths, where `temperature` is fixed at 0.0 (greedy decoding, to keep descriptions from drifting) and
the value sent is ignored; `top_p` and `repetition_penalty` do apply there.

`stop` (string or list of strings) ends the answer at the first sequence that matches; the
sequence itself is removed. The match is against the model's text: the image-metadata header
that precedes a vision answer is not searched. What that costs differs by surface:

- **Batch:** the sequences are applied to the finished text, so the answer ends where OpenAI says
  it ends and `finish_reason` is `"stop"` — but the tokens generated past the cut were generated,
  and still count in `usage`. A chunked vision stream is a batch answer per chunk: the chunk's
  text is cut the same way, and no later chunk is generated.
- **Stream:** each token is checked as it is emitted, and the terminal chunk reports `"stop"`. The
  check is per token, so a sequence split across two of them is not seen, and the token carrying a
  match has already been sent — the answer ends one token late rather than exactly at the sequence.
- **Dedicated STT (Whisper, Voxtral) through chat completions:** the transcript is returned whole;
  `stop` is not applied on that path.

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

`finish_reason` is `"stop"` when the model ended its turn, `"length"` when the generation budget cut
the answer, and `null` when neither is known — see [Token Limits](#token-limits-text-vs-multimodal-models)
for every case.

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
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "model=whisper-large" \
  -F "language=en" \
  -F "response_format=json"
```

**Form Fields:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | File | ✅ | Audio file. **WAV, MP3 and FLAC are always accepted.** M4A/AAC, OGG/Opus and WebM additionally require `ffmpeg` and `ffprobe` on the server host — no endpoint exposes whether they are present, so treat those formats as best-effort and handle the documented failure (see [Audio Errors](#audio-errors)) |
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

**Note:** This endpoint needs `mlx-audio` — included in the base install (Python 3.10–3.12).

**Translation:** for audio-to-English translation, use the dedicated
[`POST /v1/audio/translations`](#post-v1audiotranslations) endpoint (or the CLI
`mlxk run --audio FILE --translate`). Both require a multilingual non-turbo Whisper model.

**vs. `/v1/chat/completions` with `input_audio`:**

| Feature | `/v1/audio/transcriptions` | `/v1/chat/completions` |
|---------|---------------------------|------------------------|
| Format | Multipart file upload | Base64 in JSON |
| Models | STT only (Whisper, Voxtral) | Multimodal (Gemma-3n) |
| Use case | Pure transcription | Chat with audio context |
| OpenAI API | Whisper API | Chat Completions API |

---

### POST /v1/audio/translations

**OpenAI Whisper API compatible speech-to-English translation (2.0.7+, Issue #54).**

Translate non-English speech directly to **English** text. This mirrors
`/v1/audio/transcriptions` but hardcodes Whisper's `translate` task, so existing
OpenAI SDKs work unchanged (`client.audio.translations.create(...)`). Output is
**always English** — a Whisper architectural constraint, there is no free target
language. Only **multilingual, non-turbo** Whisper variants support it.

**Request (multipart/form-data):**
```bash
curl -X POST http://localhost:8000/v1/audio/translations \
  -F "file=@german-news.mp3" \
  -F "model=mlx-community/whisper-large-v3-4bit"
```

```python
# OpenAI SDK (drop-in)
client.audio.translations.create(
    model="mlx-community/whisper-large-v3-4bit",
    file=open("german-news.mp3", "rb"),
)
```

**Form Fields:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | File | ✅ | Audio file. **WAV, MP3 and FLAC are always accepted.** M4A/AAC, OGG/Opus and WebM additionally require `ffmpeg` and `ffprobe` on the server host — no endpoint exposes whether they are present, so treat those formats as best-effort and handle the documented failure (see [Audio Errors](#audio-errors)) |
| `model` | String | ✅ | Multilingual non-turbo Whisper model (e.g. `mlx-community/whisper-large-v3-4bit`) |
| `language` | String | ❌ | **Source**-language hint (e.g. `de`). Auto-detected if omitted. Never sets the output language — output is always English. A documented superset of the OpenAI translations spec (which has no `language` field). |
| `prompt` | String | ❌ | Optional vocabulary/context bias hint (no synthetic default is injected on the translate path) |
| `response_format` | String | ❌ | `json` (default), `text`, `verbose_json` |
| `temperature` | Float | ❌ | Sampling temperature (default: 0.0 for greedy) |

**Response (JSON - default):**
```json
{
  "text": "Pioneers of the Frankfurt Aviation Association turn one hundred ..."
}
```

**Response (verbose_json):** identical shape to transcriptions, with `task` set to `"translate"`:
```json
{
  "task": "translate",
  "language": "de",
  "duration": 119.5,
  "text": "Pioneers of the Frankfurt Aviation Association turn one hundred ..."
}
```

> **Field semantics (same as transcriptions):** `duration` is the server-side processing
> wall-time in seconds, not the clip length. `language` echoes the requested source-language
> hint (or `"auto"`); it is **not** the output language (always English) and is never a
> server-detected code.

**Incompatible models are rejected up front — never a silent transcription:**

| Condition | Status | Example |
|-----------|--------|---------|
| Not an audio model at all | **400** | a text or vision model |
| Audio model that cannot translate | **422** | whisper-turbo (reduced decoder), `whisper-*.en` (no `<\|translate\|>` token), non-Whisper STT (Voxtral, VibeVoice) |

**Note:** This endpoint needs `mlx-audio` — included in the base install (Python 3.10–3.12).

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
| `model` | String | ✅ | Model ID. The backend serves a **single** model, so this is informational (the loaded model answers regardless). The response echoes the canonical `org/name` **selector** (and adds a `system_fingerprint` realization token — see Notes). |
| `input` | String or String[] | ✅ | One text, or a batch (one vector per item, in order). |
| `encoding_format` | String | ❌ | `base64` (**default** — little-endian float32, what the OpenAI SDK decodes) or `float` (raw JSON array, handy for `curl`). |
| `dimensions` | Integer | ❌ | Accepted only if equal to the model's native width; any other value → **400** (no Matryoshka truncation). |
| `user` | String | ❌ | Accepted and ignored (OpenAI passthrough). |
| `input_type` | String | ❌ | **mlxk extension** (RAG): `document` (default) or `query` (applies the model's query-instruction prefix). Ignored by standard OpenAI clients. |
| `instruct` | String | ❌ | **mlxk extension**: overrides the query task instruction; implies `input_type: query`. **Decoder embedders only (Qwen3)** — the BERT-family encoders (bge/e5) ignore this field; encoder support is pending. |

**Response (`encoding_format: float`):**
```json
{
  "object": "list",
  "data": [
    {"object": "embedding", "index": 0, "embedding": [0.0123, -0.0456, "..."]}
  ],
  "model": "mlx-community/bge-small-en-v1.5",
  "system_fingerprint": "a1b2c3d4.gpu",
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
- **`model` vs `system_fingerprint` (the same-model rule).** `model` is the clean, re-sendable
  **selector** (`org/name`, identical to the `/v1/models` id). `system_fingerprint` is the
  **realization token** `hash.device` (e.g. `a1b2c3d4.gpu`) — the change-detection signal. A vector
  space is fixed by the model, its revision/quant **and** the device (CPU vs GPU diverge ~0.98 cosine
  on a 4-bit model); any of those changing — the backend restarted on a different model, a re-quant
  under the same name, or a `--cpu` flip — flips `system_fingerprint`. **Compare it by equality:** pin
  a vector store to one `system_fingerprint`, and re-index the instant it differs instead of silently
  mixing incomparable vectors. `embed-serve`'s `GET /health` carries the same `model` +
  `system_fingerprint`, so — **talking directly to the backend port** — you can poll for a swap without
  embedding; **through the `serve` gateway the backend's `/health` is not exposed**, so detection is
  reactive (from the next response — see [Embeddings Backend](#embeddings-backend-embed-serve)). Treat the token as **opaque**
  (don't parse it). You still send a store's query and corpus to the **same backend/gateway** — the
  token lets you *detect* a mismatch, it doesn't reconcile one.
- **`system_fingerprint` is an additive extension.** It is a standard OpenAI field on
  chat/completions; on the embeddings response it is an mlxk addition carrying the same documented
  meaning ("the backend configuration changed"). Generic OpenAI clients ignore it; a RAG client reads
  it for change-detection.
- **Supported models:** the verified embedders (`docs/MODEL-COVERAGE.md`) — decoder
  (`Qwen3-Embedding-*`, via `mlx-lm`) and encoder (`bge-*`, `*-e5-*`, `mxbai-*`; `model_type: bert`).
  A declared-but-not-vendored embedder (e.g. `xlm-roberta`/`modernbert`) is rejected at backend
  **startup**, never silently.
- **Experimental:** the backend requires `MLXK2_ENABLE_ALPHA_FEATURES=1`.

---

### GET /v1/models

**List available models.**

Returns the runnable models — healthy and runtime-compatible — from both the
HF cache and the workspace home (`MLXK_WORKSPACE_HOME`, ADR-022). This is the
same set of models as the default human `mlxk list` view (without `--all`);
a model preloaded from outside the workspace home is included as well.
The preloaded model (if any) appears exactly once, sorted first; all other
models follow alphabetically.

> **Embedders are excluded.** Embedding models (e.g. `bge-*`, `Qwen3-Embedding-*`) are
> **not** listed here — they are served by the separate `embed-serve` backend, whose model
> list is not merged in. This is the one case where `/v1/models` differs from `mlxk list`,
> which *does* show embedders.

> **No per-model capability label and no `dimensions` field.** Entries carry no capability
> label (e.g. `chat` / `+vision` / `+audio`) — an unsupported modality is signalled at
> **request** time with HTTP **422** `capability_not_supported`, not advertised here — and no
> embedding `dimensions`
> (read it from the first `/v1/embeddings` response: the returned vector's length).

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
- `context_length`: Maximum context window in tokens, read from the model's `config.json`; `null` when the config states none — then no window guard applies and the generation budget is the ceiling alone

**Why context_length matters:**

MLX Knife uses **client-side context management** (unlike OpenAI's server-side history):
- **Vision models:** Fully stateless - client holds entire conversation history
- **Text models:** The server keeps no history either; every request carries the whole conversation as the prompt. The default generation budget is `min(32768, context_length − prompt tokens)`, and a prompt that fills the window is rejected with **400** `context_length_exceeded` before any token is generated (see [Token Limits](#token-limits-text-vs-multimodal-models))
- **Clients need this** to prune history so the prompt stays under the window, and to size their token budgets
- **Load balancing:** BROKE Cluster and similar tools use this for scheduling decisions

Note: LM Studio provides similar field as `max_context_length`.

---

### GET /health

**Liveness only — 200 OK means the process is up and answering.**

```json
{ "status": "healthy", "service": "mlx-knife-server-2.0" }
```

The response is a constant: it inspects neither the loaded model nor the inference backend, so
`"status": "healthy"` is not a statement about whether the next request will succeed. A process whose
backend has failed still answers `healthy`. Treat it as a liveness probe — it detects a dead or
unreachable server, not an unhealthy one — and not as a readiness or retry signal.

(The `embed-serve` backend has its own `/health` — see [Embeddings Backend](#embeddings-backend-embed-serve) — which returns `{"status": "ok", "model": "org/name", "system_fingerprint": "hash.device"}` and `503` until its model is loaded. The `system_fingerprint` matches the `/v1/embeddings` response, so a client **talking directly to the backend port** can poll that `/health` to detect a model/device swap without an embed request. **Through the `serve` gateway the backend's `/health` is not exposed** — a gateway client detects swaps reactively, from the next `/v1/embeddings` response.)

---

## Features & Capabilities

### Vision Support (2.0.4-beta.1)

See `examples/vision_pipe.sh` for a practical Vision→Text pipeline example (CLI).

**Supported:**
- ✅ Base64 data URLs (`data:image/jpeg;base64,...`)
- ✅ Multiple images (no count limit; processed in chunks of up to 5, see `chunk`)
- ✅ Formats: JPEG, PNG, GIF, WebP

**Limits:**
- **Per-image:** 20 MB max
- **Per request:** 50 MB of images in total; the count is not limited — images are processed in chunks of up to 5

**Important Characteristics:**

- **Stateless Server:** No server-side state required
- **Sequential Images:** Only images from the **last user message** are processed (OpenAI API compliant)
- **Each request is independent:** The model sees only the last user message (Metal memory limitations); the generation budget is the 2048-token vision ceiling alone, with no context-window guard

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
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "model=whisper-large"
```

**Supported:**
- ✅ File upload (multipart/form-data)
- ✅ Formats: WAV, MP3, FLAC — decoded in-process. Always available; depend on nothing outside the install
- ⚠️ Formats: M4A/AAC, OGG/Opus, WebM — **best-effort**. They are decoded by invoking `ffmpeg` **and** `ffprobe`, which do not ship with mlx-knife and must be installed on the server host. **No endpoint reports whether they are there**, so a client cannot negotiate this up front: either restrict uploads to the always-available set, or send the container format and handle the failure documented under [Audio Errors](#audio-errors)
- ✅ Response formats: `json`, `text`, `verbose_json`
- ✅ Language detection or explicit `language` parameter

**Models:** Whisper, Voxtral (needs `mlx-audio` — included in the base install)

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
- **Per-audio:** 50 MB max (same limit on both endpoints)
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

`max_tokens` counts *generated* tokens. Text and multimodal (Vision/Audio) requests resolve it
differently: text is guarded by the model's context window, multimodal is not.

#### Text Models (MLXRunner)

**Rule:** generation budget = `min(ceiling, context_length − prompt tokens)` — on
`/v1/chat/completions` and `/v1/completions` alike. The rule follows the loaded model's class, not
the request: a vision-capable model answering a text-only chat uses the vision ceiling below.

- **Ceiling:** the request's `max_tokens`, else the operator ceiling, else **32768** (see
  [Precedence](#precedence) below).
- **Window guard:** whatever the ceiling, the budget is clamped to what the context window still
  holds after the prompt — an explicit `max_tokens` too; the server never promises more than the
  window holds. The prompt is counted as the model sees it (after the chat template on
  `/v1/chat/completions`).
- **Full window:** `context_length − prompt tokens ≤ 0` is rejected before any token is generated
  with **400** `context_length_exceeded`; the error's `detail` carries `prompt_tokens` and
  `context_length`. On `stream: true` the reject is still an HTTP status, never an SSE event.
- **Unknown window:** when the model's `config.json` states no context length (`/v1/models`
  reports `null`), there is no guard — the budget is the ceiling alone.

**Example:** Llama-3.2-3B (128K context), 500-token prompt, no `max_tokens` → budget 32768.
The same request with `"max_tokens": 200000` → budget 130572, the window's remainder.

#### Vision/Audio Models (VisionRunner)

**Strategy:** stateless — each request is independent; with media, the model sees only the last
user message. Applies to every request a vision model serves, media or not.

**Default:** **2048** tokens on server and CLI, set explicitly (not inherited from mlx-vlm).
No window guard: the budget is the ceiling alone. The operator ceiling applies here too.

**Override** — an explicit `max_tokens` in the request:
```json
{
  "model": "mlx-community/gemma-3n-E2B-it-4bit",
  "messages": [...],
  "max_tokens": 4096
}
```

#### Precedence

1. Request `max_tokens` (must be ≥ 1; `0` or negative → **400** `validation_error`)
2. Operator ceiling: `mlxk serve --max-tokens N`, else `MLXK2_MAX_TOKENS=N` — one server-wide
   ceiling that replaces both defaults, text and vision. The flag wins over the environment.
   It must be a whole number ≥ 1; anything else refuses the start with one line, naming
   whichever of the two was used
3. Default: **32768** (text) / **2048** (vision)

Text budgets from every level are then clamped to the context window as above.

**Audio stands outside this chain.** Both `/v1/audio/*` endpoints always ask for **4096** tokens,
and a chat request against an audio model passes its own `max_tokens` through unclamped, falling
back to 4096. The operator ceiling reaches neither. The number is nominal in any case: the
transcription backend produces the whole transcript regardless of the budget it is handed.

#### finish_reason

Every completion reports how it ended — batch responses in `choices[0].finish_reason`, streams in
the final chunk before `data: [DONE]`:

- `"stop"` — the model ended its turn (EOS), or a `stop` sequence matched; a stream that ends on
  a sequence reports it in the final chunk as well.
- `"length"` — the generation budget cut the answer. This is the OpenAI value: a client can offer
  the user a "continue", raise `max_tokens`, or shorten the prompt. On chunked vision requests one
  cut chunk makes the whole response `"length"`.
- `null` — no outcome was recorded: the backend ended the generation without reporting a reason,
  or the stream failed part-way (see below).

These are OpenAI's values; `content_filter`, `tool_calls` and `function_call` are never emitted.

**A stream that fails part-way** keeps `finish_reason: null` — a backend fault is not a generation
outcome — and carries the failure in a top-level `error` object instead, in the same shape as an
HTTP error body:

```json
data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1702345678,"model":"...","choices":[{"index":0,"delta":{},"finish_reason":null}],"error":{"type":"internal_error","message":"..."}}
```

The tokens already sent stand; that event is the last one, and **no `[DONE]` follows** — the stream
did not complete. An OpenAI client needs no special handling: its SDK raises on the `error` key.
Batch responses never carry `error` in the body — they fail with an HTTP status.

The server logs one line per text generation — `Generation finished: <reason>` with `request_id`,
`model`, `stream`, `prompt_tokens`, `completion_tokens`, `max_tokens` and `finish_reason` — so a
cut answer is visible operator-side as well.

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
- ⚠️ **Batch mode only:** the answer is generated whole, then emitted as three `data:` events (role, content, `finish_reason`) and `[DONE]`
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

The final chunk carries `"finish_reason": "length"` instead of `"stop"` when the generation budget
cut the answer. A prompt that fills the context window never reaches the stream — it is rejected
with HTTP 400 `context_length_exceeded` before the response starts.

**Note:** `stream_options.include_usage` is not supported.

#### Closing the connection

A client that closes a streaming connection stops the generation. Measured: a 300-token
generation that takes 56 seconds to completion, with the client killed after 3 seconds, had
not finished 156 seconds later — the remaining 53 seconds of work were never done. The machine
is freed, not just the client.

Two consequences worth knowing:

- **Nothing is logged for the abandoned generation.** The completion line a finished generation
  writes never appears, so the server-side record shows the request starting and nothing else.
  Tokens already delivered are the client's; there is no way to resume.
- **The guarantee comes from the ASGI runtime, not from this server.** No code here watches for a
  disconnect. The runtime finalizes the response generator when the connection drops, and that
  closes the token generator. A deployment that buffers the response — a proxy that reads ahead,
  for instance — can therefore keep the generation running after the client is gone.

There is no explicit cancellation endpoint. Closing the connection is the way to abort.

### Embeddings Backend (embed-serve)

**Experimental.** Text embeddings run in a **separate process**, `mlxk embed-serve` —
not inside `mlxk serve`. This keeps the main server's memory gates (8 GB vision / 4 GB audio)
intact: an embedding model is never loaded into serve's address space. The backend exposes two
routes: `POST /v1/embeddings` (the OpenAI surface) and `GET /health` (liveness **+ identity** —
`200` with `{status, model, system_fingerprint}` once the model is loaded, `503` before).

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

> **Both halves are experimental and alpha-gated:** the `embed-serve` backend and the
> `serve --embed-backend` proxy. `GET /v1/models` on `serve` does **not** advertise the backend's
> embedders — embeddings work, but a client cannot discover the embedding model over HTTP. It has to
> be configured, or read from `mlxk list` on the host.

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

**Model identity & reachability.** Every `/v1/embeddings` response carries `model` +
`system_fingerprint` (the change-detection token; client store-discipline in
[Embeddings: Model Identity](#embeddings-model-identity--change-detection)). How a client reaches it
**proactively** depends on topology: talking **directly** to the backend port it can poll
`GET /health` (same `model` + `system_fingerprint`, no embed request); **through the `serve` gateway**
only `/v1/embeddings` is proxied — the backend's `/health` is not exposed — so a gateway client detects
a model/revision/device swap **reactively**, from the next embeddings response.

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
# Server binding — `serve` sets these from --host and --port on every start, so an exported
# value does not apply; the values shown are the flag defaults. Use the flags.
MLXK2_HOST=127.0.0.1
MLXK2_PORT=8000

# Logging — --log-level sets MLXK2_LOG_LEVEL the same way, so an exported value does not
# apply; an exported MLXK2_LOG_JSON=1 does apply without --log-json
MLXK2_LOG_JSON=1          # JSON logs (production)
MLXK2_LOG_LEVEL=info      # debug|info|warning|error

# Feature gates — open only for 1 / true / yes / on; every other value, 0 included, keeps them shut
MLXK2_ENABLE_PIPES=1              # Unix pipe integration (beta, 2.0.4-beta.1)
MLXK2_ENABLE_ALPHA_FEATURES=1     # Alpha: embed, embed-serve, serve --embed-backend

# Generation ceiling for max_tokens (text and vision; audio is not covered) — normally set for
# you by `serve --max-tokens N`, which wins over this variable. A whole number >= 1, or the
# server refuses to start. Text budgets stay clamped to the context window minus the prompt.
MLXK2_MAX_TOKENS=4096

# Set for you by a flag; an exported value applies when the flag is absent
MLXK2_PRELOAD_MODEL=mlx-community/Llama-3.2-3B-Instruct-4bit   # serve --model: load at startup, refuse to start if it cannot run
MLXK2_VISION_CHUNK_SIZE=1         # serve --chunk: images per vision inference, 1-5; a request's `chunk` wins
MLXK2_RELOAD=1                    # serve --reload: uvicorn auto-reload, development only

# Environment only — no flag sets these
MLXK_WORKSPACE_HOME=/path/to/workspaces   # workspace models resolve by name and appear in /v1/models
MLXK2_EXIF_METADATA=0             # drop the EXIF columns (location, date, camera) from the vision filename header
MLXK2_VISION_METADATA_CONTEXT=0   # do not prepend image metadata to the vision prompt
MLXK2_AUDIO_SEGMENTS=1            # append a segment table (start, end, text) to transcripts
MLXK2_DEBUG=1                     # print stream-error diagnostics to stdout

# Embeddings proxy (ADR-015) — set for you by `serve --embed-backend URL` and cleared without
# the flag, so an exported value cannot enable the proxy on its own. When unset,
# POST /v1/embeddings on serve returns 501.
MLXK2_EMBED_BACKEND=http://127.0.0.1:8002
```

### Supervised Mode (Default)

**Behavior:**
- Runs server in subprocess for improved signal handling
- **Stopping it:** Ctrl-C, `SIGTERM` and `SIGHUP` all take the same path — the server gets
  5s to finish, then it is killed. A second stop signal skips the rest of that grace. A
  script, a shell `trap`, launchd or systemd can therefore stop `mlxk serve` the way they
  stop anything else
- If the supervisor is killed outright (`SIGKILL`) or crashes, the server process notices
  and stops itself, so neither the model nor the port is left behind. Not covered: a server
  wedged inside a native call — no in-process mechanism can end that, only an external
  supervisor or the OS
- Logs go to stderr — application *and* access logs, with and without `--log-json` — so stdout
  stays clean for data
- `--log-json` produces 100% JSON output; without it Uvicorn's plain format applies
- **Note:** No auto-restart on crashes (use systemd/supervisor for production)

**Start:**
```bash
mlxk serve --port 8000 --log-json

# Capturing needs a stderr redirect — a bare `| tee` writes an empty file.
mlxk serve --port 8000 --log-json 2>&1 | tee serve.log   # capture and watch
mlxk serve --port 8000 --log-json 2> serve.log           # capture only
```

**Stop:**
```bash
kill "$SERVER_PID"        # graceful; the port is free once the process is gone
kill -9 "$SERVER_PID"     # the server notices and stops itself too
```
The exit status follows the shell convention: `143` (128+SIGTERM) when the server was
stopped by a signal — Ctrl-C included, because the supervisor stops the server with
SIGTERM — and `137` when it had to be forced. A server that exits on its own reports its
own code.

The supervised child inherits the parent's descriptors, so one shell redirect captures both
processes. There is no `--log-file` option.

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
- **400 Bad Request:** Invalid input (e.g., an oversized image or request, invalid format, validation failures incl. `max_tokens` below 1 — `validation_error`); a prompt that fills the model's context window (`context_length_exceeded`, `detail` carries `prompt_tokens` and `context_length`); for `/v1/embeddings`: empty or non-string `input` (incl. empty array items), unsupported `encoding_format` or `input_type`, or a non-native `dimensions` value)
- **404 Not Found:** Model not found in cache or workspace, or a spec that matches several models (`model_not_found`); no endpoint matches the request path (`not_found`)
- **405 Method Not Allowed:** The endpoint exists, but not for this method (`method_not_allowed`); the response carries an `Allow` header. `HEAD` is not accepted where only `GET` is declared
- **413 Payload Too Large:** Audio upload above the 50 MB limit (both audio endpoints) (`payload_too_large`)
- **422 Unprocessable Entity:** The model cannot serve the request (`capability_not_supported`): a request carrying images or audio while the loaded model is text-only — the modality is rejected, never silently dropped — or `POST /v1/audio/translations` against an audio model that cannot translate

### Server Errors (5xx)
- **500 Internal Server Error:** Unexpected backend failure
- **501 Not Implemented:** The server cannot run this (`not_implemented`): a missing dependency (mlx-lm, mlx-vlm or mlx-audio absent), Python below 3.10 for a vision model, an audio model of unknown backend, a checkpoint the runtime reports incompatible, or `POST /v1/embeddings` when `serve` has no `--embed-backend` configured (ADR-015)
- **502 Bad Gateway:** Embed backend unreachable / connection refused / connect-timeout (`bad_gateway`, **retryable**; `serve --embed-backend` proxy, ADR-015)
- **503 Service Unavailable:** Server shutting down (`server_shutdown`, retryable)
- **504 Gateway Timeout:** Embed backend read-timeout on a slow / large batch (`gateway_timeout`, **retryable**; `serve --embed-backend` proxy, ADR-015)
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

---

## Troubleshooting

### Vision/Audio Require Python 3.10+

mlx-knife itself requires Python 3.10+ (`requires-python >=3.10`), so a normal `pip install`
cannot land on 3.9. This 501 only appears when running from a source checkout on an unsupported
interpreter.

**Symptom:** HTTP 501 "Vision models require Python 3.10+"

**Solution:**
```bash
# Upgrade Python (3.10-3.12 required)
pyenv install 3.10
pyenv local 3.10

# Reinstall — the vision and audio backends are base dependencies,
# there are no extras to select
pip install mlx-knife
```

### Memory Constraint Errors (HTTP 507)

**Symptom:** `Model size (X GB) exceeds 70% of system memory (Y GB). Vision models crash with Metal OOM due to Vision Encoder overhead.`

**Solutions:**
1. Use smaller quantized model (e.g., 4-bit instead of 8-bit)
2. Add more system RAM
3. Try different model architecture

### Vision Responses Too Short

**Symptom:** Responses truncated mid-sentence

**Cause:** Default `max_tokens: 2048` might be too low for complex descriptions

**Solution:** raise `max_tokens` in the request:
```json
{
  "model": "mlx-community/Llama-3.2-11B-Vision-Instruct-4bit",
  "messages": [...],
  "max_tokens": 4096
}
```

### Image Upload Fails (HTTP 400)

**Common causes:**
- Image size > 20 MB per image
- More than 50 MB of images in one request
- Unsupported format (use JPEG, PNG, GIF, WebP)
- External URLs (not supported, use Base64 data URLs)
- Invalid Base64 encoding

**Solution:** Resize images, reduce count, or check encoding

### Audio Errors

#### Audio Request Fails

**HTTP 413** — the upload is above the 50 MB limit (hard limit, both endpoints), with
`error.type: "payload_too_large"`.

**HTTP 400** — any of:
- An empty file
- More than 1 audio per request (multi-audio not supported)
- Unsupported format (use WAV or MP3 for chat `input_audio`; WAV/MP3/M4A/FLAC/OGG for `/v1/audio/transcriptions`)
- Invalid Base64 encoding (chat endpoint only)

**Solution:** Compress audio, ensure single audio per request, use supported format

#### Container Format Fails Although It Is Listed (HTTP 500)

M4A/AAC, OGG/Opus and WebM are decoded by invoking external `ffmpeg` and `ffprobe`
executables. When either is absent from the `PATH` of the process running the server, the
upload is accepted and then fails during decoding — **HTTP 500** with `error.type:
"internal_error"`, `retryable: false`, and a message naming the missing tool:

```json
{
  "status": "error",
  "error": {
    "type": "internal_error",
    "message": "Transcription failed: ... ffmpeg not found! ... Install ffmpeg: macOS: brew install ffmpeg ...",
    "retryable": false
  },
  "request_id": "..."
}
```

The status code reflects where the failure surfaces, not its nature: this is a missing
deployment prerequisite, not a server fault, and retrying will not help.

**There is no way to detect this in advance.** Neither `GET /health` nor `GET /v1/models`
reports which decoders the host can reach, so a client cannot probe for it and cannot negotiate
it during setup. Two workable client strategies:

- **Avoid it:** upload only WAV, MP3 or FLAC. These never invoke an external tool, so they
  behave identically on every host.
- **Handle it:** send the container format and treat a 500 whose message names `ffmpeg` or
  `ffprobe` as a server-host configuration problem — surface it to the operator, do not retry,
  and fall back to transcoding client-side if you can.

Verified against mlx-knife **2.0.7** (the current PyPI release) with the tools absent; the
routing lives in the audio backend and is not affected by the 2.0.8 dependency wave.

**Solution:** install ffmpeg on the server host (`brew install ffmpeg` provides both binaries),
or restrict uploads to WAV, MP3 and FLAC, which never touch an external tool.

#### Audio Model Not Found

**Symptom:** `Model 'xxx' does not support audio inputs (no audio capability detected)`

**Cause:** Model lacks audio capability

**Solution:** Use an audio-capable model:
```bash
mlxk list | grep audio    # dedicated STT models list as `audio`, multimodal ones as `chat+audio`
```

**Note:** Some HuggingFace models may require `mlxk convert --repair-index` before use.

#### Audio Output is Garbled/Multilingual

**Symptom:** Transcription includes unexpected languages (Arabic, Hindi, etc.)

**Cause:** A non-zero sampling temperature. Every audio surface defaults to `0.0` (greedy),
`/v1/chat/completions` against an audio model included, so drift means the request set
`temperature` itself.

**Solution:** Drop `temperature` from the request, or send it as `0.0`:
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
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "model=whisper-large"

# For multimodal chat: Gemma-3n (use chat/completions instead)
# See "Audio Messages Format" in Appendix
```

#### mlx-audio Not Installed

**Symptom:** `STT models require mlx-audio`

**Cause:** `mlx-audio` is a base dependency, so this only appears when the install is
incomplete — most commonly on **Python 3.13 / macOS-ARM**, where the `miniaudio` wheel is
missing and the build fails.

**Solution:**
```bash
# Use Python 3.10-3.12, then reinstall
pip install --force-reinstall mlx-knife
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
| Images per request | No limit; processed in chunks | Chunking keeps each Metal batch small |
| Images per chunk | 5 (`chunk` maximum) | Metal API stability (tested) |
| Image size | 20 MB | Metal OOM prevention |
| Total image size | 50 MB | Metal OOM prevention |
| **Audio per request (chat)** | **1** | **mlx-vlm limitation** |
| **Audio size (both endpoints)** | **50 MB** (52,428,800 bytes) | **Measured in raw bytes, codec-agnostic. WAV @ 16 kHz mono 16-bit caps at ~27 min; compressed formats fit much more (verified: 55 min MP3 transcription via Whisper stays under the limit). Whisper handles long audio robustly; multimodal chat audio is bounded by `max_tokens` only (see Audio Support caveat).** |
| Vision model RAM | 70% system | Metal OOM prevention |
| Text model RAM | 70% (warning) | Swap tolerance |
| Vision max_tokens | 2048 (default) | Stateless, slow inference; set explicitly on server and CLI |
| Audio max_tokens | 4096, and the operator ceiling does not apply | Nominal — the transcription backend produces the whole transcript regardless |
| Text max_tokens | 32768 (default), clamped to context_length − prompt | Runaway guard |

---

## Migration Guide

### From 2.0.3 → 2.0.4

**New Features:**

| Feature | Endpoint | Requirements |
|---------|----------|--------------|
| Vision (images) | `/v1/chat/completions` | Base install (`mlx-vlm`) |
| Audio Chat (Gemma-3n) | `/v1/chat/completions` | Base install (`mlx-vlm`) |
| Audio STT (Whisper) | `/v1/audio/transcriptions` | Base install (`mlx-audio`) |
| Memory pre-load checks | All endpoints | Built-in (HTTP 507) |
| Server audio preload | `mlxk serve --model whisper-large` | Built-in |

**Breaking Changes:**

| Change | Before | After | Impact |
|--------|--------|-------|--------|
| Python version | 3.9+ | 3.10-3.12 | Upgrade required |
| Vision `max_tokens` default | 1024 | 2048 | Longer responses |
| Memory checks (Vision) | None | 70% RAM limit | HTTP 507 possible |

**New Dependencies (auto-installed):**
- `mlx-vlm==0.3.10` (Vision + Gemma-3n audio)
- `mlx-audio==0.3.1` (Whisper STT)
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
**not** advertise embedders.

**New error codes:** the proxy adds **502** `bad_gateway` (backend unreachable) and **504**
`gateway_timeout` (backend read-timeout) — both **retryable** (ADR-015). Backend `4xx/5xx`
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
| Audio translation | New CLI flag `mlxk run --audio FILE --translate` and server endpoint `POST /v1/audio/translations` (Whisper, multilingual non-turbo; non-capable models rejected 400/422, never silently transcribed). See [POST /v1/audio/translations](#post-v1audiotranslations). |

**Client updates required:** none for existing chat/audio/models clients. A RAG client can now
point at one base URL (serve's port) for both `/v1/chat/completions` and `/v1/embeddings` once
`serve --embed-backend` is configured. A RAG client persisting a vector store must honor the
same-model rule — pin the store to the response `system_fingerprint` and re-index when it changes
(see [Appendix: Embeddings](#embeddings-model-identity--change-detection)).

---

### From 2.0.7 → 2.0.8

> Unreleased. This records what the tree carries beyond released 2.0.7.

**Endpoint surface:** unchanged. **Response shapes** move in two places: `finish_reason` gains
`"length"`, and the `error` a failed stream carries is an object where it was a string. A new **400**
error type `context_length_exceeded` exists. Request shapes are unchanged. See *Generation budget*
below.

**Two rejects stop looking like faults.** An audio upload above the size limit (**413**) and
`POST /v1/audio/translations` against a model that cannot translate (**422**) now carry
`payload_too_large` and `capability_not_supported`. Both statuses were already correct; the
`error.type` beside them said `internal_error`, so a client routing on the type could not tell a
deliberate reject from a server fault. A client that special-cased `internal_error` on those two
statuses should drop that branch.

**Process behaviour changed.** `mlxk serve` takes the same teardown path for Ctrl-C, `SIGTERM` and
`SIGHUP`, and a server whose supervisor is killed stops itself instead of holding the port. Exit
status follows the shell convention: `143` when stopped by a signal, `137` when it had to be forced.
See [Supervised Mode](#supervised-mode-default).

**Dependency bumps (auto-installed):**

| Package | 2.0.7 | 2.0.8 |
|---------|-------|-------|
| `mlx` | `>=0.30.0,<0.32` | `>=0.30.0,<0.32.1` |
| `mlx-vlm` | `==0.6.2` | `==0.6.10` |
| `transformers` | `==5.5.4` | `==5.14.1` |
| `torch` | `>=2.0` (base dep) | **removed** |
| `torchvision` | `>=0.15` (base dep) | **removed** |

(`mlx-lm==0.31.3` unchanged; `mlx-audio` moves `0.4.4` → `0.4.8`.)

**Why `torch` + `torchvision` go away:** mlx-vlm #1011 — the Pixtral / Mistral-Small-3.1 processor
pulling torch in as a base dependency — is resolved as of `mlx-vlm 0.6.4`. The sunset marker they
carried since 2.0.6 (ADR-023 Workaround-Sunset Policy) is therefore retired. Re-verified torch-free
in the 2.0.8 dependency wave: `pixtral`, `mistral3` and `gemma4`, plus `qwen2_5_vl` and `qwen3_5`,
verified for the first time under it; `mllama` keeps its 2.0.6 verification.

**Install size impact:** the base install shrinks by 524 MB (36 %, measured against a fresh 2.0.7
install). Operators on size-constrained images can drop the allowance they were told to plan for in 2.0.6.

**Behavior changes:**

| Change | Effect on operators |
|--------|---------------------|
| Torch-free install | Packaging only. No endpoint or schema change, and no change to which model types are gated — those gates never keyed on torch. |
| Model listing follows the pin set | `/v1/models` stays the authority on what this server can run; a dependency wave can shift which models qualify. No API contract change. Per-model detail lives in `docs/MODEL-COVERAGE.md`, not here. |
| More vision models are listed | A check withheld every checkpoint carrying `temporal_patch_size` while transformers reported 5.x. Those models load and answer correctly, so it is gone and they appear. A client that hard-coded the shorter list should re-read `/v1/models`. |

**Generation budget** ([#66](https://github.com/mzau/mlx-knife/issues/66)) — for **text**,
`min(ceiling, context_length − prompt tokens)`, the same rule the CLI applies. Vision and audio
keep their own ceiling with no window guard; see
[Token Limits](#token-limits-text-vs-multimodal-models):

| Change | 2.0.7 | 2.0.8 | Effect on clients |
|--------|-------|-------|-------------------|
| Text default `max_tokens` | `context_length / 2` | `min(32768, context_length − prompt tokens)` | On a 128K model: 65536 → 32768. The halving was a static reservation for history under the name "shift-window"; the reservation is now exact — the prompt that is actually there. |
| Explicit `max_tokens` | passed through | clamped to `context_length − prompt tokens` | Never more than the window holds. |
| `finish_reason` | `"stop"`, or `"error"` on a failed stream | `"stop"`, `"length"`, or `null` | A cut answer is reported as such. `"error"` is gone — it was never an OpenAI value. |
| Failed stream | `finish_reason: "error"`, `error` a message string, then a second chunk saying `"stop"` and `[DONE]` | `finish_reason: null`, `error` an object (`type`, `message`), stream ends there | The only breaking change in this release. |
| Prompt fills the window | budget ignored the prompt; prompt + output could exceed the window | **400** `context_length_exceeded` before any token | `detail.prompt_tokens` / `detail.context_length` say how much to shorten. |
| `max_tokens` below 1 | accepted | **400** `validation_error` | |
| `/v1/models` `context_length` | `4096` when `config.json` states no window | `null` | The number was invented; `null` means "no window guard". |
| Vision / audio-chat default | 2048 on the server, inherited from mlx-vlm on the CLI | 2048, set explicitly on both | No wire change. |
| `max_completion_tokens` | ignored | ignored | Unchanged — use `max_tokens`. |

**Client updates required:**
- Handle `finish_reason: "length"` — offer "continue", raise `max_tokens`, or shorten the prompt.
- Drop any branch keyed on `finish_reason: "error"`, and read a failed stream's `error` as an object
  rather than a string. An OpenAI SDK client needs no change: it raises on the `error` key either way.
- Accept `null` for `/v1/models` `context_length`; deserializing it as a non-nullable integer breaks.
- Handle **400** `context_length_exceeded` by shortening history; `detail` carries the two numbers.
- Clients that relied on the 64K default on 128K models must pass `max_tokens` explicitly (still
  clamped to the window's remainder).

---

## References

- **Architecture Principles:** `docs/ARCHITECTURE.md`
- **Verified Multimodal Coverage:** `docs/MODEL-COVERAGE.md` (per-release operation × model_type matrix)

### ADRs (development decisions)

- **ADR-004:** Enhanced Error Logging (the error-type taxonomy that `bad_gateway`/`gateway_timeout` extend)
- **ADR-012:** Vision Support
- **ADR-015:** Embeddings API (`/v1/embeddings` backend, `embed-serve`, `serve --embed-backend` proxy; 502/504 gateway codes)
- **ADR-016:** Memory-Aware Loading (HTTP 507 rationale)
- **ADR-020:** Audio Backend Architecture (STT routing, MLX_AUDIO vs MLX_VLM)
- **ADR-022:** Workspace-First Paradigm (background; surface-transparent on the server)
- **ADR-023:** Text-First + Verified Multimodal (HTTP 501 `unsupported_multimodal` policy + the Workaround-Sunset Policy that retired the `torch` / `torchvision` base deps)
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
       {"role": "user", "content": "describe"},
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
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "model=whisper-large" \
  -F "language=en" \
  -F "response_format=json"
```

**Form Fields:**

| Field | Required | Description |
|-------|----------|-------------|
| `file` | ✅ | Audio file. **WAV/MP3/FLAC always accepted**; M4A/AAC, OGG/Opus, WebM are best-effort — they need `ffmpeg` + `ffprobe` on the server host, which the client cannot detect |
| `model` | ✅ | Model ID (e.g., `whisper-large`, full HF path) |
| `language` | ❌ | Language code (`en`, `de`, etc.). Auto-detect if omitted. |
| `prompt` | ❌ | Optional context to guide transcription |
| `response_format` | ❌ | `json` (default), `text`, `verbose_json` |
| `temperature` | ❌ | Sampling temperature (default: 0.0) |

**Response Formats** — `json` (default):

```json
{"text": "Hello world."}
```

`verbose_json`:

```json
{"task": "transcribe", "language": "en", "duration": 2.5, "text": "Hello world."}
```

`text`:

```text
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
- Maximum file size: 50 MB — see [Limits Summary](#limits-summary) for what that means per format
- Requires `mlx-audio` on the server — included in the base install (Python 3.10–3.12)

### Embeddings: Model Identity & Change Detection

Embedding clients (RAG) use the OpenAI **Embeddings** API (`POST /v1/embeddings`). One requirement is
non-obvious and, if ignored, corrupts a vector store **silently** (no error): a stored vector is only
comparable to another from the **same model, revision, and device**.

The response carries two identity fields:

```json
{
  "object": "list",
  "data": [{"object": "embedding", "index": 0, "embedding": "..."}],
  "model": "mlx-community/bge-small-en-v1.5",
  "system_fingerprint": "a1b2c3d4.gpu",
  "usage": {"prompt_tokens": 4, "total_tokens": 4}
}
```

- `model` — the **selector** (`org/name`): what you send, what `/v1/models` lists. Stable, re-sendable.
- `system_fingerprint` — the **realization token** `hash.device`: the change-detection signal. It flips
  when the backend's model, its revision/quant, or its device (`--cpu` vs GPU) changes — the three
  things that change the vector space. Additive mlxk field (standard on OpenAI chat/completions).

**Client MUST, for any persisted vector store:**

- **Stamp the store with the `system_fingerprint` it was built under, and compare on every use.** When
  the served value differs, the vector space changed → **re-index** (or refuse to mix), never silently
  append. (Reaching identity proactively — direct-backend `/health` poll vs. reactive-via-response
  through the gateway — is covered under [Embeddings Backend](#embeddings-backend-embed-serve).)
- **Build a store with one backend** (= one model + one device): send its query and corpus to the same
  `serve` / gateway URL. The token *detects* a mismatch; it does not reconcile one.
- Treat `system_fingerprint` as **opaque** — compare by equality, never parse the segments.

The RAG extension fields (`input_type: "document" | "query"`, `instruct`) and `encoding_format`
(`base64` default / `float`) are documented under [POST /v1/embeddings](#post-v1embeddings).

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

- **Unreleased:** 2.0.8 — generation budget, `finish_reason`, stream failures
  - **CHANGED:** default text `max_tokens` is `min(32768, context_length − prompt tokens)`; an explicit value is clamped to the window too.
  - **NEW:** `finish_reason: "length"` when the budget cut the answer.
  - **NEW: 400** `context_length_exceeded` — prompt fills the window, rejected before any token; `detail` carries `prompt_tokens` and `context_length`. A status even on `stream: true`.
  - **CHANGED:** `max_tokens` below 1 → **400** `validation_error`.
  - **CHANGED:** `/v1/models` `context_length` is `null` when the config states no window (was a hard-coded `4096`).
  - **CHANGED:** a failed stream carries a top-level `error` object, keeps `finish_reason: null`, and ends.
  - **DOCUMENTED:** closing a streaming connection stops the generation; nothing is logged for it. Behaviour unchanged.
  - **FIXED:** **413** and **422** carry `payload_too_large` / `capability_not_supported`; both reported `internal_error` before, so a deliberate reject looked like a server fault.
  - **FIXED:** `/v1/models` lists vision models it wrongly withheld — a check rejected every checkpoint carrying `temporal_patch_size` under transformers 5.x, and those models load and answer correctly.
  - **CHANGED:** `mlxk serve` takes one teardown path for Ctrl-C, `SIGTERM` and `SIGHUP`, and stops itself if its supervisor dies. Exit `143` on signal, `137` when forced.
  - Dep-wave: `mlx-vlm==0.6.10`, `mlx-audio==0.4.8`, `transformers==5.14.1`, `mlx>=0.30.0,<0.32.1`; `torch`/`torchvision` dropped as base deps (524 MB smaller install).
  - Before/after per change, and what clients must update: *From 2.0.7 → 2.0.8* in the Migration Guide.

- **2026-07-24:** 2.0.7 stable — embeddings + audio translation, embeddings model identity
  - **NEW:** the `/v1/embeddings` response (and `embed-serve` `/health`) carries
    `system_fingerprint` = `hash.device` — a change-detection token so a RAG client detects a
    model/revision/device swap; `model` stays the clean `org/name` selector (= `/v1/models` id).
    Additive field (standard on OpenAI chat/completions). ADR-015 §Model Identity.
  - **NEW:** `/v1/embeddings` (OpenAI Embeddings API), served by the new `mlxk embed-serve`
    backend (separate single-model process; ADR-015). Experimental — requires
    `MLXK2_ENABLE_ALPHA_FEATURES=1`.
  - `encoding_format`: `base64` (default, SDK-compatible) and `float`; batch `input`; L2-normalized
    vectors; mlxk extensions `input_type` / `instruct` for RAG query embedding (`instruct` applies on
    the decoder path only).
  - Topology: `serve --embed-backend URL` proxies `/v1/embeddings` to the backend (same
    release). Proxy errors: **501** (no `--embed-backend` configured), **502** `bad_gateway`
    (backend unreachable), **504** `gateway_timeout` (read-timeout) — 502/504 retryable; backend
    `4xx/5xx` pass through verbatim. `GET /v1/models` does not advertise embedders.
  - **NEW:** Whisper audio-to-English translation — CLI `mlxk run --audio FILE --translate`
    and server `POST /v1/audio/translations` (multilingual non-turbo; OpenAI-compatible
    translations endpoint, hardcoded `task=translate`; non-capable models rejected 400/422).
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

**📝 Note:** This handbook tracks the server and changes when the server changes. The Changelog above
is the record of what changed and when; `Last Updated` is maintained by hand and is the weaker of
the two. Neither is reachable from the running server — see *Which server does this describe?* at
the top.
