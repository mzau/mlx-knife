# MLX Knife Server Handbook

**Version:** 2.0.8, unreleased — the tree as it stands; the 2.0.8 betas published so far carry part
of it. The latest stable release is 2.0.7: its endpoint surface is the same, its request and response
shapes differ in places, and the [Migration Guide](#migration-guide) records every difference.
**Scope:** what the server does today. Planned work, deferred features and target releases are
deliberately absent — this is a contract, not a roadmap.
**Last Updated:** 2026-09-23

> **Audience:** Server operators, DevOps, API consumers
> **For implementation details:** See `ARCHITECTURE.md` and `docs/ADR/` (developer documentation)

> **The server does not report its own version** — `GET /health` returns the family string
> `mlx-knife-server-2.0` — and there is no capability negotiation. The Changelog at the end records
> when each endpoint appeared. One behaviour a client cannot probe: container audio formats depend
> on tooling installed on the server host (see [Audio Errors](#audio-errors)).

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
- Python 3.11–3.14; every version in that range installs from wheels. `mlx-audio` is a **base** dependency — there is no audio-free install.
- `mlx>=0.30.0,<0.32.1`
- `mlx-lm==0.31.3` (text backend)
- `mlx-vlm==0.6.10` (vision + multimodal audio)
- `mlx-audio==0.4.8` (STT backend)
- `transformers==5.14.1` (required by `mlx-vlm >=0.6.5`)
- **no `torch` / `torchvision`**

The pins are exact (ADR-023), `mlx` excepted: an upstream bump goes through an mlx-knife release. Do not loosen them on `pip install`.

> **On released 2.0.7 (PyPI):** the previous pin set — `mlx-vlm==0.6.2`, `transformers==5.5.4`, `torch`/`torchvision` as base deps. *From 2.0.7 → 2.0.8* in the [Migration Guide](#migration-guide) lists every difference.

---

## OpenAI API Compatibility

MLX Knife implements a **subset** of the OpenAI API with documented behavioral differences.

### Supported Endpoints

| Endpoint | Status | Notes |
|----------|--------|-------|
| `/v1/chat/completions` | ✅ Supported | Text, Vision (`image_url`), Audio (`input_audio`) |
| `/v1/completions` | ✅ Supported | Legacy text completion |
| `/v1/audio/transcriptions` | ✅ Supported | OpenAI Whisper API |
| `/v1/audio/translations` | ✅ Supported | OpenAI Whisper translations API — speech→English (multilingual non-turbo Whisper; non-capable models → 400/422). See [POST /v1/audio/translations](#post-v1audiotranslations) |
| `/v1/embeddings` | ✅ Supported (experimental) | OpenAI Embeddings API. Served by the separate `embed-serve` backend; `serve` proxies it via `--embed-backend`. Returns **501** on a plain `serve` started without `--embed-backend` (embeddings not enabled). See [Embeddings Backend](#embeddings-backend-embed-serve) |
| `/v1/models` | ✅ Supported | HF cache + workspace models (ADR-022); extended with `context_length` and `loaded` fields. Does **not** list embedders — they belong to the separate `embed-serve` backend |
| `/health` | ✅ Custom | MLX Knife extension — `live`: `200` while the process runs; no model or backend state (see [GET /health](#get-health)) |

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
OpenAI surface (`serve`: `/v1/chat/completions`, `/v1/embeddings`, `/v1/models`, `/v1/audio/*`;
`embed-serve`: `/v1/embeddings`) is callable **directly from a browser**.

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
- Same ID appears in error response body as `"request_id"`. Known issue
  ([#80](https://github.com/mzau/mlx-knife/issues/80)): an error the embedding backend returns
  through `serve --embed-backend` carries the backend's own ID in its body
- Use for request correlation and distributed tracing

### Behavioral Deviations from OpenAI

| Behavior | OpenAI | MLX Knife | Reason |
|----------|--------|-----------|--------|
| Vision history | Full history to model | Images or audio in the last user message: only that message | Prevents pattern reproduction (hallucinations) |
| Image URLs | HTTP URLs + Base64 + File IDs | Base64 data URLs only | No external fetching |
| Audio+Vision | Both processed | The audio is dropped, the images are processed | |
| Multi-audio | Supported | 1 per request | mlx-vlm limitation |
| Error format | `{"error": {"message", "type", "code"}}` | ADR-004 envelope (see below) | Richer error context |
| `max_completion_tokens` | Preferred | Silently ignored — the request falls to `max_tokens`, else the default ceiling | Unknown request fields are dropped, not rejected |
| `stream_options` without `"stream": true` | Rejected with 400 | Ignored — the response carries `usage` anyway | Clients that send it on every request keep working |
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

**Error types** — every type a server response can carry:

| Type | HTTP | Meaning |
|------|------|---------|
| `validation_error` | 400 | Invalid request payload (e.g. an image over 20 MB or more than 50 MB of images in one request, malformed audio, `max_tokens` below 1) |
| `context_length_exceeded` | 400 | The prompt fills the model's context window; nothing is left to generate. The message names prompt tokens and window, `detail` carries both as `{"prompt_tokens", "context_length"}`; never retryable |
| `model_not_found` | 404 | Model spec does not resolve to a cached / workspace model |
| `not_found` | 404 | No endpoint matches the request path — most often a base URL that already ends in `/v1` |
| `method_not_allowed` | 405 | The endpoint exists, but not for this method; the response carries an `Allow` header |
| `payload_too_large` | 413 | Audio upload above the 50 MB limit, on either audio endpoint |
| `capability_not_supported` | 422 | The request is well-formed and the feature exists, but *this* model cannot serve it — images or audio in the last user message while the loaded model is text-only (media in earlier messages become placeholders, see [Cross-Model Workflows](#cross-model-workflows-visionaudio--text)), or `/v1/audio/translations` against a model that cannot translate |
| `server_shutdown` | 503 | Lifespan shutdown in progress; new requests are rejected |
| `insufficient_memory` | 507 | Model exceeds the memory threshold (ADR-016) |
| `not_implemented` | 501 | The server cannot run this: a missing dependency, an audio model of unknown backend, a checkpoint the runtime reports incompatible (a declared `model_file` included), or `/v1/embeddings` without `--embed-backend` |
| `bad_gateway` | 502 | Embed backend (`serve --embed-backend`) unreachable / connection failed / connect-timeout (retryable; ADR-015) |
| `gateway_timeout` | 504 | Embed backend read-timeout on a slow / large batch (retryable; ADR-015) |
| `internal_error` | 500 | Unexpected backend failure |

(`bad_gateway` / `gateway_timeout` are raised only by the `serve --embed-backend` proxy; a backend's own `4xx`/`5xx` envelopes otherwise pass through verbatim.)

The CLI's `--json` output has error types of its own; see `docs/json-api-specification.md`.

**Status and type always agree.** A fixed mapping binds the two, so routing on either one gives the
same answer; the type is the more specific of the two where a status carries two meanings — 400
(`validation_error`, `context_length_exceeded`) and 404 (`model_not_found`, `not_found`). Every other
status maps to exactly one type.

Two types describe a request the server declines rather than fails. `not_implemented` — the server
cannot run this. `capability_not_supported` — the feature exists and the request is fine, but the
model you named cannot serve it. The cases you will meet: images or audio in the last user message
for a text-only model, and `POST /v1/audio/translations` against a whisper-turbo or `.en` variant
(see [the reject matrix](#post-v1audiotranslations)). Neither is `retryable`.

---

## API Endpoints

### POST /v1/chat/completions

**OpenAI-compatible chat completion endpoint.**

Every request is formatted with the model's chat template (a plain fallback when the model has
none); no request field bypasses it. `/v1/completions` takes a prompt the client formats itself.

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
- `chunk` (integer, optional): Batch size for vision processing (default: 1). Controls how many images are processed per inference session. Higher values may trigger OOM on resource-constrained systems. Maximum: 5 (enforced by server). `usage` sums the chunks.

**Also honored** (standard OpenAI sampling fields): `top_p` (default `0.9`) and
`repetition_penalty` (default `1.1`), in addition to `temperature` and `max_tokens`. `temperature`
defaults to **0.7**, and to **0.0** against an audio model on any surface. An explicit value always wins, except on the vision
paths, where `temperature` is fixed at 0.0 (greedy decoding, to keep descriptions from drifting) and
the value sent is ignored; `top_p` and `repetition_penalty` do apply there.

`stop` (string or list of strings) ends the answer at the first sequence that matches; the
sequence itself is removed. The match is against the model's text: the image-metadata header
that precedes a vision answer is not searched. What that costs differs by surface:

- **Batch:** the sequences are applied to the finished text, so the answer ends where OpenAI says
  it ends and `finish_reason` is `"stop"` — but the tokens generated past the cut were generated,
  and still count in `usage`. A chunked vision stream is a batch answer per chunk: the chunk's
  text is cut the same way, and no later chunk is generated.
- **Stream:** each token is checked as it is emitted, and the last chunk with a choice reports `"stop"`. The
  check is per token, so a sequence split across two of them is not seen, and the token carrying a
  match has already been sent — the answer ends one token late rather than exactly at the sequence.
- **Dedicated STT (Whisper, VibeVoice) through chat completions:** the transcript is returned whole;
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

**OpenAI Whisper API compatible audio transcription.**

Use this endpoint for **direct file upload** transcription with STT models (Whisper, VibeVoice).

**Request (multipart/form-data):**
```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "model=mlx-community/whisper-large-v3-turbo-4bit" \
  -F "language=en" \
  -F "response_format=json"
```

**Form Fields:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | File | ✅ | Audio file. **WAV, MP3 and FLAC are always accepted.** M4A/AAC, OGG/Opus and WebM additionally require `ffmpeg` and `ffprobe` on the server host — no endpoint exposes whether they are present, so treat those formats as best-effort and handle the documented failure (see [Audio Errors](#audio-errors)) |
| `model` | String | ✅ | Model ID (e.g. `mlx-community/whisper-large-v3-turbo-4bit`) |
| `language` | String | ❌ | Language code (e.g., `en`, `de`). Auto-detect if omitted. |
| `prompt` | String | ❌ | Optional context to guide transcription |
| `response_format` | String | ❌ | `json` (default), `text`, `verbose_json` |
| `temperature` | Float | ❌ | Sampling temperature (default: 0.0 for greedy); for Whisper the only decoding temperature, see [Audio Support](#audio-support) |

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

**Supported Models:** Whisper and VibeVoice; the verified checkpoints are listed in
`docs/MODEL-COVERAGE.md`.

**Translation:** for audio-to-English translation, use the dedicated
[`POST /v1/audio/translations`](#post-v1audiotranslations) endpoint (or the CLI
`mlxk run --audio FILE --translate`). Both require a multilingual non-turbo Whisper model.

**vs. `/v1/chat/completions` with `input_audio`:**

| Feature | `/v1/audio/transcriptions` | `/v1/chat/completions` |
|---------|---------------------------|------------------------|
| Format | Multipart file upload | Base64 in JSON |
| Models | STT only (Whisper, VibeVoice) | Multimodal (Gemma-3n) or STT |
| Use case | Pure transcription | Chat with audio context |
| OpenAI API | Whisper API | Chat Completions API |

---

### POST /v1/audio/translations

**OpenAI Whisper API compatible speech-to-English translation.**

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
| `temperature` | Float | ❌ | Sampling temperature (default: 0.0 for greedy); for Whisper the only decoding temperature, see [Audio Support](#audio-support) |

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

---

### POST /v1/embeddings

**OpenAI Embeddings API compatible text embeddings (experimental).**

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
| `model` | String | ✅ | Model ID. The backend serves a **single** model, so this is informational (the loaded model answers regardless). The response names the served model and adds a `system_fingerprint` realization token — see Notes. |
| `input` | String or String[] | ✅ | One text, or a batch (one vector per item, in order). |
| `encoding_format` | String | ❌ | `base64` (**default** — little-endian float32, what the OpenAI SDK decodes) or `float` (raw JSON array, handy for `curl`). |
| `dimensions` | Integer | ❌ | Accepted only if equal to the model's native width; any other value → **400** (no Matryoshka truncation). |
| `user` | String | ❌ | Accepted and ignored (OpenAI passthrough). |
| `input_type` | String | ❌ | **mlxk extension** (RAG): `document` (default) or `query` (applies the model's query-instruction prefix). Ignored by standard OpenAI clients. |
| `instruct` | String | ❌ | **mlxk extension**: overrides the query task instruction; implies `input_type: query`. **Decoder embedders only (Qwen3)** — the BERT-family encoders (bge/e5) ignore this field. |

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
- **`model` vs `system_fingerprint` (the same-model rule).** `model` identifies the model served by
  the embedding backend — its `org/name` when the model was pulled or cloned from the Hub.
  Embedders are not listed by `serve`'s `/v1/models`; read the served identity from an embeddings
  response, or directly from the embedding backend's `/health`.
  `system_fingerprint` is the **realization token** `hash.device` (e.g. `a1b2c3d4.gpu`) — the
  change-detection signal. A vector
  space is fixed by the model, its revision/quant **and** the device (CPU and GPU vectors differ);
  any of those changing — the backend restarted on a different model, a re-quant
  under the same name, or a `--cpu` flip — flips `system_fingerprint`. **Compare it by equality:** pin
  a vector store to one `system_fingerprint`, and re-index the instant it differs instead of silently
  mixing incomparable vectors. Known issue ([#81](https://github.com/mzau/mlx-knife/issues/81)): for
  some workspace models the vectors depend on the workspace's path — the same model files under
  another name return different vectors, while `model` and `system_fingerprint` stay the same.
  `embed-serve`'s `GET /health` carries the same `model` +
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

Returns the runnable models — healthy on disk (the file-integrity check of `mlxk health`) and
runtime-compatible — from both the HF cache and the workspace home (`MLXK_WORKSPACE_HOME`,
ADR-022). This is the same set of models as the default human `mlxk list` view (without `--all`);
a model preloaded from outside the workspace home is included as well.
The preloaded model (if any) appears exactly once, sorted first; all other
models follow alphabetically. Being listed is a prediction: a listed model can still be refused
when a request names it.

> **Embedders are excluded.** Embedding models (e.g. `bge-*`, `Qwen3-Embedding-*`) are
> **not** listed here — they are served by the separate `embed-serve` backend, which has no model
> list. `mlxk list` does show embedders. Known issue: embedding models that also take images
> are listed here as chat models; they are not supported.

> **No per-model capability label and no `dimensions` field.** Entries carry no capability
> label (e.g. `chat` / `+vision` / `+audio`) — a text-only model answers images or audio in the last
> user message at **request** time with HTTP **422** `capability_not_supported`; it is not advertised
> here — and no embedding `dimensions`
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
      "context_length": 131072,
      "loaded": true
    },
    {
      "id": "mlx-community/Llama-3.2-3B-Instruct-4bit",
      "object": "model",
      "owned_by": "mlx-knife-2.0",
      "permission": [],
      "context_length": 8192,
      "loaded": false
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
- `context_length`: The model's context window in tokens, taken from its `config.json`, or `null`
  when none is known. Text models are budgeted against it; with `null` they have no window guard and
  the budget is the ceiling alone (see [Token Limits](#token-limits-text-vs-multimodal-models)).
  `null` means unknown, not unlimited.
- `loaded`: `true` on the model in memory now, so a request naming this `id` is served without
  loading it; `false` on every other model, and on all of them while nothing is loaded or a model is
  loading. It says where the weights are, not that the next request will succeed.

**Why context_length matters:**

MLX Knife uses **client-side context management** (unlike OpenAI's server-side history):
- **Vision models:** Fully stateless - client holds entire conversation history
- **Text models:** The server keeps no history either; every request carries the whole conversation as the prompt. The default generation budget is `min(32768, context_length − prompt tokens)`, and a prompt that fills the window is rejected with **400** `context_length_exceeded` before any token is generated (see [Token Limits](#token-limits-text-vs-multimodal-models))
- **Clients need this** to prune history so the prompt stays under the window, and to size their token budgets

---

### GET /health

**`live` — a `200` means the process is running and able to answer.** The status code is the answer.

```json
{"status": "ok", "service": "mlx-knife-server-2.0"}
```

The endpoint reads no model and no backend state. It answers while the server works — during a
generation, a model load, a vision answer, a transcription or a model listing — because none of that
runs on the loop that answers it. Reading a request does: a large upload delays it while it is read.

What the server can say about its state, one word per question:

| Question | Word | Ask | Logged on stderr | Answerable |
|----------|------|-----|------------------|------------|
| Is the process alive and able to answer? | `live` | `GET /health` → `200` | startup, shutdown | yes |
| Can it accept a request now? | `ready` | the same `200` | — | yes — the same answer as `live` on this server |
| Is this model in memory? | `loaded` | `loaded` on a row of [`GET /v1/models`](#get-v1models) | each model load | yes |
| Is a generation still producing, and since when? | `progressing` | — | a finished text-model generation, image answer or transcription | **no** — the server keeps no clock on a generation |
| Is the model complete on disk? | `healthy` | `mlxk health`; `health` in `mlxk list --json` and `mlxk show --json` | — | yes — never on the HTTP surface |

**`ready` is `live` here.** Without `--reload`, the port opens only once startup has finished — a
`--model` preload included, which lasts as long as the model takes to load — and closes when shutdown
begins, so a server that answers accepts requests. Accepting is not being served at once: a request
waits while another model operation runs (see [Concurrent Requests](#concurrent-requests)). During
startup a connection is refused, not answered with `503`; give a probe that much grace before reading
a refused connection as a dead server. A preload that fails ends the process.

**What a `200` does not tell you:**
- **That a generation is progressing.** A stalled generation — a GPU fault, a hang inside a native
  call — leaves this endpoint answering. Only the client waiting for the answer sees it: detect it with
  your own request timeout. `max_tokens` bounds tokens, not time, and the server never times a
  generation out.
- **That the next request will succeed.** A process whose inference backend has failed keeps answering
  `200` here while it fails requests; restart it from outside (see [Supervised Mode](#supervised-mode-default)).
- **Which model is loaded.** Read `loaded` on [`GET /v1/models`](#get-v1models).

(The `embed-serve` backend has its own `/health` — see [Embeddings Backend](#embeddings-backend-embed-serve) — which returns `{"status": "ok", "model": …, "system_fingerprint": "hash.device"}`. Its port opens only once its model has loaded, so every answer comes from a loaded model: with a single model, `ready` and `loaded` are one fact. `model` and `system_fingerprint` match the `/v1/embeddings` response, so a client **talking directly to the backend port** can poll that `/health` to detect a model/device swap without an embed request. **Through the `serve` gateway the backend's `/health` is not exposed** — a gateway client detects swaps reactively, from the next `/v1/embeddings` response.)

---

## Features & Capabilities

### Vision Support

See `examples/pipes/vision_pipe.sh` for a practical Vision→Text pipeline example (CLI).

**Supported:**
- ✅ Base64 data URLs (`data:image/jpeg;base64,...`)
- ✅ Multiple images (no count limit; processed in chunks of up to 5, see `chunk`)
- ✅ Formats: JPEG, PNG, GIF, WebP

**Limits:**
- **Per-image:** 20 MB max
- **Per request:** 50 MB of images in total; the count is not limited — images are processed in chunks of up to 5

**Important Characteristics:**

- **Stateless Server:** No server-side state required
- **Sequential Images:** Only images from the **last user message** are processed
- **Each request is independent:** when the last user message carries images, the model sees only that message; when it carries neither images nor audio, the model answers from the conversation's text (see [Vision: Stateless Prompt](#vision-stateless-prompt-history-based-ids)). The generation budget has no context-window guard (see [Token Limits](#token-limits-text-vs-multimodal-models))

#### Stable Image IDs (History-Based)

Image numbers ("Image 1, 2, 3...") stay stable across requests: the server scans the full `messages[]` array (which clients send with each request per OpenAI API) and assigns IDs chronologically based on content hash:

```
Request 1: beach.jpg (hash: 5733332c) → Image 1
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

---

### Audio Support

**Two methods** for audio transcription:

#### Method 1: `/v1/audio/transcriptions` (Whisper API)

**Direct file upload** for STT models (Whisper, VibeVoice). Recommended for pure transcription.

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "model=mlx-community/whisper-large-v3-turbo-4bit"
```

**Supported:**
- ✅ File upload (multipart/form-data)
- ✅ Formats: WAV, MP3, FLAC — decoded in-process. Always available; depend on nothing outside the install
- ⚠️ Formats: M4A/AAC, OGG/Opus, WebM — **best-effort**. They are decoded by invoking `ffmpeg` **and** `ffprobe`, which do not ship with mlx-knife and must be installed on the server host. **No endpoint reports whether they are there**, so a client cannot negotiate this up front: either restrict uploads to the always-available set, or send the container format and handle the failure documented under [Audio Errors](#audio-errors)
- ✅ Response formats: `json`, `text`, `verbose_json`
- ✅ Language detection or explicit `language` parameter

**Models:** Whisper, VibeVoice

#### Method 2: `/v1/chat/completions` with `input_audio`

**Base64-encoded audio** in chat messages — for multimodal models (Gemma-3n), and for STT models,
which return the transcript.

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

**Limits (both methods):**
- **Per-audio:** 50 MB max (same limit on both endpoints)
- **Count:** 1 audio per request

> **Caveat — multimodal chat audio.** A multimodal model (Gemma-3n) can keep generating past the
> end of the audio; `max_tokens` (default 2048) is the only bound. Keep chat audio short.

**Models:** Gemma-3n (Vision + Audio + Text); STT models (Whisper, VibeVoice)

**Important Characteristics:**

- **Stateless Server:** Same as Vision — no server-side state
- **Single Audio:** Only one audio file per request
- **Audio+Vision:** When both are present in chat, the audio is dropped and only the images are processed
- **Temperature:** Fixed at 0.0 for a multimodal model (the vision path); an STT model defaults to
  0.0. Whisper decodes at that one value, the default included: a window that fails Whisper's quality
  check is not retried at a higher temperature ([#74](https://github.com/mzau/mlx-knife/issues/74))

**History Handling:**

When switching from Audio to Text model mid-conversation:
- Server replaces `input_audio` blocks in earlier messages with a placeholder; audio in the last user
  message is rejected (see [Cross-Model Workflows](#cross-model-workflows-visionaudio--text))
- Text model sees `[n audio(s) were attached]` placeholder

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
- **Unknown window:** when no context window is known for the model (`/v1/models` reports `null`),
  there is no guard — the budget is the ceiling alone.

**Example:** Llama-3.2-3B (128K context), 500-token prompt, no `max_tokens` → budget 32768.
The same request with `"max_tokens": 200000` → budget 130572, the window's remainder.

#### Vision/Audio Models (VisionRunner)

**Strategy:** stateless — each request is independent; when the last user message carries images or
audio, the model sees only that message.

**Default:** **2048** tokens on server and CLI, for every request a vision model serves, with media or
without; set explicitly (not inherited from mlx-vlm). No window guard: the budget is the ceiling
alone. The operator ceiling applies here too.

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

**Audio stands outside this chain.** The server sets no token budget for a transcription model.
A chat request's `max_tokens` reaches the model unchanged and unclamped; without one, and on both
`/v1/audio/*` endpoints, which take no budget field, the model's own default applies. The operator
ceiling reaches none of them. What the budget does depends on the model: one that transcribes in a
single pass stops at it, so a long transcript can end early at the model's default and a larger
`max_tokens` lifts that; one that decodes in fixed windows, such as Whisper, takes no budget and
ignores the value.

#### finish_reason

Every completion reports how it ended — batch responses in `choices[0].finish_reason`, streams in
the last chunk that carries a choice:

- `"stop"` — the model ended its turn (EOS), or a `stop` sequence matched; a stream that ends on
  a sequence reports it there as well.
- `"length"` — the generation budget cut the answer. This is the OpenAI value: a client can offer
  the user a "continue", raise `max_tokens`, or shorten the prompt. On chunked vision requests one
  cut chunk makes the whole response `"length"`.
- `null` — no outcome was recorded: the backend ended the generation without reporting a reason —
  always the case for a transcription model —, or the stream failed part-way (see below).

These are OpenAI's values; `content_filter`, `tool_calls` and `function_call` are never emitted.

**A stream that fails part-way** keeps `finish_reason: null` — a backend fault is not a generation
outcome — and carries the failure in a top-level `error` object instead, with the `type` and `message` of
an HTTP error body:

```json
data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1702345678,"model":"...","choices":[{"index":0,"delta":{},"finish_reason":null}],"error":{"type":"internal_error","message":"..."}}
```

The tokens already sent stand; that event is the last one, and **no `[DONE]` follows** — the stream
did not complete. An OpenAI client needs no special handling: its SDK raises on the `error` key.
A batch request never fails inside a `200` body — it fails with an HTTP error status.

Each text-model generation that runs to its end logs a completion line, `Generation finished: <reason>`.
Under `--log-json` the line carries `request_id`, `model`, `stream`, `prompt_tokens`,
`completion_tokens`, `max_tokens` and `finish_reason` as fields.

---

### Memory-Aware Loading (ADR-016)

**Pre-load memory checks prevent OOM crashes.**

#### Vision Models and Multimodal Audio
- **Threshold:** 70% system RAM
- **Behavior:** Model size > 70% → HTTP 507 (Insufficient Storage)
- **Rationale:** Vision Encoder has unpredictable per-image overhead

**Example (64GB system):**
- Llama-3.2-11B-Vision (5.6GB) → ✅ Loads (8.75% of RAM)
- Llama-3.2-90B-Vision (46.4GB) → ❌ HTTP 507 (72.5% of RAM)

#### Text Models
- **Threshold:** 70% system RAM
- **Behavior:** Model size > 70% → **Warning only**; the model loads and swaps

---

### Streaming (SSE - Server-Sent Events)

#### Text Models
- ✅ **True streaming:** Tokens streamed as generated
- **Format:** SSE (`data: {...}\n\n`)
- **Completion:** `data: [DONE]\n\n`
- A text request to a vision model is served by that model and arrives as one event (see [Vision Models](#vision-models))

#### Vision Models
- ✅ **Per-chunk streaming:** Real SSE events as each image chunk completes (2.0.4-beta.7+)
- **Multiple images:** Each chunk (1-5 images) streams as it finishes processing
- **Single image:** Behaves like batch mode (one SSE event); so does a request without images
- **Format:** OpenAI-compatible SSE with per-chunk deltas

#### Audio Models
- ⚠️ **Batch mode only:** the answer is generated whole, then emitted as `data:` events for role, content and `finish_reason`, and `[DONE]`
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

The last chunk with a choice carries `"finish_reason": "length"` instead of `"stop"` when the generation budget
cut the answer. A prompt that fills the context window never reaches the stream — it is rejected
with HTTP 400 `context_length_exceeded` before the response starts.

#### Token usage in a stream

A stream carries no token counts unless the request asks for them, on `/v1/chat/completions` and
`/v1/completions` alike: `"stream_options": {"include_usage": true}`. Every chunk then carries
`"usage": null`, and one more chunk follows the last one with a choice — its `choices` is empty and
its `usage` holds the counts for the whole request. `data: [DONE]` comes after it:

```json
data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1702345678,"model":"mlx-community/Llama-3.2-3B-Instruct-4bit","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":null}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1702345678,"model":"mlx-community/Llama-3.2-3B-Instruct-4bit","choices":[],"usage":{"prompt_tokens":12,"completion_tokens":8,"total_tokens":20}}
```

On `/v1/completions` both are `text_completion` objects. A stream that fails part-way or is
interrupted ends without the usage chunk. Without `"stream": true` the option is ignored — the
response carries `usage` anyway.

#### Closing the connection

Closing a streaming connection stops the generation at the stream's next step: a text stream after
the token being generated, a multi-image vision stream after the chunk being generated. Known issue
([#82](https://github.com/mzau/mlx-knife/issues/82)): a stream that arrives as one event (see
[Streaming](#streaming-sse---server-sent-events)) has no such step — its generation runs to the end,
and requests behind it wait (see [Concurrent Requests](#concurrent-requests)). Whatever runs before
a response starts, a model load included, completes either way. A stream whose client left before
its response started can still run its first step — for a multi-image stream, one whole image chunk.

- **A stream that stopped early writes no completion line;** a multi-image stream keeps the lines of
  the chunks it delivered. A stream that arrives as one event runs to the end and logs as its
  non-streaming form does. There is no way to resume.
- **The guarantee comes from the ASGI runtime, not from this server.** No code here watches for a
  disconnect. The runtime finalizes the response generator when the connection drops, and the
  generation takes no further step. A deployment that buffers the response — a proxy that reads
  ahead, for instance — can therefore keep the generation running after the client is gone.

There is no explicit cancellation endpoint. Closing the connection is the only way to abort, and it
stops only a stream that has steps.

### Embeddings Backend (embed-serve)

**Experimental.** Text embeddings run in a **separate process**, `mlxk embed-serve` —
not inside `mlxk serve`: an embedding model is never loaded into serve's address space. The backend exposes two
routes: `POST /v1/embeddings` (the OpenAI surface) and `GET /health` (readiness **+ identity** —
`200` with `{status, model, system_fingerprint}`; the port opens only once the model has loaded).

**Topology — one OpenAI surface:**
```bash
# Embedding backend — separate process, owns the model, localhost-internal
MLXK2_ENABLE_ALPHA_FEATURES=1 mlxk embed-serve bge-small-en-v1.5 --port 8002

# Main server — proxies /v1/embeddings to the backend; clients use ONE base URL
MLXK2_ENABLE_ALPHA_FEATURES=1 mlxk serve --model chat-model --embed-backend http://127.0.0.1:8002
```
A RAG client points at `serve` for both `/v1/embeddings` and `/v1/chat/completions`, or calls the
backend port directly.

> **Both halves are experimental and alpha-gated:** the `embed-serve` backend and the
> `serve --embed-backend` proxy. `GET /v1/models` on `serve` does **not** list the backend's
> embedders — embeddings work, but a client cannot list the embedding models over HTTP. The served
> model's identity is in every embeddings response and in the backend's own `GET /health`.

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
`--cpu` — the single Metal GPU stays free for chat (on unified memory this trades GPU contention,
not RAM).

**Memory:** an embedding model is small (~300 MB–1 GB). On RAM-constrained machines, don't start
`embed-serve`.

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
  supervisor or the OS; `GET /health` keeps answering meanwhile, so a probe cannot find it
  (see [GET /health](#get-health))
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
- `python -m` searches the directory it is started in before the installed packages — unlike
  `mlxk serve`, which keeps that directory out. Start it where you trust the files, or add
  `-P`

**Start:**
```bash
python -P -m mlxk2.core.server_base
```

---

## HTTP Status Codes

### Success
- **200 OK:** Request successful

### Client Errors (4xx)
- **400 Bad Request:** Invalid input (e.g., an oversized image or request, invalid format, validation failures incl. `max_tokens` below 1 — `validation_error`); a prompt that fills the model's context window (`context_length_exceeded`, `detail` carries `prompt_tokens` and `context_length`); for `/v1/embeddings`: empty or non-string `input` (incl. empty array items), unsupported `encoding_format` or `input_type`, or a non-native `dimensions` value)
- **404 Not Found:** Model not found in cache or workspace (`model_not_found`); no endpoint matches the request path (`not_found`)
- **405 Method Not Allowed:** The endpoint exists, but not for this method (`method_not_allowed`); the response carries an `Allow` header. `HEAD` is not accepted where only `GET` is declared
- **413 Payload Too Large:** Audio upload above the 50 MB limit (both audio endpoints) (`payload_too_large`)
- **422 Unprocessable Entity:** The model cannot serve the request (`capability_not_supported`): images or audio in the last user message while the loaded model is text-only — the modality is rejected, never silently dropped; media in earlier messages become placeholders — or `POST /v1/audio/translations` against an audio model that cannot translate

### Server Errors (5xx)
- **500 Internal Server Error:** Unexpected backend failure
- **501 Not Implemented:** The server cannot run this (`not_implemented`): a missing dependency (mlx-lm, mlx-vlm or mlx-audio absent), an audio model of unknown backend, a checkpoint the runtime reports incompatible (a declared `model_file` included), or `POST /v1/embeddings` when `serve` has no `--embed-backend` configured (ADR-015)
- **502 Bad Gateway:** Embed backend unreachable / connection refused / connect-timeout (`bad_gateway`, **retryable**; `serve --embed-backend` proxy, ADR-015)
- **503 Service Unavailable:** Server shutting down (`server_shutdown`, retryable)
- **504 Gateway Timeout:** Embed backend read-timeout on a slow / large batch (`gateway_timeout`, **retryable**; `serve --embed-backend` proxy, ADR-015)
- **507 Insufficient Storage:** Memory constraints violated (vision or multimodal-audio model >70% RAM, ADR-016)

---

## Performance Characteristics

### Model Loading
- **Time:** Paid by a request whose model is not `loaded`
- **Caching:** One model stays loaded until a request names another or the server stops; `GET /v1/models` marks it `loaded`
- **Memory:** Held until then — no endpoint unloads it

### Inference Speed

Throughput depends on model and hardware. A vision request pays for the vision encoder on top of
generation, per chunk of images (default 1, max 5 via `--chunk`).

### Concurrent Requests
- **One model operation at a time.** Loading a model, a batch answer, one step of a stream, a vision
  chunk and a transcription never overlap. Requests naming the same model are accepted concurrently
  and wait for their turn; there is no queue limit and no busy status.
- **A stream shares the turn step by step** — a text stream token by token, a multi-image vision
  stream image chunk by chunk. A request that arrives mid-stream is served between two steps, and the
  stream pauses for as long as that request's work takes — a whole batch answer, a model load, a
  transcription. A stream that arrives as one event takes the turn whole.
- **`GET /health` and `GET /v1/models` do not wait** — they answer while a model operation runs.
- **One model in memory.** A request naming another model unloads the current one first, also while
  another request still needs it. That request fails: with **500** `internal_error`, *Model not
  loaded*, while its response has not started (no HTTP status has reached the client); once a
  stream's response has started, with the terminal `error` event (see [finish_reason](#finish_reason)).
  A stream that arrives as one event starts its response only when its answer is ready. The server
  does not arbitrate between requests naming different models — do not overlap them; `loaded` on
  [`GET /v1/models`](#get-v1models) says which model is in memory.
- **Leaving does not cancel a batch request.** A client that closes a non-streaming request — its
  own timeout included — leaves the generation running until it ends on its own, and requests behind
  it wait. Closing a stream stops it at its next step; one that arrives as one event runs to the end,
  a known issue (see [Closing the connection](#closing-the-connection)).
- **Reason:** Metal backend, single GPU.

---

## Troubleshooting

### Memory Constraint Errors (HTTP 507)

**Symptom:** `Model size (X GB) exceeds 70% of system memory (Y GB). Vision models crash with Metal OOM due to Vision Encoder overhead.`

**Solution:** Use a smaller quantization (e.g. 4-bit instead of 8-bit) or a smaller model.

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
- Unsupported format (WAV or MP3 for chat `input_audio`; on the `/v1/audio/*` endpoints WAV, MP3 and FLAC, plus M4A/AAC, OGG/Opus and WebM when the host has `ffmpeg`)
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

**Solution:** install ffmpeg on the server host (`brew install ffmpeg` provides both binaries),
or restrict uploads to WAV, MP3 and FLAC, which never touch an external tool.

#### Model Has No Audio Capability

**Symptom:** `Model 'xxx' does not support audio inputs (no audio capability detected)`

**Solution:** Use an audio-capable model:
```bash
mlxk list | grep audio    # STT models list as `audio`, multimodal ones carry `+audio` (e.g. `chat+vision+audio`)
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

**Cause:** `/v1/audio/transcriptions` only works with STT models (Whisper, VibeVoice)

**Solution:** Use the correct model type:
```bash
# For transcription endpoint: STT models
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "model=mlx-community/whisper-large-v3-turbo-4bit"

# For multimodal chat: Gemma-3n (use chat/completions instead)
# See "Audio Messages Format" in Appendix
```

#### mlx-audio Not Installed

**Symptom:** `STT models require mlx-audio`

**Cause:** `mlx-audio` is a base dependency, so this only appears when the install is
incomplete — an interrupted `pip install`, or an interpreter outside the supported range, where
no macOS-ARM `miniaudio` wheel exists and the build from source fails.

**Solution:**
```bash
# Use Python 3.11-3.14, then reinstall
pip install --force-reinstall mlx-knife
```

### Embeddings Errors (experimental)

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
| Images per chunk | 5 (`chunk` maximum) | Metal API stability |
| Image size | 20 MB | Metal OOM prevention |
| Total image size | 50 MB | Metal OOM prevention |
| **Audio per request (chat)** | **1** | **mlx-vlm limitation** |
| **Audio size (both endpoints)** | **50 MB** (52,428,800 bytes) | Raw bytes, codec-agnostic: WAV at 16 kHz mono 16-bit fits ~27 min; compressed formats fit more |
| Vision model RAM | 70% system | Metal OOM prevention |
| Text model RAM | 70% (warning) | Swap tolerance |
| Vision max_tokens | 2048 (default) | No context-window guard on the vision path |
| Audio max_tokens | None of its own — the model's default; a chat request's `max_tokens` passes unclamped, the operator ceiling does not apply | A transcription ends with its audio; a server-side cap could only cut it short |
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
| Memory checks (Vision) | None | 70% RAM limit | HTTP 507 possible |

**New Dependencies (auto-installed):**
- `mlx-vlm==0.3.10` (Vision + Gemma-3n audio)
- `mlx-audio==0.3.1` (Whisper STT)
- `python-multipart>=0.0.9` (file uploads)

**Client Updates Required:**
- Handle HTTP 507 (Insufficient Storage) for large Vision models
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
| ADR-023 Text-First + Verified Multimodal | `mlxk convert --quantize` rejects multimodal types outside the verified list (CLI). No server change. |
| Workspace model spec (CLI) | `--model <path-to-workspace-dir>` accepted by `mlxk serve --model` (path-based; transparent to API consumers). |

**Client-visible:** none.

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

**Endpoint surface:** unchanged. Requests gain one optional field, `stream_options` (see
[Token usage in a stream](#token-usage-in-a-stream)). The tables below list every change a client
or operator sees, 2.0.7 against 2.0.8. Corrections to earlier handbooks, with no change in
behaviour, are listed under *Documented* in the [Changelog](#changelog).

**413 and 422 carry their own types.** An audio upload above the size limit (**413**) and
`POST /v1/audio/translations` against a model that cannot translate (**422**) carry
`payload_too_large` and `capability_not_supported`; 2.0.7 said `internal_error` beside the same
statuses. A client that special-cased `internal_error` on those two statuses should drop that branch.

**Requests and responses:**

| Change | 2.0.7 | 2.0.8 | Effect on clients |
|--------|-------|-------|-------------------|
| Images or audio in the last user message for a text-only model | **200**, the media dropped | **422** `capability_not_supported` | Send media only to a model that takes it; media in earlier messages stay allowed. |
| Images for `qwen2_5_vl` / `qwen3_5` checkpoints | **200**, answered without the image | answered with the image | |
| `"model": ""` | answered by an arbitrary model | **404** `model_not_found` | |
| A checkpoint whose `config.json` declares `model_file` | loaded, its module executed | **501** `not_implemented`; not listed | |
| Unmatched path, wrong method | `{"detail": …}` | ADR-004 envelope: **404** `not_found`, **405** `method_not_allowed` with `Allow` | Route on `error.type`. |
| `stop` on a batch request | ignored — the whole answer | the text ends at the first match, `finish_reason: "stop"` | Tokens generated past the cut still count in `usage`. |
| `usage` on text and vision | a word-count estimate | the runner's token counts | Audio and embeddings keep the estimate. |
| `temperature` unset, chat against an audio model | `0.7` | `0.0` | |
| Streaming | many times slower than the same generation unstreamed | as fast as unstreamed | |

**Operators:**

| Change | 2.0.7 | 2.0.8 |
|--------|-------|-------|
| `serve --max-tokens`, `MLXK2_MAX_TOKENS` | reached no request | the ceiling for text and vision |
| A feature gate set to `0` (`MLXK2_ENABLE_ALPHA_FEATURES`, `MLXK2_ENABLE_PIPES`, `MLXK2_DEBUG`) | opened the gate | keeps it shut; only `1`, `true`, `yes`, `on` open it |
| Python modules in the start directory | `serve` imported a `mlxk2/`, `fastapi.py` or `uvicorn/` there instead of the installed package | not imported |
| A transcription | wrote `transcript.txt` into the server's start directory, over an existing one | writes nothing there |
| `serve --json` with a rejected option | two JSON documents on stdout | one error document |
| Python | 3.10–3.12 | 3.11–3.14 |

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
install).

**Behavior changes:**

| Change | Effect on operators |
|--------|---------------------|
| Torch-free install | Packaging only. No endpoint or schema change, and no change to which model types are gated. |
| Model listing follows the pin set | `/v1/models` stays the authority on what this server can run; a dependency wave can shift which models qualify. No API contract change. Per-model detail lives in `docs/MODEL-COVERAGE.md`, not here. |
| More vision models are listed | A check withheld every checkpoint carrying `temporal_patch_size` while transformers reported 5.x. Those models load and answer correctly, so it is gone and they appear. A client that hard-coded the shorter list should re-read `/v1/models`. |

**Generation budget** ([#66](https://github.com/mzau/mlx-knife/issues/66)) — for **text**,
`min(ceiling, context_length − prompt tokens)`, the same rule the CLI applies. Vision keeps its own
ceiling with no window guard, and a transcription model runs at its own default; see
[Token Limits](#token-limits-text-vs-multimodal-models):

| Change | 2.0.7 | 2.0.8 | Effect on clients |
|--------|-------|-------|-------------------|
| Text default `max_tokens` | `context_length / 2` | `min(32768, context_length − prompt tokens)` | On a 128K model: 65536 → 32768. The prompt is subtracted as it is, not reserved for. |
| Explicit `max_tokens` | passed through | clamped to `context_length − prompt tokens` | Never more than the window holds. |
| `finish_reason` | `"stop"`, or `"error"` on a failed stream | `"stop"`, `"length"`, or `null` | A cut answer is reported as such. `"error"` is gone — it was never an OpenAI value. |
| `finish_reason` of a transcription on `/v1/chat/completions` | `"stop"` | `null` | Batch and stream alike. |
| Failed stream | `finish_reason: "error"`, `error` a message string, then a second chunk saying `"stop"` and `[DONE]` | `finish_reason: null`, `error` an object (`type`, `message`), stream ends there | |
| Prompt fills the window | budget ignored the prompt; prompt + output could exceed the window | **400** `context_length_exceeded` before any token | `detail.prompt_tokens` / `detail.context_length` say how much to shorten. |
| `max_tokens` below 1 | accepted | **400** `validation_error` | |
| `/v1/models` `context_length` | `4096` when no window was known | `null` | `null` means no window guard, not an unlimited window. |
| Vision / audio-chat default | 2048 on the server, inherited from mlx-vlm on the CLI | 2048, set explicitly on both | No wire change. |
| `max_tokens` against a transcription model | dropped before the model; it always ran at its own default | reaches the model unclamped; without it, the model's own default | A single-pass transcript that ended early can be completed with a larger `max_tokens`. |
| `max_completion_tokens` | ignored | ignored | Unchanged — use `max_tokens`. |

**Server state** ([#64](https://github.com/mzau/mlx-knife/issues/64)) — see [GET /health](#get-health):

| Change | 2.0.7 | 2.0.8 | Effect on clients |
|--------|-------|-------|-------------------|
| `GET /health` body | `{"status": "healthy", …}` | `{"status": "ok", …}` | The status code was and is the answer. `healthy` names a model's file integrity in the CLI and no longer appears on the HTTP surface. |
| `GET /health` and `GET /v1/models` while the server works | no answer for the whole of a non-streaming generation, a model load, a vision answer or a transcription; a model listing held up `GET /health` | answer while the server works; a large upload delays them while it is read | No longer silent for the length of a generation. |
| `loaded` on `GET /v1/models` | absent | `true` on the model in memory, `false` on every other row | Additive. |
| The loaded model, named by its listed `id` | loaded again when it had been loaded under another spelling | served from memory | |
| Concurrency | documented as one request at a time | one model operation at a time; a stream shares its turn step by step, and a model switch can fail a request that is already under way | Do not overlap requests naming different models. See [Concurrent Requests](#concurrent-requests). |

**Client updates required:**
- Decide on `GET /health` by its status code; a check for `"status": "healthy"` fails from this release on.
- Keep your own request timeout: `GET /health` answering does not mean a generation is progressing.
- Do not overlap requests that name different models. A model switch can fail an in-flight request:
  before its response starts with HTTP **500** `internal_error`; after an SSE response has started
  with the terminal `error` event, `finish_reason: null`, and no `[DONE]`.
- Accept a boolean `loaded` on `GET /v1/models` rows.
- Handle `finish_reason: "length"` — offer "continue", raise `max_tokens`, or shorten the prompt.
- Accept `finish_reason: null` on a transcription; a check for `"stop"` fails from this release on.
- Drop any branch keyed on `finish_reason: "error"`, and read a failed stream's `error` as an object
  rather than a string. An OpenAI SDK client needs no change: it raises on the `error` key either way.
- Accept `null` for `/v1/models` `context_length`; deserializing it as a non-nullable integer breaks.
- Handle **400** `context_length_exceeded` by shortening history; `detail` carries the two numbers.
- Send images or audio in the last user message only to a model that takes them: a text-only model
  answers **422** `capability_not_supported` instead of answering without the media. Media in
  earlier messages stay allowed and reach a text model as placeholders.
- Read an unmatched path (**404**) and a wrong method (**405**) from the error envelope's `type`,
  not from a `detail` key.
- Expect `stop` to cut a batch answer; it was accepted and ignored there.
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
- **ADR-023:** Text-First + Verified Multimodal (the verified-multimodal reject of `mlxk convert --quantize`; the Workaround-Sunset Policy that retired the `torch` / `torchvision` base deps)
- **ADR-024:** Pre-Execution Capability-Mismatch Reject (Class A — CLI-side; surface-transparent on the server today)
- **ADR-025:** content_hash v2 (background; surface-transparent on the server)
- **ADR-029:** Server State Vocabulary (`live` on `GET /health`, `loaded` on `GET /v1/models`; `healthy` stays the CLI's file-integrity word)

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

**Note:** For Vision models, a request whose last user message carries images forwards only that message to the model (stateless prompt); the server still scans the full history for image ID reconstruction.

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
| **Prompt to model** | Images in the last user message: only that message | Prevents pattern reproduction (model copying old mappings) |
| **Image ID assignment** | Full history scanned | Consistent numbering across session (Image 1, 2, 3...) |

**What this means:**
- When the last user message carries images, the Vision model sees only that message. With neither
  images nor audio there, it answers from the conversation's text; images sent earlier are not seen
  again
- Image numbering remains stable across the conversation
- The Vision model describes each image on its own; for questions across images or about their
  descriptions, switch to a **Text model** (see [Cross-Model Workflows](#cross-model-workflows-visionaudio--text))

**Recommended workflow:**
```
1. Vision model: User sends beach.jpg → "Image 1 shows a beach..."
2. Vision model: User sends mountain.jpg → "Image 2 shows a mountain..."
3. Text model: User asks "Compare these two locations" → Full context available
```

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
   | 1 | image_5733332c.jpeg | image_5733332c.jpeg | 📍 34.0522°N, 118.2437°W | 📅 2024-06-15 | iPhone 14 |

   </details>

   A sandy beach with blue water.
   ```

   **Note:** EXIF columns (Original, Location, Date, Camera) are enabled by default.
   Disable with `MLXK2_EXIF_METADATA=0` for minimal output (Image, Filename only). A data URL
   carries no file name, so *Original* repeats the generated name.

3. **Client Storage Optimization:** Client can **drop Base64 from history**, keep only the text and
   the assistant response as the server sent it:
   ```json
   {"role": "user", "content": "describe"}
   {"role": "assistant", "content": "<details>\n<summary>📸 Image Metadata (1 image)</summary>\n\n<!-- mlxk:filenames -->\n| Image | Filename | Original | Location | Date | Camera |\n|-------|----------|----------|----------|------|--------|\n| 1 | image_5733332c.jpeg | image_5733332c.jpeg | 📍 34.0522°N, 118.2437°W | 📅 2024-06-15 | iPhone 14 |\n\n</details>\n\nA sandy beach with blue water."}
   ```

4. **Request 3 (Vision after Text):** Client sends mountain.jpg with text-only history
   ```json
   {
     "messages": [
       {"role": "user", "content": "describe"},
       {"role": "assistant", "content": "<details>\n<summary>📸 Image Metadata (1 image)</summary>\n\n<!-- mlxk:filenames -->\n| Image | Filename | Original | Location | Date | Camera |\n|-------|----------|----------|----------|------|--------|\n| 1 | image_5733332c.jpeg | image_5733332c.jpeg | 📍 34.0522°N, 118.2437°W | 📅 2024-06-15 | iPhone 14 |\n\n</details>\n\nA sandy beach with blue water."},
       {"role": "user", "content": "What color?"},
       {"role": "assistant", "content": "Blue."},
       {"role": "user", "content": [
         {"type": "text", "text": "new picture"},
         {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
       ]}
     ]
   }
   ```

5. **Server Reconstruction:** Server scans history:
   - Finds `<!-- mlxk:filenames -->` marker in assistant response
   - Parses: `image_5733332c.jpeg` → Image ID 1
   - Assigns: mountain.jpg → Image ID 2 ✅

**Client Recommendations:**
- **After first Vision request:** Drop Base64 image_url from history, keep text + assistant response
- **History format:** Text-only user messages + full assistant responses (with mapping tables)
- **⚠️ Preserve verbatim:** Do not sanitize or strip HTML comments from assistant responses — the `<!-- mlxk:filenames -->` markers are required for ID reconstruction

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
- ❌ Audio + Vision combined: the audio is dropped, only the images are processed

### Audio Transcriptions (File Upload)

For direct STT transcription with dedicated models (Whisper, VibeVoice), use the `/v1/audio/transcriptions` endpoint:

**Request (multipart/form-data):**
```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "model=mlx-community/whisper-large-v3-turbo-4bit" \
  -F "language=en" \
  -F "response_format=json"
```

**Form Fields:**

| Field | Required | Description |
|-------|----------|-------------|
| `file` | ✅ | Audio file. **WAV/MP3/FLAC always accepted**; M4A/AAC, OGG/Opus, WebM are best-effort — they need `ffmpeg` + `ffprobe` on the server host, which the client cannot detect |
| `model` | ✅ | Model ID (e.g. `mlx-community/whisper-large-v3-turbo-4bit`) |
| `language` | ❌ | Language code (`en`, `de`, etc.). Auto-detect if omitted. |
| `prompt` | ❌ | Optional context to guide transcription |
| `response_format` | ❌ | `json` (default), `text`, `verbose_json` |
| `temperature` | ❌ | Sampling temperature (default: 0.0); for Whisper the only decoding temperature, see [Audio Support](#audio-support) |

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
| Pure transcription | `/v1/audio/transcriptions` | STT (Whisper, VibeVoice) | File upload |
| Chat with audio context | `/v1/chat/completions` | Multimodal (Gemma-3n) | Base64 JSON |
| Long audio (>30s) | `/v1/audio/transcriptions` | STT (Whisper) | File upload |

**Client Implementation Notes:**
- Use `multipart/form-data` content type (not `application/json`)
- File field name must be `file`
- Maximum file size: 50 MB — see [Limits Summary](#limits-summary) for what that means per format

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

- `model` — identifies the model served by the embedding backend. Embedders are not listed by
  `serve`'s `/v1/models`; read the served identity from an embeddings response, or directly from the
  embedding backend's `/health`.
- `system_fingerprint` — the **realization token** `hash.device`: the change-detection signal. It flips
  when the backend's model, its revision/quant, or its device (`--cpu` vs GPU) changes — the three
  things that change the vector space. Known issue ([#81](https://github.com/mzau/mlx-knife/issues/81)):
  for some workspace models the workspace's path changes it too, and the token does not flip.
  Additive mlxk field (standard on OpenAI chat/completions).

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

A text-only model rejects images or audio in the last user message with **422**
`capability_not_supported`. Media in earlier messages remain permitted as conversation history and
are replaced with textual placeholders.

When switching from Vision or Audio to Text model mid-conversation:

1. **Client:** Continue sending full message list (media payloads can be stripped if mapping tables exist)
2. **Server:** Replaces media still present in earlier messages with placeholders for a text model
3. **Result:** Text model sees `[n image(s) were attached]` or `[n audio(s) were attached]` where the
   media were; stripped payloads leave only the text. Either way, what an image showed reaches the
   text model through the vision model's answer in the history

**Example workflow:**
```
1. Vision model: User sends 2 images → Model describes both
2. Switch to Text model: User asks "What's different?"
3. Text model: Receives the conversation with "[2 image(s) were attached]" and both descriptions, compares the descriptions
```

**Storage optimization:** After the first Vision request, clients can drop Base64 payloads from history while preserving assistant responses with `<!-- mlxk:filenames -->` markers. The server reconstructs image IDs from these markers.

---

## Changelog

- **2026-09-30:** 2.0.8 stable — everything below is stated against released 2.0.7.

  **Security**
  - A checkpoint whose `config.json` declares `model_file` is refused before any backend is called (CVE-2026-5843): **501** `not_implemented`, and `/v1/models` does not list it. mlx-lm imports and executes that file; the pinned release does so unconditionally.
  - `serve` no longer imports Python modules from the directory it is started in — a `mlxk2/`, `fastapi.py` or `uvicorn/` there ran instead of the installed package. Its child processes no longer do either.

  **New**
  - `stream_options.include_usage` — a stream that asks for it ends with a usage chunk (`choices: []`) before `[DONE]`; without it the stream is unchanged.
  - `loaded` on every `GET /v1/models` row — `true` on the model in memory.
  - `finish_reason: "length"` when the generation budget cut the answer.
  - **400** `context_length_exceeded` — the prompt fills the window, rejected before any token; `detail` carries `prompt_tokens` and `context_length`. A status even on `stream: true`.
  - **404** `not_found` and **405** `method_not_allowed`: an unmatched path and a wrong method carry the error envelope instead of `{"detail": …}`.

  **Changed**
  - `GET /health` answers `{"status": "ok"}` where it said `healthy`; the status code is the answer. `healthy` stays the CLI's word for a model's file integrity.
  - Default text `max_tokens` is `min(32768, context_length − prompt tokens)`; an explicit value is clamped to the window too. `max_tokens` below 1 → **400** `validation_error`.
  - `/v1/models` `context_length` is `null` when no window is known for the model (was a hard-coded `4096`).
  - A failed stream carries a top-level `error` object, keeps `finish_reason: null`, and ends.
  - A transcription through `/v1/chat/completions` reports `finish_reason: null` in batch and stream; the batch answer claimed `"stop"`.
  - Requests naming different models must not overlap: a request whose model is unloaded before it has generated fails — with **500** `internal_error` before its response starts, with the terminal `error` event after.
  - `mlxk serve` takes one teardown path for Ctrl-C, `SIGTERM` and `SIGHUP`, and stops itself if its supervisor dies. Exit `143` on signal, `137` when forced.
  - Python 3.11–3.14 (was 3.10–3.12).
  - Dep-wave: `mlx-vlm==0.6.10`, `mlx-audio==0.4.8`, `transformers==5.14.1`, `mlx>=0.30.0,<0.32.1`; `torch`/`torchvision` dropped as base deps (524 MB smaller install).

  **Fixed**
  - Images or audio in the last user message for a text-only model are rejected with **422** `capability_not_supported`; the request was answered with the media dropped.
  - **413** and **422** carry `payload_too_large` / `capability_not_supported`; both reported `internal_error`, so a deliberate reject looked like a server fault.
  - `/v1/models` lists vision models it withheld — a check rejected every checkpoint carrying `temporal_patch_size` under transformers 5.x, and those models load and answer correctly.
  - A vision model outside the type whitelist (`qwen2_5_vl`, `qwen3_5`) is served by the vision backend; the server's own probe called it text-only, so an image request was answered without the image. The server now decides with the detector behind `mlxk list`.
  - An empty `model` selects no model: it is answered **404** `model_not_found` instead of by an arbitrary one.
  - Streaming is no longer many times slower than the same generation unstreamed.
  - `stop` cuts a batch answer at the first matching sequence and reports `finish_reason: "stop"`; it was accepted and ignored on every batch surface.
  - `usage` on text and vision responses carries the runner's token counts instead of a word-count estimate.
  - A chat request against an audio model decodes at `temperature 0.0` unless it sends one; the request model defaulted to `0.7`.
  - A chat request's `max_tokens` reaches a transcription model; it was dropped, so a model that transcribes in one pass always stopped at its own default. The `/v1/audio/*` endpoints still take no budget.
  - `GET /health` and `GET /v1/models` answer while the server works. A non-streaming generation, a model load, a vision answer or a transcription silenced both for its whole duration ([#64](https://github.com/mzau/mlx-knife/issues/64)), and a model listing held up `GET /health`.
  - A request naming the loaded model by its listed `id` no longer loads it again when the model had been loaded under another spelling.
  - A transcription writes no `transcript.txt` into the server's start directory, where it overwrote an existing file.
  - `serve --max-tokens` and `MLXK2_MAX_TOKENS` reach the requests; the ceiling was set on a module copy that never answered.
  - A feature gate set to `0` keeps the feature shut; any non-empty value opened it.
  - `mlxk serve --json` prints one JSON document when an option is rejected, not two.

  **Documented**
  - What a `200` from `GET /health` does not tell; one model operation at a time; a stream fails when another model is requested; a batch request runs on after its client has gone; closing a streaming connection stops the generation at the stream's next step — one that arrives as one event runs to the end — and a stream that stopped early writes no completion line.
  - A request to a vision model whose last user message carries neither images nor audio is answered from the conversation's text; with images there, only that message reaches the model. Every chat request is formatted with the model's chat template.
  - Corrected: earlier handbooks listed `unsupported_multimodal` as a server error (501); no release has sent it. The verified-list check it names belongs to `mlxk convert --quantize`; the server has no such check, and its 501 is `not_implemented`. A client branch for `unsupported_multimodal` can be dropped.
  - Corrected: earlier handbooks named Voxtral as a transcription model where VibeVoice belonged: VibeVoice has transcribed since 2.0.5, and no stable release has run Voxtral.
  - Before/after per change, and what clients must update: *From 2.0.7 → 2.0.8* in the Migration Guide.

- **2026-07-24:** 2.0.7 stable — embeddings + audio translation, embeddings model identity
  - **NEW:** the `/v1/embeddings` response (and `embed-serve` `/health`) carries
    `system_fingerprint` = `hash.device` — a change-detection token so a RAG client detects a
    model/revision/device swap; `model` stays the clean `org/name` selector.
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
  - ADR-023: `mlxk convert --quantize` rejects multimodal types outside the verified list (CLI; the server is unchanged).
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

