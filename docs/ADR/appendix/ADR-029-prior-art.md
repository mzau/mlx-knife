# ADR-029 Appendix: Prior Art — how other inference servers express server and model state

**Purpose.** Evidence for
[ADR-029](../ADR-029-Server-State-Vocabulary-and-Generation-Controllability.md) Decision 1.
This appendix holds no decision; the ADR body stands without it. Read it to check whether the vocabulary chosen there is aligned with the
field or diverges from it.

**Surveyed 2026-09-13, extended 2026-09-14** with NVIDIA Dynamo, llama.cpp's router mode, LM Studio,
the health bodies, and which projects actually work on status monitoring. Every entry is
labelled with how it was established:

- **[source]** — read from the version installed in this tree, or from upstream at the commit named.
- **[docs]** — read from the project's own documentation.
- **[issue]** — an open defect or feature request in the project's tracker.

Third-party behaviour changes; re-check before relying on a row.

---

## Overview

| Server | Liveness | Readiness | Per-model state | Busy / progress |
|---|---|---|---|---|
| **KServe v2 / Triton** | `/v2/health/live` | `/v2/health/ready` | `/v2/models/{name}/ready` | — (separate metrics) |
| **NVIDIA Dynamo** | `/live` → `{"status": "live"}` | `/health` → `200` `healthy` / `503` `not_ready` | — (a model is listed once a worker serves it) | canary requests; `/metrics` |
| **SGLang** | `/liveness` | `/health` (503 while warming) | — | `/health_generate` proves the pipeline |
| **llama.cpp** | `/health` 200 | `/health` 503 `Loading model` | router mode: `status.value` on each `/v1/models` row | `/slots`, `/props` |
| **vLLM** | `/health` | **none** — `/health` answers once the port binds; `503` once the engine is dead | — | `/metrics` (Prometheus) |
| **Ollama** | — | — | `/api/ps` (loaded, `size_vram`, `expires_at`) | — |
| **LM Studio** | — | — | `state` on each `/api/v0/models` row | — |
| **mlx-vlm** | always `200` | not expressed | in the `/health` body | `/metrics` |
| **mlx-lm** | constant `{"status":"ok"}` | not expressed | — | — |

---

## KServe v2 / NVIDIA Triton — the standard **[docs]**

Three levels, and the split is explicitly designed against the Kubernetes probe model:

- `/v2/health/live` — the server process is running; intended directly as `livenessProbe`.
- `/v2/health/ready` — every model loaded at startup is ready; intended as `readinessProbe`.
- `/v2/models/{name}/ready` — readiness of one named model.

The documented distinction: model readiness answers *"did the model load and can it serve?"*,
server readiness/liveness answers *"is the service and its infrastructure running?"*

This is the same three-way split ADR-029 adopts as `live` / `ready` / `loaded`. Note that
Triton expresses all three through **status codes on separate paths**, not through a word in
a body.

Sources: [KServe V2 Protocol](https://kserve.github.io/website/docs/concepts/architecture/data-plane/v2-protocol) ·
[Triton Inference Protocols](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/customization_guide/inference_protocols.html)

## SGLang — four levels, including proof by execution **[docs]** **[issue]**

- `/liveness` — the HTTP process is alive; returns 200 during warmup.
- `/health` — ready for inference; returns 503 during warmup.
- `/health_generate` — **runs an actual generation**, so the answer is a demonstration
  rather than a prediction.

Two open defects are directly instructive for us:

- [#12731](https://github.com/sgl-project/sglang/issues/12731) — `/health` behaves like
  `/health_generate`: both generate. The separation is declared but not enforced.
- [#30770](https://github.com/sgl-project/sglang/issues/30770) — *"Synchronous request
  preprocessing blocks the API event loop and delays health endpoints."* This is the same
  failure mode ADR-029 measures on our non-streaming path, open in a major framework.

## llama.cpp / llama-server — state moved *out* of `/health` **[docs]**

Current documented behaviour:

- `503` with `{"error": {"code": 503, "message": "Loading model", "type": "unavailable_error"}}`
  while loading.
- `200` with `{"status": "ok"}` once ready.

State lives in its own endpoints, not in the health body:

- `/slots` — per slot: `id`, `is_processing`, generation `params`, and `next_token` with
  `has_next_token`, `n_remain`, `n_decoded`.
- `/props` — `total_slots`, `model_path`, `chat_template`, `modalities`, `is_sleeping`,
  `build_info`.

⚠ **Historically `/health` itself carried `slots_idle` / `slots_processing` and a
`fail_on_no_slot` query parameter. Those are gone.** The project moved state out of the
health endpoint into dedicated ones. This is the closest thing to a decided precedent for
ADR-029's rejected alternative *"enrich `/health` with fields"* — and it was decided against
enrichment.

`next_token.n_remain` / `n_decoded` are, in our terms, live per-slot observability of the
decoding phase.

Source: [llama.cpp server README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)

## vLLM — observability solved, readiness not **[docs]** **[issue]**

- `/metrics` is Prometheus-compatible and carries `vllm:num_requests_running`,
  `vllm:num_requests_waiting`, `vllm:num_preemptions_total` and time-to-first-token
  histograms. This is the observability half done properly.
- `/health` returns `200` as soon as the API server process binds its port — **not** when
  the weights finish loading. That is the same defect ADR-029 records for our server.

Open requests to fix it: [#6073](https://github.com/vllm-project/vllm/issues/6073) (add a
separate `/ready`, return `/health` earlier) and
[#36960](https://github.com/vllm-project/vllm/issues/36960) (a `/health/ready` that verifies
the GPU).

Source: [vLLM Production Metrics](https://docs.vllm.ai/en/stable/usage/metrics/)

## Ollama — warmth as its own endpoint **[docs]**

No Kubernetes-style health surface, but `/api/ps` lists the models **currently in memory**
with `size_vram` and `expires_at`. Residency is governed by `keep_alive` (a duration, `0` for
immediate unload, a negative value to pin).

This is ADR-029's `loaded` fact, exposed as a dedicated endpoint rather than as a field on
the model list — and with one dimension we do not have: an expiry time.

Source: [Ollama API](https://github.com/ollama/ollama/blob/main/docs/api.md)

## mlx-vlm — rich body, no status-code semantics **[source]**

Read from `mlx_vlm/server/app.py` at the pinned version. `GET /health` always returns `200`:

```json
{"status": "healthy", "loaded_model": …, "loaded_adapter": …, "loaded_models": …,
 "loaded_context_size": …, "configured_context_limit": …, "effective_context_limit": …,
 "loaded_tool_parser": …, "continuous_batching_enabled": …, "apc_enabled": …}
```

A separate `/metrics` endpoint returns a metrics snapshot plus the same runtime snapshot.

So mlx-vlm answers *which model* and *what limits* — but never *not ready*, because the
status code does not vary. It shares the word `healthy` with NVIDIA Dynamo's frontend (below).

## mlx-lm — a constant **[source]**

Read from `mlx_lm/server.py` at the pinned version: `handle_health_check` writes
`{"status": "ok"}` with `200`, unconditionally, reading no state.

Our own constant is the same shape. It was inherited with the 1.x → 2.0 port, not invented.

## NVIDIA Dynamo — liveness and readiness as separate paths, plus a canary **[docs]**

Dynamo describes itself as *"the orchestration layer above inference engines — it doesn't replace
SGLang, TensorRT-LLM, or vLLM"*. Its *Health Check Reference*:

- Frontend `/live` → `200` `{"message": "Service is live", "status": "live"}`; `503` with
  `shutting_down` once draining has finished.
- Frontend `/health` → `200` `{"status": "healthy", "endpoints": […], "instances": […]}` while
  ready; `503` with `not_ready` and a `stage` as soon as draining starts.
- Workers report `notready` / `ready` on both paths, with a per-endpoint map.
- Optional canary: a minimal inference request through the backend's normal path, sent after ten
  seconds without successful endpoint activity (`DYN_CANARY_WAIT_TIME`), three-second timeout.

Dynamo follows neutral definitions where they exist: its frontend offers *"OpenAI-compatible HTTP
endpoints and KServe gRPC endpoints"*, and its Kubernetes routing *"implements the GAIE Lightweight
Endpoint Picker (LW-EPP) `ext_proc` interface"* of the Gateway API Inference Extension.

Sources: `docs/fern/pages/reference/observability/health-checks.mdx`,
`docs/fern/pages/developer-guide/knowledge-base/modular-components/frontend/overview.md`,
`docs/fern/pages/reference/components/gateway-api-routing.mdx` in
[ai-dynamo/dynamo](https://github.com/ai-dynamo/dynamo), read 2026-09-14.

## vLLM and SGLang — status codes, and what they changed **[source]** **[issue]**

- vLLM's `/health` answers with an empty body and, since
  [#24897](https://github.com/vllm-project/vllm/pull/24897) (merged 2025-09-18), `503` once the
  engine process has died — *"an explicit 503 response allows automated systems like Kubernetes and
  load balancers to make better decisions"*.
- SGLang's engine serves `/health` and `/health_generate` from one handler that generates
  (`python/sglang/srt/entrypoints/http_server.py`, main, read 2026-09-14). `/liveness` and
  `/readiness` belong to its router, `sgl-model-gateway`.

## llama.cpp router mode — load state on the model list **[docs]**

Started without a model, `llama-server` lists every model in its cache on `GET /models` (and
`/v1/models`) and loads on demand. Each row carries `"status": {"value": …}` —
`unloaded`, `loading`, `loaded`, `sleeping` or `downloading` — with `args`, `failed`/`exit_code` or
`progress` where they apply, next to `POST /models/load` and `POST /models/unload`. This is the one
server in the survey that, like `serve`, lists more models than it holds and says which are loaded.

Source: [llama.cpp server README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md),
*GET `/models`*, read 2026-09-14.

## LM Studio — load state as a row field **[docs]**

`GET /api/v0/models` rows carry `"state"`; the documented example shows `"not-loaded"`.

Source: [LM Studio REST endpoints](https://lmstudio.ai/docs/developer/rest/endpoints), read 2026-09-14.

## Who works on status monitoring

Merged pull requests from 2025-09-14 to 2026-09-14 with `health` or `metrics` in the title, counted
through the GitHub search API — a rough measure, checked against the titles:

| Project | `health` | `metrics` |
|---|---|---|
| NVIDIA Dynamo | 40 | 145 |
| SGLang | 44 | 136 |
| vLLM | 5 | 84 |
| llama.cpp | 1 | 7 |
| KServe | 1 | 4 |
| Ollama | 2 | 1 |
| Triton server | 0 | 1 |
| mlx-vlm · mlx-lm | 1 · 1 | 2 · 0 |

Hugging Face TGI is archived (last push 2026-03-21). LiteLLM (96 · 66) is a gateway whose health
checks concern upstream providers, a different subject.

Where these projects meet is **metrics**, not a health body: the Kubernetes Gateway API Inference
Extension's *Model Server Protocol* requires the OpenAI Completions and Chat APIs and, through a
Prometheus endpoint, a queue gauge, a running-requests gauge and KV-cache utilization — mapped per
engine to vLLM's, SGLang's and TensorRT-LLM's own metric names. Dynamo's endpoint picker implements
the same extension's routing interface.

Source: `docs/proposals/003-model-server-protocol/README.md` in
[kubernetes-sigs/gateway-api-inference-extension](https://github.com/kubernetes-sigs/gateway-api-inference-extension),
read 2026-09-14.

---

## What this establishes for ADR-029

Stated as findings, not as decisions — the decisions are in the ADR body.

1. **The three-way split is standard, not novel.** `live` / `ready` / `loaded` is the KServe
   v2 structure. Adopting it aligns the contract rather than diverging from it.
2. **Separate paths beat an enriched health body.** Triton and SGLang separate by design;
   llama.cpp actively removed state from `/health`; vLLM keeps state in `/metrics`. The only
   server that puts state into the health body is mlx-vlm, and there the status code carries
   no meaning.
3. **Reserving `healthy` costs nothing in compatibility — but not because the word is rare.**
   NVIDIA Dynamo's frontend and mlx-vlm say `healthy`, llama.cpp and mlx-lm say `ok`, vLLM, SGLang
   and Triton say nothing. It costs nothing because nothing reads the word (finding 6). *(Corrected
   2026-09-14: this finding first called `healthy` unusual on this surface.)*
4. **Proof-by-execution exists and is not free.** SGLang's `/health_generate` is the only
   non-predictive answer to *"is the pipeline usable?"* — see the ADR body for why a periodic
   probe of that kind is rejected here and where it is still useful.
5. **The blocked-event-loop failure is state of the art, not a local mistake.** SGLang has it
   open; vLLM's readiness gap is the same class. This does not reduce the need to fix it, but
   it locates the problem.
6. **The body is not a compatibility surface; the path and the status code are.** The projects
   most active on status monitoring disagree on the body — empty, `healthy`, `live` — and agree
   that `GET /health` answers `200` or `503`. Kubernetes judges a probe by its status code alone.
7. **A per-row load state belongs to servers that load on demand.** llama.cpp's router and
   LM Studio carry one; Ollama and Triton expose load state on separate endpoints; vLLM, SGLang and
   Dynamo list what is being served and have nothing to add. No field form has a client that would
   work with `serve` for having copied it.
8. **The field coordinates on metrics.** The Gateway API Inference Extension maps vLLM's and
   SGLang's metrics into one protocol, and Dynamo plugs into the same extension as an endpoint
   picker. That interface answers a scheduler's questions — queue, load, cache — which is terrain
   ADR-029 does not enter.
