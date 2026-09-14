# ADR-029: Server State Vocabulary and Generation Controllability

**Status:** ACCEPTED 2026-09-14 — Decision 1 implemented, published with the `docs/SERVER-HANDBOOK.md`
revision that adopts it (see [Implementation](#implementation)). Decision 2: Stage 2 built; Stages 1
and 3 carry their own conditions.
The two Decision blocks below are one unit: the words are only admissible if the
mechanism can answer them, and the mechanism is only meaningful once the words exist.
**Created:** 2026-09-13
**When:** [#64](https://github.com/mzau/mlx-knife/issues/64) was the pull. The vocabulary came
*before* any surface word changed, because the words decide what an endpoint is allowed to claim.
**Related:** ADR-003 (Server/Run port — the origin of the constant this ADR replaces),
ADR-004 (error taxonomy — the project's other vocabulary decision), ADR-009 (stop-token
detection / generation exits), ADR-011 (E2E live test architecture — the consumer that
owns its timeout), ADR-015 (Embeddings API §Model Identity — `model` + `system_fingerprint`),
ADR-016 (Memory-Aware Model Loading — the single-slot model cache),
ADR-023 (Text-First + Verified — *runnable is a prediction, not a certificate*),
[#64](https://github.com/mzau/mlx-knife/issues/64), [#65](https://github.com/mzau/mlx-knife/issues/65).
**Affects:** `mlxk2/core/server_base.py`, `mlxk2/core/server/inference.py`,
`mlxk2/core/server/handlers/chat.py`, `mlxk2/core/server/handlers/models.py`,
`mlxk2/core/server/model_manager.py`, `mlxk2/core/runner/__init__.py`, `mlxk2/operations/serve.py`,
`examples/rag-server/rag-server.py`, `docs/SERVER-HANDBOOK.md` (§GET /health, §GET /v1/models,
endpoint table, §Concurrent Requests, Migration Guide).

---

## Context

The Context describes the tree this ADR was written against, 2026-09-13. What changed since is
recorded under [Implementation](#implementation).

### One word, three meanings

`health` currently names three unrelated things. Two of them live on the wire.

| Surface | Word | What it actually states |
|---|---|---|
| CLI `mlxk health`, `list --json`, `show --json` | `healthy` / `unhealthy` | **File integrity of an artifact on disk.** Required files present, shards complete, no LFS pointers or partial markers, modality-specific auxiliary assets. `mlxk2/operations/health.py` says so in its own docstring: *"This is INTEGRITY only — does NOT check runtime compatibility."* |
| `serve` `GET /health` | `{"status": "healthy", ...}` | A constant. `mlxk2/core/server_base.py` returns it without reading any state. |
| `embed-serve` `GET /health` | `{"status": "ok", "model", "system_fingerprint"}` | `503` until the model is loaded, then readiness **plus identity** (ADR-015 §Model Identity). |

The collision exists at exactly one place on the wire: `serve` uses `healthy`, the CLI's
integrity word, as a process literal. `embed-serve`, written later, already avoids it.

The constant was not a decision. It entered with the 1.x → 2.0 port (ADR-003) as parity
scaffolding; its contract text — *"Liveness only"* in `docs/SERVER-HANDBOOK.md` — was
written eleven months later and describes what was found, not what was chosen.

### The asymmetry between the two servers is structural, not arbitrary

- **`embed-serve` is a one-model server.** The model is resolved, routed and loaded in
  `lifespan` and held for the process lifetime; if it cannot load, the process does not
  start. "Ready" is therefore a well-defined binary process state, and "which model" is a
  constant. Both questions are *answerable*.
- **`serve` is a multi-model server with lazy loading.** `ModelManager` loads per request.
  There is no "the model". An optional `--model` preload also runs in `lifespan`, and a
  failed preload aborts startup.

So `embed-serve` does not have a richer health endpoint. It has a shape in which the
question can be asked at all.

### Measured: the constant is wrong in both directions

Measured 2026-09-13, `mlx-lm==0.31.3`, `uvicorn==0.52.4`, Apple Silicon, GPU idle
(`Device Utilization %` = 0) before the run. `mlx-community/Qwen2.5-0.5B-Instruct-4bit`,
preloaded; `GET /health` polled every 200 ms from a second thread with a 3 s timeout,
while one chat completion was in flight.

| Path | Generation | `/health` during generation |
|---|---|---|
| non-streaming | 9.1 s | 3 polls → **2 × ReadTimeout**; the single answer took **2.548 s** |
| streaming | 9.0 s | 40 polls → **40 × 200**, latency 16–86 ms |

The cause is in the code, not in the load:

- `create_chat_completion` is `async def`; the non-streaming path calls
  `runner.generate_batch(...)` **synchronously** (`handlers/chat.py`). The event loop is
  blocked for the whole generation.
- `get_or_load_model(...)` is likewise called synchronously inside the same coroutine, so
  a request-triggered cold load blocks the loop too.
- The streaming path yields between chunks and stays responsive.

Consequences, stated plainly:

- **Non-streaming:** the process is alive and working, and `/health` runs into the
  supervisor's timeout. A supervisor reads "dead" and restarts a healthy server.
- **Streaming:** the stream can be stalled while `/health` answers `healthy` in 20 ms.
  This is the failure mode named in #64, and it holds **only** on this path.

The same constant therefore produces both a false negative and a false positive, on two
different paths, for two different reasons.

### The generation loop is bounded in tokens and unbounded in time

`MLXRunner.generate_batch` / `generate_streaming` (`mlxk2/core/runner/__init__.py`) exit on
four conditions: EOS id, the `max_tokens` budget, the `_interrupted` flag, or generator
exhaustion. `_end_generation` names exactly these four outcomes, the last as `None`
("generator ended early").

Nothing in the loop reads a clock. A token budget is not a time budget: the same 1000
tokens are seconds on a small model and minutes on a large one under memory pressure.

Per-request observability is absent:

- `last_finish_reason`, `last_completion_tokens`, `last_max_tokens` are written when
  generation **ends**.
- They live on the runner, not on the request. A second generation on the same runner
  overwrites them — the docstring of `_end_generation` already says so.

Per-request control is a single flag:

- `request_interrupt()` sets `_interrupted`, checked once per token.
- On the non-streaming path nobody calls it on client disconnect: `generate_batch` holds
  no reference to the request or the connection. A client whose timeout has fired leaves
  the GPU running until the token budget is spent.
- `shutdown_event` is passed to the streaming path only; `generate_batch` does not receive it.

### The actuator inventory

| Actuator | Sampled | Reach | Who operates it today |
|---|---|---|---|
| `_interrupted` | once per decode step | loop exits, `finish_reason = interrupted`, state defined | CLI SIGINT and the shutdown path only |
| `shutdown_event` | per chunk | as above | streaming path only |
| SIGTERM to the child | when the event loop next runs | uvicorn shutdown, lifespan cleanup | supervisor, on an external signal |
| SIGKILL after 5 s grace | unconditional | process gone | supervisor, only after SIGTERM |

`mlxk2/operations/serve.py::_run_supervised_uvicorn` already implements the coarse
actuator completely — SIGTERM to the child's process group, five seconds of grace, then
SIGKILL, plus a pipe by which the child notices the parent's death. **It has no internal
trigger.** It reacts to signals from outside and never restarts. The coarse actuator
exists; what it lacks is a controller.

Note that SIGTERM is not an actuator on the non-streaming path at all: the signal is only
processed once the event loop runs again, which is after the generation one wanted to stop.

### The observable boundary is narrower than it first appears

A decode step is one `next()` on `mlx_lm.generate.generate_step`: forward pass plus
`mx.eval`, i.e. a Metal command buffer. There is no sampling point inside it, and MLX
exposes no cancellation for an in-flight `eval`.

But the prompt phase is **not** a black box. Under the pinned `mlx-lm==0.31.3`,
`generate_step` accepts `prompt_progress_callback: Callable[[int, int], None]` and calls it
per prefill chunk (`prefill_step_size=2048`). mlxk does not use it (no occurrence in
`mlxk2/`).

The genuinely blind unit is therefore:

- **one prefill chunk** (2048 tokens), or
- **one decode step** (one token).

Nothing coarser.

### Even an error does not restore a defined state

Upstream [ml-explore/mlx#3675](https://github.com/ml-explore/mlx/pull/3675) describes state
corruption when a primitive throws during `eval`: the stream epilogue is skipped, leaving
half-finished streams — no crash, but wrong numbers or a hang. This is the condition under
which a process that has hit a Metal fault does not recover, and it is why the process
boundary is the only complete remedy.

---

## Terminology: observability and controllability

Two terms from control theory, used here with their exact meanings
([Kalman 1960](#kalman-1960), [Kalman 1963](#kalman-1963)):

- **Controllability.** A system is controllable if an available input can transfer it from
  any given state to a desired state in **finite time**.
- **Observability.** A system is observable if its internal state can be **uniquely
  reconstructed** from the measured output over a finite interval.

**This is an analogy, and it is the claim we hold ourselves to.** The terrain is complex and
nonlinear — a stochastic sampling process over a nonlinear model, run by a scheduler and a
GPU driver we do not own, on hardware whose internal state we cannot read. There is no
well-defined state vector, so the formal criteria are not merely inapplicable: they cannot
be written down. No theorem transfers, and nothing in this ADR is a proof. What transfers is
the standard. The terms name what *complete* would mean, and thereby turn every gap we
accept into a **stated** gap instead of an unnoticed one — the discipline this project
applies elsewhere, where an intended divergence carries a written reason and an unnoticed
one carries none. Without the yardstick, "we cannot stop a wedged generation" is an
anecdote; with it, it is a located, bounded deficiency with a known remedy and a known price.

The rest of this document uses the terms directly, and every claim names its own evidence.

| Control-theory term | Here |
|---|---|
| State | where a generation sits: *loading* · *prefill* · *decoding* · *finished* |
| Input / actuator | `_interrupted`, SIGTERM, SIGKILL |
| Output / measurement | phase marks and heartbeats — **none exist today** |
| Sampling instant | once per decode step; once per prefill chunk, if the upstream hook is used |

> **The system is not completely controllable.** There is a region — an in-flight
> `mx.eval` — from which no available input returns it to a defined state in finite time.
> SIGKILL does reach a defined state in finite time, but by removing the plant rather than
> steering it: replacement, not control.

Evidence: `mlxk2/core/runner/__init__.py` samples `_interrupted` once per decode step and
nowhere else, and MLX exposes no cancellation for an in-flight `eval`.

**Observability has the same boundary**, sampled at the same point in the loop. That is an
observed property of this implementation, not a consequence of the duality between the two
concepts — which does not transfer here.

The operating-system guarantee that makes replacement viable is reclamation, not
cancellation: the GPU context belongs to the kernel driver, so process death causes the
driver to tear down command queue, buffers and address space. An already-submitted command
buffer still drains or is ended by the driver watchdog. This is the strongest guarantee
available to a user-space process, and it is the reason the process boundary — and only the
process boundary — is a complete actuator.

---

## Decision 1 — Vocabulary

Four questions, four words, **no word in two roles**.

| Question | Word | Where it lives | Who asks |
|---|---|---|---|
| Is the process alive and *able* to answer? | **live** | process level | supervisor |
| Can it accept a request *now*? | **ready** | process level | scheduler, load balancer |
| Is *this model* loaded? | **loaded** (`warm` / `cold`) | one row of `GET /v1/models` | client, scheduler |
| Is the **artifact on disk** complete? | **healthy** | CLI only (`mlxk health`, `--json`) | human, tooling |

This is the KServe v2 / Triton split — server liveness, server readiness, per-model
readiness — under our own names. The vocabulary is aligned with the field rather than
invented for it ([prior art](#prior-art)).

**The reserved word.** `healthy` denotes an artifact and does not appear on the HTTP
surface. This is the rule the rest of the vocabulary exists to protect. It is our rule, not the
field's: NVIDIA Dynamo's frontend and mlx-vlm say `healthy` on `/health`, llama.cpp and mlx-lm
say `ok`, vLLM, SGLang and Triton send no word at all — and a probe reads none of them, only the
status code. Giving the word up therefore costs no client anything.

**Readiness is per model, not per server.** `GET /v1/models` already filters on
`healthy ∧ runtime_compatible` (`handlers/models.py`), so membership in that list is the
*runnable* prediction. The new field is the orthogonal fact:

- **membership = `runnable`** — a prediction, only as honest as `runtime_compatible`
  (ADR-023: a prediction, not a certificate).
- **`loaded` = a fact** — verifiable at the moment of asking.

The two must never merge into one field. `ModelManager` already holds the fact
(`current_model`) and its cache is single-slot — any request for a different model calls
`_cleanup_previous_model()` — so `loaded: true` applies to one model at most.

> **The trap this placement had, closed.** The cache was keyed by the *request spelling*, not
> the resolved name. A client asking for `qwen` stored the warm entry under `qwen` while the list
> advertised the resolved id, so a naive comparison would report `cold` for a warm model — and a
> request naming the listed id did not merely look cold, it loaded the model a second time
> (measured). The cache is now keyed by the model directory a spec reaches, as the filesystem
> identifies it: a name compared as a string would still miss another case of it on a
> case-insensitive volume.

**`embed-serve` is the boundary case n = 1.** With a single model, `ready` and `loaded`
coincide. Its port opens only once the model has loaded — uvicorn runs the lifespan before it
binds — so every answer comes from a loaded model; the `503` branch in its handler is not
observable over the socket (measured 2026-09-14: a probe from process start sees refused
connections, then `200`). The `model` + `system_fingerprint` fields it returns are **identity**,
not health, and remain governed by ADR-015 §Model Identity — this ADR does not touch them.

**A fifth word, deliberately not yet answerable:**

| Question | Word | Status |
|---|---|---|
| Is something still arriving, and since when? | **progressing** | **not expressible today** — no clock is held |

`progressing` is admitted into the vocabulary and marked unanswerable. This is the honest
form: the word is right independently of whether the mechanism exists, and the contract
says which words the server can currently say.

---

## Decision 2 — Observability and controllability

### Phase marks

A generation carries four monotonic marks (`time.monotonic()`, never wall clock — NTP steps
and sleep/wake must not distort stall detection), held **on the request, not on the runner**:

`t_accepted` → `t_model_loaded` → `t_first_token` (prefill complete) → `t_last_token`

Each mark names *which* unobservable region the request currently sits in. This is the
point of the marks, and it is worth more than a single stall timer:

> "No token for 90 s" is normal during prefill on a long prompt and means *wedged* during
> decoding. A single threshold without a phase is either too tight for prefill or too loose
> for decoding.

`t_last_token` and "the moment the current `eval` started" are numerically the same during
decoding — they differ by the loop body, i.e. microseconds. They differ at the **edges**,
which is exactly where the phase marks earn their place.

### Heartbeat coverage

- One heartbeat per decode step (the existing loop).
- One heartbeat per prefill chunk, via the upstream `prompt_progress_callback`.

Together these cover the entire generation. Blindness remains only *inside* one such step —
the same place where the actuator does not reach.

### Three stages, each honest on its own

Conditions, not schedules. Each stage is a precondition for the words above, not for the
next stage.

**Stage 1 — the clock.** Phase marks plus a request-bound generation state. No
architectural change, no new failure modes. Makes `progressing` expressible and separates
*working* from *wedged*, which today are the same silence. Carries the disconnect path with
it, because the batch path gains a request-bound structure either way.
*Due when:* `progressing` is to be claimed anywhere, or a consumer's timeout must leave the
server in a defined state.
The same request-bound structure closes two windows measured 2026-09-14 and recorded in the
handbook as the server's behaviour: a request for another model, served between two steps of a
stream, unloads the runner and the stream fails; and a non-streaming request keeps generating
after its client has gone, while the requests behind it wait. The field treats both as defects —
vLLM, SGLang, llama.cpp and Ollama stop a generation whose client left, and llama.cpp, Ollama,
LocalAI and mlx-lm's own server let a running generation finish before loading another model.
For a one-user, one-client server both are rare, which is why they are recorded rather than
fixed. ⚠ Starlette's `BaseHTTPMiddleware`, which `serve` uses, makes `request.is_disconnected()`
always false; a disconnect has to be read from `receive()`.

**Stage 2 — every model touch off the event loop. Built 2026-09-14.** Not a pool: an `mx.array`
carries the stream it was made on, so a model loaded on the main thread raises *"There is no
Stream(gpu, N) in current thread"* when it is generated with elsewhere — for some checkpoints and
not others, which is how a pool would fail: intermittently. One worker (`max_workers=1`) owns
loading, batch and streaming generation, vision chunks, transcription, the startup preload and the
shutdown cleanup, and serialization follows from that shape rather than from a lock. The
precedents first cited here — `asyncio.to_thread` for vision chunks and `embed-serve`'s threadpool —
are pools; applied to generation, they would have carried exactly that failure.
*Due when:* any statement about `live` is to be made contractually. Without this stage such
a statement is untrue under load; the measurement above is the proof.

**Stage 3 — a controller on the existing actuator.** The supervisor can already shut down
completely; what is missing is the internal condition that triggers it and the decision
whether it restarts afterwards. This is the only complete control for the Metal region, and
it has a real price: it takes the other requests with it and costs a model reload. That is
precisely liveness semantics in the Kubernetes sense, and it is an owner decision, not a
technical one.
*Due when:* the Stage-1 clock exists (a stall raises nothing, so there is no trigger without
it) and a stall has been observed that Stage 1 cannot resolve. A fault does raise, but not
specifically — see #65 under [Consequences](#consequences).

> Stage 3 is deliberately coarse. The pathological case — a genuine GPU hang — cannot be
> induced on demand and therefore cannot be tested. A mechanism finer than the process
> boundary would *claim* to handle a case we cannot exercise.

### The consumer owns the timeout

The timeout belongs to the consumer, not the server: a test, a gateway, a client knows what
it is willing to wait for. Two halves follow, and today only the first exists:

- **(a) the consumer decides to terminate.** Already real — an HTTP timeout *is* that
  decision. Staging those timeouts by model size is the correct shape of (a).
- **(b) the termination is controllable.** Absent below the token boundary. Today the
  decision does not even arrive: a disconnect is invisible to `generate_batch`.

(a) without (b) makes clients green without making the server controllable.

---

## What is decided here, and what is not

**Decided: `healthy` leaves the HTTP surface.** That is what the reserved word in Decision 1
means, so `serve`'s literal changed. It was not a separate question to be taken later.

**Decided with the publication: the status code is the answer, and the body keeps a word.**
`GET /health` stays one path and answers `200` with `{"status": "ok", "service":
"mlx-knife-server-2.0"}`; every `GET /v1/models` row carries `loaded: true | false`. The reasoning
is about consumers, not resemblance:

- **Where the field agrees, conform completely.** vLLM, SGLang and NVIDIA Dynamo — the projects
  that work on status monitoring most actively, and that coordinate with each other — all answer
  readiness on `GET /health` with `200` / `503`, and Kubernetes reads nothing but that code. `serve`
  already had the path and the code.
- **Where it does not, the body is ours.** Those same servers disagree on the body — empty in vLLM
  and SGLang, `healthy` in Dynamo — so no body word is compatible with them as a group, and no probe
  reads one. `ok` was chosen because `embed-serve` already says it: both mlx-knife servers answer
  alike, the key `status` and the identity string `service` stay, and a client migrates one literal.
- **A separate `ready` path was not added.** On `serve` it would never answer differently from
  `live` (see [Rejected alternatives](#rejected-alternatives)).
- **`loaded` is our own field.** None of the three loads models on demand, so none has a per-row
  load state. The one server that does — llama.cpp's router, `status.value` — pairs it with load and
  unload endpoints `serve` does not have; borrowing the field alone would serve no client written
  for it.

Where the field's actual shared interface lies — Prometheus metrics for queue depth, running
requests and KV-cache use, standardized by the Kubernetes Gateway API Inference Extension — is out
of this ADR's scope ([prior art](#prior-art)).

**Coupled, deliberately.** This ADR commits with the `SERVER-HANDBOOK` revision that adopts the
vocabulary, and that revision states the state of the tree. **Publication and a conforming
implementation are therefore one act** — the handbook cannot teach a word the server does not say.
Which side sets the scope is a genuine choice: either the handbook is written first and the
implementation is cut to match, or the implementation's reach decides how much the handbook may
claim. Both are legitimate. Publishing the rule while shipping a server that breaks it is not.

Recorded so the cost is known: the literal was **asserted** by the live test
`test_server_e2e.py`, which migrated with the change, and merely **emitted** by
`examples/rag-server/rag-server.py`, which read only the status code. That example also re-emitted
`"service": "mlx-knife-server-2.0"` as its own identity, which it is not, and reported a backend
that did not answer within five seconds as `unreachable`; it now names itself and tells a timeout
from a refused connection.

---

## Rejected alternatives

**Mirror `embed-serve`: `503` until loaded, on `serve`.** Rejected. `serve` has no single
model to gate on. Without a preload there is nothing to wait for; with one, the port is
already gated — under uvicorn 0.52.4, `Server.startup()` runs the lifespan before
`loop.create_server`, so the socket binds only after the preload has completed. The shape
would add a status code without adding a statement.

**Enrich `/health` with fields first (current model, free memory).** Rejected as a first
step. Adding fields to a response that reads no state makes the answer richer, not truer —
and on the non-streaming path it makes no answer at all. Honesty precedes enrichment.

**Split this ADR into vocabulary and mechanism.** Rejected. The two share a boundary:
`progressing` is inadmissible without the phase marks, and `live` is dishonest without
Stage 2. Two documents would re-open exactly the gap this ADR closes.

**Prove readiness by generating, as SGLang's `/health_generate` does.** Rejected **as a
periodic probe**; kept as a manual diagnostic. It is the only non-predictive answer to *"is
the pipeline usable?"*, and on a multi-slot server it is cheap. Here it is not:

- `ModelManager` holds a **single** slot. A probe naming any model other than the warm one
  calls `_cleanup_previous_model()` and evicts it — a periodic probe would thrash the very
  model it is meant to certify.
- It would take a turn on the single model thread, so a real request waits behind the probe.
- It cannot be a server-level probe at all: `serve` has no single model to generate with, so
  it would be a per-model operation — Decision 1's `loaded` row, with a side effect.

SGLang's own separation of `/health` from `/health_generate` is currently not enforced (both
generate), which is evidence that the distinction is hard to hold even when it is designed
in from the start. NVIDIA Dynamo runs the refined form: a canary request only after ten seconds
without successful activity, its timer reset by every request and every streamed chunk. It still
needs a model to name, which is the part `serve` cannot supply.

---

## Consequences

- **A word is given up.** `healthy` leaves the HTTP surface and belongs to the artifact
  alone. Everything that reads it changes with it, including our own example.
- **Readiness moves to the model list.** It becomes a per-row fact instead of a process
  claim, which is where it is true. Any plan that placed warmth on `/health` is superseded
  by this placement.
- **`GET /v1/models` inherits Stage 2.** The `loaded` field is honest, and it is reachable
  while the server works — once the listing itself left the loop too (see
  [Implementation](#implementation)).
- **The contract states what it can say.** `docs/SERVER-HANDBOOK.md` carries the vocabulary
  with a column recording which words the server can currently answer, so no endpoint
  promises more than the tree holds. §Concurrent Requests, whose claim *"Sequential processing
  (one request at a time)"* rested on the blocked event loop, now states what the single model
  thread guarantees — one model operation at a time — and what it does not.
- **#65 is not resolved here, and waits on no upstream release.** A backend fault does arrive
  as an exception, but MLX reports every command-buffer failure — the timeout of #65 as well
  as out-of-memory — with the same message prefix, so the exception alone does not identify a
  failed backend. The phase marks make its symptom observable, not its cause. When a fix can
  tell a failed backend, the field's precedent is a `503` on `GET /health` (vLLM does so for a
  dead engine).

---

## Implementation

Published 2026-09-14 with the handbook revision; `mlx-lm==0.31.3`, `uvicorn==0.52.4`, GPU idle
during every measurement.

- **Wire form.** `GET /health` answers `200` with `{"status": "ok", "service":
  "mlx-knife-server-2.0"}`; every `GET /v1/models` row carries `loaded`, `true` on the model in
  memory. The reasoning is under *What is decided here*.
- **Stage 2** runs every model touch on one worker thread. `GET /health` answered within 1.2–4.3 ms
  during a non-streaming and a streaming generation, a 4.7 GB cold load with a vision answer, and a
  3.1 GB cold load with a 220-second transcription. The live test
  `test_health_answers_while_generating` fails against the tree before Stage 2 and passes after.
- **The trap is closed.** `ModelManager` keys its cache by the model directory a spec reaches — its
  device and inode — and records which kind of runner it holds; `GET /v1/models` marks the row whose
  directory matches. Before, a server started with `--model Qwen2.5-0.5B` loaded the model again for
  a request naming its listed id, and again for the original spelling after that; so did a workspace
  preloaded by path and requested by its listed name. Both are now served from memory.
- **The listing left the loop.** It reads every model directory — about half a second for 98
  models — and `GET /health` waited for it: 539 ms at worst. It runs on a helper thread, not the
  model thread, which it would queue behind; the worst probe during a listing is now 17 ms. The
  runtime check it calls saves and restores process-wide logger levels, now under a lock, since two
  listings can run at once.
- **Still on the loop: reading a request.** Parsing the body and decoding its images runs before any
  model work; a vision request carrying 38 MB of images held `GET /health` up for 401 ms at most.
  Below a one-second probe timeout, so recorded rather than moved.
- **Recorded, not fixed:** the two windows under Stage 1.

---

## References

<a id="kalman-1960"></a>
**Kalman 1960** — R. E. Kalman, *On the general theory of control systems*, Proceedings of
the 1st IFAC Congress, Moscow 1960; Butterworths, London 1961, pp. 481–492. Where
controllability and observability are introduced.

<a id="kalman-1963"></a>
**Kalman 1963** — R. E. Kalman, *Mathematical description of linear dynamical systems*,
Journal of the Society for Industrial and Applied Mathematics, Series A: Control, vol. 1,
no. 2 (1963), pp. 152–192, DOI [10.1137/0301010](https://doi.org/10.1137/0301010). The
precise definitions and the canonical decomposition into controllable and observable
subspaces.

<a id="prior-art"></a>
**Prior art** — [`docs/ADR/appendix/ADR-029-prior-art.md`](appendix/ADR-029-prior-art.md):
how other inference servers express server and model state (KServe v2 / Triton, SGLang,
llama.cpp, vLLM, Ollama, mlx-vlm, mlx-lm). Evidence for Decision 1; it holds no decision, and
this ADR stands without it.
