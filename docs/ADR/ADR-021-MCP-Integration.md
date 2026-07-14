# ADR-021: MCP Integration (Model Context Protocol)

**Status: REJECTED — MCP will not be built into mlx-knife. It is not planned, for any release.**
No `mlxk mcp serve`, no `mlxk2/mcp/`, no `mcp` dependency, no feature gate, no slot on the roadmap. An MCP surface over mlx-knife is a **consumer of `serve`** — a separate tool, buildable by anyone at any time without touching this repo. Issue [#56](https://github.com/mzau/mlx-knife/issues/56), closed as *not planned* (2026-07-14), carries the guidance for whoever builds one; this ADR records why it does not belong here.

**Created:** 2026-02-02 · **Rewritten:** 2026-07-14 — replaces the Options A–D brainstorm (kept as history of thought in the appendix).

---

## Context

The 2026-02 framing treated MCP as the cure for three mlx-knife "limitations" — manual model selection, statelessness, no pipelines — and put intent detection and model selection *inside the server* (Options A–C). That is orchestration in the node, and it contradicts the one discipline: **a node describes and executes itself; it does not orchestrate.**

A 2026-06 direction note corrected this to "Option D": an in-process `mlxk mcp serve` exposing four stateless atoms (`generate`, `vision`, `transcribe`, `embed`). That fixed the orchestration error but left one assumption unexamined — that the surface has to live *inside* mlx-knife.

It does not. And the symptom was visible all along: MCP has been sliding since beta.9 (2.0.6 → 2.0.7-stretch → 2.1 → post-2.1) because it never earned a slot against work that mattered more. A feature that never wins a slot is a feature in the wrong place.

## Decision

**Rejected.** mlx-knife ships no MCP surface. Three reasons.

**1. It is fully expressible on the public contract.** Each proposed tool is an ordinary OpenAI request against `serve`: `generate` → `/v1/chat/completions`; `vision` → the same with an `image_url` (base64 `data:` URL) content part; `transcribe` → `/v1/audio/transcriptions` (and `/v1/audio/translations`, #54); `embed` → `/v1/embeddings`. None needs mlxk2 internals. **A surface that can be built entirely on the public contract does not belong behind it.**

**2. In-core makes the slot economy actively worse.** `serve` holds exactly one model; every model change is a full evict + reload with the manager's lock held throughout. An in-process MCP server would add a *second* single-slot model manager with its own eviction policy — and a `vision` or `transcribe` tool exists precisely to use a *different* model from the host's chat model. Two GB-scale models then contend for one machine, and mlx-knife, not the operator, decides the trade-off. The memory gate cannot arbitrate it: its lever is per-process, its signal machine-wide (ADR-016).

Outside, the same problem is a config question. mlx-knife's node shape is already *one warm model per process, `serve` as the single OpenAI ingress proxying outward* — `serve --embed-backend URL` is exactly that, and the README sells the payoff: one base URL for chat and embeddings, two processes each holding a model warm. A consumer inherits whatever topology the operator runs; an in-process surface hard-wires one.

**3. Hygiene.** The `mcp` SDK never enters the core. No incubation, no gate to remember to remove, no stable-promotion to schedule.

## What this obliges mlx-knife to do

Exactly what it already owes every consumer: **keep `serve`'s OpenAI surface and the `--json` contract stable.** Nothing more.

Working the design through did surface two defects in that contract. Both are mlx-knife's, both exist with or without MCP, and both are now tracked:

- **`/v1/models` carries no capability field** — `{id, object, owned_by, permission, context_length}` and nothing else. No HTTP client can tell a vision model from a text-only one. The data exists, but only on the CLI contract (`mlxk list --json`), which is why the `model-routing` example is a CLI consumer rather than an HTTP one. → **[#51](https://github.com/mzau/mlx-knife/issues/51)**, open since 2026-02.
- **`serve` silently strips media it cannot process.** An image sent to a text-only model is not rejected: it is replaced with a text placeholder and the model answers anyway, HTTP 200. `docs/RUNTIME-FEATURES.md` §3.2 forbids this behaviour by name (*"Hard error. Never silently strip the image and run text-only."*) — the CLI honours it, the server does not. → 2.0.8 honesty bite.

Together these mean an HTTP client today can neither discover what a model can do nor find out when it guessed wrong. That is a hole in the node's self-description, and it is worth fixing for nChat, for broke, and for the `model-routing` example — not only for a hypothetical MCP adapter.

One structural gap is worth naming but not fixing: there is no `--vision-backend`. Warm-multi-model has exactly one shape today, and it covers only embeddings. If that is ever worth generalising, it belongs in **`serve`'s proxy pattern**, not in a model manager hidden behind a subcommand.

## Non-Goals

- **MCP is not a cluster data-plane.** The contract between a cluster router and mlx-knife is `serve` HTTP (data plane) plus SSH/CLI-JSON (maintenance plane). The earlier forward-pointer — *"an MCP host can be a cluster coordinator"* — is **struck**: a host LLM choosing among N node servers per prompt is a routing brain with no warmth accounting, no single ingress and no failover semantics. A consumer may build that; this ADR does not sell it as the cluster path.
- **No orchestration in the node** — no server-side pipelines, no intent detection, no automatic model selection.

## Alternatives considered

- **In-process `mlxk mcp serve` (Option D)** — **rejected**, per reason 2. It would also "solve" the discovery gap by importing `capabilities.py` directly: bypassing the contract instead of fixing it, for exactly one consumer, while nChat, broke, the `model-routing` example and every other OpenAI client still see no capability field.
- **In-tree hybrid** — a `mlxk mcp` subcommand that is internally only a `serve` client. **Deferred, not rejected.** It would buy back the one-install UX (`uv tool install mlx-knife` brings no MCP) at the price of release coupling. The fallback if packaging friction ever turns out to dominate.
- **Options A–C** (intent detection / embedding-based routing / model selection *in the server*) — **rejected**: orchestration in the node. See appendix.
- **Adopt an existing generic OpenAI→MCP bridge** ("boring solution first") — surveyed 2026-07-14: **none exists.** What exists is chat-only and unmaintained, or a workflow suite rather than a thin adapter; the numerous "MCP-Bridge"-style projects solve the *reverse* direction (MCP tools → an OpenAI client), and no local-inference vendor (LM Studio, Ollama, llama.cpp, vLLM, LocalAI) ships its models *as* MCP tools — all implement the host side. So "use it, don't write it" was not available. That does not change the decision; it only means the consumer writes a few hundred lines.

## Consequence for ADR-014

This decision cleaned something out of ADR-014 that should never have been there.

Appendix C had modelled its typed-pipe artifact envelope on **MCP's content schema**, and cited MCP#793 as evidence that media-by-URI was arriving upstream. Both are now removed. MCP#793 was opened and closed on the same day, with zero comments, and drove no spec change — `ImageContent` still carries base64 `data` and no URL field. And more fundamentally: **a protocol mlx-knife does not ship has no business shaping mlx-knife's formats.** A never-committed brainstorm had become a design input.

The constraint that genuinely binds the envelope comes from our own surface, and now lives in ADR-014 where it belongs: `serve` accepts media **only** as inline `data:` URLs — deliberately, since a server that dereferences URIs or reads paths on the caller's behalf is a file-fetcher. The requirement is therefore **byte-lossless *resolvability*, not structure-preserving projection**: the record is a superset, the wire form is its inline projection, and dereferencing a locator to bytes is the executor's job. A record forced to project losslessly onto the wire would collapse to inline-only and destroy the content-addressed handle — the one part that travels between nodes.

If someone later builds an MCP adapter and needs to translate our artifacts into MCP content blocks, that translation is **their** problem, on their side of the contract.

## Cross-references

- **[#56](https://github.com/mzau/mlx-knife/issues/56)** — the public MCP baseline. Closed as *not planned*; its closing comment carries this decision and the practical guidance for anyone building an adapter. **That is where builder-facing detail lives, not here.**
- **[#55](https://github.com/mzau/mlx-knife/issues/55)** — the same call, one level down: a server-side `suffix` parameter for FIM was declined because *"editor-specific templating fits better in a thin client/adapter layer than in mlx-knife's core — the server stays a lean, stateless inference engine."* This ADR applies that to the MCP surface.
- **[#51](https://github.com/mzau/mlx-knife/issues/51)** — Model Capabilities in Server API. The missing capability field; prerequisite for any HTTP client that wants to select a model rather than be told one.
- **ADR-014** (Appendix C — the artifact model) · **ADR-015** (the precedent: separate process, `serve` proxies, 501 without a backend) · **ADR-016** (the one-slot gate and its per-process scope) · **ADR-024** (pre-execution capability reject).

---
---

# Appendix: History of Thought (superseded, non-normative)

The "✅ RECOMMENDED" marker Option A once carried is historical. Option A is the disavowed anti-pattern.

**Option A — MCP as a standalone orchestration service.** A separate server on its own port, holding pipeline state, calling `/v1/*` internally. Right instinct about *where the process lives*; wrong about *what it is*. As written it carried server-side pipelines (`transcribe_and_format`, `vision_rag`) and its own `select_best_stt_model()` — a pipeline engine wearing an MCP hat. This ADR keeps the separate-process instinct and discards the orchestration.

**Option B — MCP proxy inside mlx-knife** (`/v1/mcp/{tool}` forwarding to an external MCP server, gated on `MLXK_MCP_SERVER`). Rejected: an MCP-shaped hole in the core, in order to forward elsewhere. Never implemented.

**Option C — Function calling.** Not an MCP architecture — a different surface for the same orchestration question. Tracked as [#39](https://github.com/mzau/mlx-knife/issues/39).

**"Level 2 — transparent MCP"** earns its own epitaph. It proposed that `/v1/chat/completions` embed the user's message, cosine-match it against pre-computed intent vectors, and — above 0.8 confidence — *silently reroute* the request to a pipeline tool. A node guessing what the user meant and acting on the guess, inside an endpoint claiming OpenAI compatibility. Unpredictable, untestable, invisible to the caller. Rejected without reservation.

**Use cases** (UC1 long-form audio, UC2 vision→RAG, UC3 reasoning chain, UC4 RAG query) remain plausible things a *consumer* may build. Each is a composition of the atoms, and composition belongs to the composer.

**Empirical data worth keeping (2026-04-16, 26 min of audio).** Whisper-large-v3-turbo-8bit: complete transcription in 71 s, no diarization. VibeVoice-ASR-8bit: roughly 19:30 of 26:00, minutes of processing, GPU pinned throughout — but diarization excellent (Speaker 0/1 cleanly separated). Chunking is mandatory for VibeVoice on long-form audio (~10–15 min chunks, with overlap for speaker continuity). This is why UC1 was interesting, and it remains the strongest case for a local `transcribe` tool: an MCP host genuinely cannot do this itself.

**Superseded roadmap.** The old status header promised *2.0.7 experimental, gated via `MLXK2_ENABLE_ALPHA_FEATURES=1`, 2.1 stable promotion*, over a timeline with "2.0.6 = RAG + MCP prototype". None of it happened: 2.0.6 shipped without a prototype, and the alpha gate was only ever implemented for embeddings — it never covered MCP, despite what `docs/ARCHITECTURE.md` claimed until 2026-07-14.
