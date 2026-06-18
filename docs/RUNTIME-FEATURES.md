# Runtime Features — Internal Technical Reference

Status: working draft, 2026-05-05 (Iter 1 landed; the iteration log is kept internally, not in the public tree).
Captures the conceptual model that informs ADR-024 (extended) and the
capability-layer refactor planned for 2.0.6 / 2.0.7.

This is an **internal definition document**. It does not prescribe code or
ship as user documentation. Code-level decisions reference this file for
shared vocabulary; user-facing behaviour is documented in README and
SERVER-HANDBOOK.

The term *capability* is intentionally avoided in the conceptual sections.
"Capability" is a JSON-schema label and conflates several independent ideas.
The term *runtime feature* is used for the underlying concept — what mlxk
can dispatch — and *capability label* only when discussing the public schema
projection thereof.

---

## 0. Reading This Document

### 0.1 Why probe instead of trust

mlxk depends on three upstream packages — `mlx-lm`, `mlx-vlm`, `mlx-audio` —
that evolve on independent cadences and overlap in non-uniform ways. A
release of one resolves bugs the others did not have; a release of another
introduces new gaps a third inherits. Static "verified" lists curated against
any single snapshot are therefore stale by construction; the same applies to
per-model capability annotations not derived from the live stack.

The runtime-features layer addresses this by *probing* instead of *trusting*:
for each model in scope, mlxk asks the live stack at the moment of use. The
probe is anchored in [ADR-023](ADR/ADR-023-Text-First-Verified-Multimodal.md) —
mlxk **does not exceed upstream** (the probe never reports more reachable
features than mlx-{lm, vlm, audio} actually delivers), and it **does not
silently degrade** (a request for a missing feature is rejected pre-execution,
never substituted with a different feature that happens to be available).
Both clauses cease to be enforceable without a probe; they degrade into
wishes against a static list whose drift is invisible.

### 0.2 Markers

Sections in this document use four short markers when the distinction
between intended behaviour and current reality matters:

- **`[TARGET]`** — target definition: how a feature, probe, policy, or fix
  is intended to behave. Provisional unless §9 declares it stable.
- **`[CURRENT]`** — describes the live code or portfolio at the time of
  writing. Subject to drift; consult code or `git log` if precision is
  needed.
- **`[POLICY]`** — design choice anchored in [ADR-023](ADR/ADR-023-Text-First-Verified-Multimodal.md)
  (text-first, verified multimodal, no silent degradation, no exceeding
  upstream).
- **`[ROADMAP]`** — forward-looking scope; not yet implemented.

When a section header carries a marker block (see §5 reading guide), the
marker applies to all sub-bullets unless individually overridden. Markers
appear inline only when the surrounding text would otherwise be ambiguous.

### 0.3 Scope: workspace primary, cache regression-protected

The reachability vocabulary and refactor trajectory target **workspace**
models as the primary store ([ADR-022](ADR/ADR-022-Workspace-First-Paradigm.md)
makes workspaces the preferred working copy). HF-cache models continue
to work under the existing code-driven classifier (`detect_model_type` /
`detect_*_capability` plus backend availability) and remain in scope as
**regression target only** — surgical fixes (e.g. §5 Class A / B) must
improve behaviour on both populations, but no new probe machinery is
built on the cache side. Code changes that touch the classifier verify
non-regression on cache models before landing.

**Consistency requirement.** The classifier is a pure function of
(on-disk inputs, code whitelists). The same model — listed from the HF
cache or as a workspace clone of the same snapshot — must produce the
same reachable set in the same mlx-stack environment. User-visible
drift between the two views (e.g. cache shows `chat+vision`, workspace
clone of the same content shows only `chat`) would be a regression.
Any future probe caching (§8 #1) must preserve this equivalence.

---

## 1. Three Independent Axes

mlxk distinguishes three properties of a model. They are conceptually
orthogonal and live in separate code paths. A patch in one axis must not
modify fields in another. The visibility and invocation policies that
derive from these axes (§3) operationalise [ADR-023](ADR/ADR-023-Text-First-Verified-Multimodal.md);
the axes themselves are the source of truth that the policy reads from.

| Axis                       | Values                                   | Meaning                                                                                                  | Source of truth                                                |
|----------------------------|------------------------------------------|----------------------------------------------------------------------------------------------------------|----------------------------------------------------------------|
| **Health**                 | `healthy` / `unhealthy` (+ reason)       | Data-integrity status of the on-disk artefact. Index↔shards, file-completeness, optionally HF ETag/blob match. **Nothing else.** | Health-aggregator (`common.py:645` plus index validator)       |
| **Reachable runtime features** | subset of `{text, chat, vision-in, audio-in, audio-out, embeddings, …}` | Which invocations actually succeed against the current mlx stack. Result of a four-layer probe (see §2). | Reachability probe (per-model, ADR-024 extended)               |
| **Quality**                | per-feature annotation                   | Known quality limits or caveats for a feature that is reachable (e.g. "vision works but spatial-positional reading lags Pixtral"). | Verified-list + `docs/MODEL-COVERAGE.md`                       |

### 1.1 Independence

The axes can take any combination. Examples from the current portfolio:

| Model                                | Health                          | Reachable                       | Quality notes                       |
|--------------------------------------|---------------------------------|---------------------------------|-------------------------------------|
| `gemma-4-26b-a4b-it-4bit`            | healthy                         | `{chat, vision-in}`             | vision quality below Pixtral        |
| `gemma-3n-E2B-4bit` (base)           | healthy                         | `{}` (none)                    | —                                   |
| `gemma-3n-E2B-it-4bit`               | healthy                         | `{chat, vision-in, audio-in}`   | —                                   |
| `gemma-4-31b-6bit`                   | healthy                         | `{text}`                        | —                                   |
| `gemma-4-e4b-it-4bit`                | healthy                         | `{chat, vision-in, audio-in}`   | audio output truncated by mlx-vlm token defaults (separate concern) |
| `Mistral-Small-3.1-24B` (pre-repair) | unhealthy (index/shards + spatial_merge) | `{}` (loader fails on `spatial_merge`; the index/shard mismatch alone is glob-survivable — see §3.1) | — (irrelevant while broken)         |
| `Mistral-Small-3.1-24B` (post-repair)| healthy                         | `{chat, vision-in}`             | —                                   |

### 1.2 What health is NOT

- Health does **not** describe whether a loader exists.
- Health does **not** describe whether a chat-template is present.
- Health does **not** describe whether mlx-vlm has support for this `model_type`.
- Health does **not** depend on the `content_hash` value (`content_hash` is
  workspace-identity / change-detection, ADR-022 / ADR-025 — orthogonal).

A model where the on-disk files are pristine but no current mlx loader can
process them is `health=healthy, reachable=∅`. The cause is upstream
tooling, not data corruption, and remediation is "wait for upstream / use a
different variant" — not `--repair`.

### 1.3 What reachable does NOT cover

- Reachability does **not** distinguish between models that succeed cleanly
  and models that succeed but with degraded quality. Quality lives on the
  third axis. Example: `gemma-4-26b-a4b-it-4bit` vision-in is *reachable*
  even though spatial-positional Pixtral-class tasks degrade.
- Reachability is a per-modus property. A model with
  `reachable={chat, vision-in}` but unsuccessful text-only invocation is
  classified `chat`-reachable, not `text`-reachable. The `text` modus does
  not contribute to the set just because the model could in principle
  generate text given a chat prompt.

---

## 2. Reachability — Four-Layer Probe

A runtime feature `f` is *reachable* for a model `M` iff all four layers
return true for `(M, f)`:

1. **Architecture** — does `config.json` claim `f`?
   - `text`: model has token-generation architecture (effectively all causal-LM `model_type` values)
   - `vision-in`: `vision_config` is a truthy dict
   - `audio-in`: `audio_config` is a truthy dict, *or* `model_type` matches an audio-only architecture (Whisper, VibeVoice-ASR, Voxtral)
   - `audio-out`: TBD — model class not yet in mlxk's data model (see §4)
   - **Rule:** key-existence is not sufficient. `audio_config: null` (Gemma4-31b/26b stub) does not satisfy the architecture layer.

2. **Loader** — can a backend load the weights?
   - For `text`: `model_type` must be in `MLX_LM_TEXT_LOADER_TYPES` (auto-discovered from `pkgutil.iter_modules(mlx_lm.models.__path__) + MODEL_REMAPPING`-keys), **and** the config shape must match what the loader expects (empirically: presence of `audio_config` truthy-dict breaks mlx-lm's text loader for gemma4 even when `model_type` is supported — see §5).
   - For `vision-in` / `audio-in` via mlx-vlm: `model_type` must be in mlx-vlm's verified set (heuristic; refined per release).
   - For `audio-in` / `audio-out` via mlx-audio: `model_type` matches an mlx-audio-supported audio architecture.

3. **Invocation** — can the prompt-layer construct a valid call?
   - For media-input modi via mlx-vlm: a `chat_template` (in `tokenizer_config.json` / `chat_template.jinja` / `chat_template.json`) must exist and emit the relevant media placeholder tokens (`<image>`, `<audio>`, etc.).
   - **Empirical:** *base* models with `vision_config`/`audio_config` truthy fail this layer for `audio-in` (no template inserts `<audio>` placeholder → mlx-vlm errors with "0 audio tokens in text"). They may succeed degraded for `vision-in` (mlx-vlm auto-injects `<image>`), but the output is base-style continuation off the prompt and is not a useful invocation in practice.
   - For `text`: any non-empty prompt suffices.

4. **Backend** — is the relevant Python package importable in the current environment?
   - mlx-lm, mlx-vlm, mlx-audio availability checks (already present in `capabilities.py:_check_*_available`).

A feature passes reachability only when all four layers succeed for the
specific model. Reachability is not derivable from any single signal — neither
config-truthiness nor model_type-allowlist alone is sufficient.

---

## 3. Visibility & Invocation Policy

This section operationalises [ADR-023](ADR/ADR-023-Text-First-Verified-Multimodal.md)
§3 (visibility) and §4 (no silent degradation) at the runtime-feature layer.
Both clauses become enforceable by reading from the reachability probe (§2)
rather than from a static list.

The reachability axis drives two symmetric policies — one on the inventory
(visibility) side, one on the invocation side. Both share the same
per-model reachable set as source of truth.

### 3.1 Visibility (inventory side)

| Invocation              | Filter                       | Rationale                                                                                                                  |
|-------------------------|------------------------------|----------------------------------------------------------------------------------------------------------------------------|
| `mlxk list` (default)   | reachable ≠ ∅                | Human view: "what can I actually use right now". Models with no reachable feature are noise in the default workflow.        |
| `mlxk list --all`       | none                         | Full inventory. Required for disk management, repair workflows, post-hoc debugging.                                         |
| `mlxk list --json`      | none                         | Machine consumers filter themselves; emitting truth is more useful than emitting opinion.                                   |

Hiding-by-default is a reachability decision, not a health decision. Two
distinct populations are hidden:

- **Bit-rot population.** `health=unhealthy`. *Genuine* artefact loss (missing shard data, missing `spatial_merge`) → loader fails → reachable=∅. **But `unhealthy` does not imply reachable=∅:** a stale `model.safetensors.index.json` (mlx-vlm #624) is unhealthy yet **glob-survivable** — mlx-vlm/mlx-lm ignore the index and load the actual shards (empirical 2026-06-02: Qwen2.5-VL-32B loads + runs while `unhealthy`). Such a model is *reachable*; the current health-based default-hide diverges from the `reachable ≠ ∅` filter above and is the over-strict case. Remediation: `mlxk convert --repair-index` / `--repair` (for portability/correctness — not a run prerequisite for #624-only).
- **Tooling-gap population.** `health=healthy` but no loader / no chat template / no backend → reachable=∅. Remediation: wait for upstream, or use a sibling variant (`-it` chat instead of base) where reachability is non-empty.

Both populations remain accessible via `--all` and via direct path
(`mlxk show /path/to/model`). The distinction between them surfaces in
`show`, where Health and reachable-features are independently displayed.

When upstream ships a missing loader or chat-template, a previously hidden
model becomes visible **without code change** — the probe simply starts
returning a non-empty set on the next `list` invocation. Verified-list
updates (ADR-023) follow separately and govern Quality, not visibility.

The Type column (compact tag, e.g. `chat+vision`) is computed from the
reachable set. It never lists a feature that did not pass all four layers.

### 3.2 Invocation (CLI / API side)

A CLI flag or API parameter that requests a runtime feature `f` is a
pre-execution check against the reachable set:

- `f ∈ reachable(M)` → proceed.
- `f ∉ reachable(M)` → reject **before invoking the backend**, with an
  error that names the missing feature, names the layer that disqualified it,
  and (where possible) names a sibling variant or remediation.

**No silent fallback.** A request for a missing feature is never quietly
substituted with a different feature that happens to be reachable. Specific
applications:

| Request                                  | Non-reachable case                                                                                                       | Required behaviour                                                                                                                          |
|------------------------------------------|--------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------|
| `mlxk run M --translate`                 | `M` lacks audio-translate (English-only Whisper variant; non-Whisper STT model; non-STT model)                           | Hard error: "Model does not support translate. <reason>". Never silently fall back to plain transcribe.                                    |
| `mlxk run M --audio FILE`                | `M` is text-only; `M` is base + media-input invocation gap; `M`'s loader is missing                                      | Hard error with specific layer named. Never silently transcribe nothing or attempt text-only.                                              |
| `mlxk run M --image FILE`                | `M` is text-only; mllama-class without text-tower; base variant of multimodal architecture                              | Hard error. Never silently strip the image and run text-only.                                                                              |
| `mlxk run M "prompt"` (text-only)        | `M` requires media input (multimodal-only architecture, e.g. mllama)                                                     | Hard error: "Model requires media input. Use `--image` / `--audio`." (ADR-024 case.)                                                       |

This couples to the same reachability probe as visibility — a feature that
is *visible* in the Type column is invocable; a feature that is *not* in the
Type column rejects on invocation. The two policies cannot drift, because
they read the same set.

**Consistency with ARCHITECTURE.md.** The pre-execution-reject behaviour is
the No-Silent-Fallbacks (#2) and Fail-Fast (#3) principles applied to the
runtime-features layer.

---

## 4. Audio Task Inventory

Audio is the highest-dimension area in the runtime-features space. To avoid
silently scoping it to "what mlxk currently does", the table below lists all
audio task classes that are conceptually distinct, with portfolio status:

- ✅ in portfolio today
- 🟡 reachable in principle via current backend, but mlxk lacks routing or no test model on hand
- ⚫ out of mlxk's data model today; would require new pipeline (not just a label)

### 4.1 Audio-In → Text-Out (understanding tasks)

| Task                       | Definition                                                            | Backend     | Status     | Notes                                                                                                              |
|----------------------------|-----------------------------------------------------------------------|-------------|------------|--------------------------------------------------------------------------------------------------------------------|
| ASR (Speech-to-Text)       | Audio → verbatim transcription                                        | mlx-audio   | ✅         | Whisper. ASR is *only* transcription — speaker info is not part of the definition.                                |
| ASR + Diarisation          | ASR + anonymous speaker clustering ("Speaker 1", "Speaker 2")         | mlx-audio   | ✅         | VibeVoice-ASR.                                                                                                     |
| ASR + Speech Translation (English-only target) | Audio in any source language → text in **English** (fixed target — Whisper architectural constraint, no free target-language argument) | mlx-audio   | ✅        | Shipped 2.0.7 (#54): CLI `mlxk run --audio FILE --translate [en]` and server `POST /v1/audio/translations` (OpenAI-compatible, hardcoded `task=translate`). `task` is threaded through `audio_runner.transcribe()`; mlxk additionally sets `condition_on_previous_text=False` on translate to break Whisper's long-form repetition loop. Reachability is per-model (`detect_audio_translate_en_capability`): multilingual non-turbo Whisper has it; `whisper.en` variants (no `<\|translate\|>` token) and turbo variants (reduced decoder, non-English garbage) do not, nor do non-Whisper STT models. Non-capable models reject pre-execution per §3.2 (CLI error / server HTTP 422) — never a silent transcribe fallback. |
| ASR + Free-target Speech-to-Text Translation | Audio in language A → text in arbitrary language B                    | —           | ⚫         | Not in mlxk portfolio. Requires Seamless-M4T-class S2TT models; not in mlx-audio. Distinct task class from English-only Whisper translation, often conflated. |
| Grounded Audio-Chat        | Audio + question text → text answer *about* the audio                 | mlx-vlm     | ✅         | Gemma3n-it, Gemma4-e4b-it. Distinct from ASR: model can answer questions, not only transcribe.                     |
| Audio Classification       | Audio → label (language, emotion, scene, instrument)                  | mlx-audio   | ⚫         | No model in portfolio.                                                                                             |
| Audio Captioning           | Audio (incl. non-speech) → free-form description                      | mlx-audio   | ⚫         | No model in portfolio.                                                                                             |
| Speaker Identification     | Audio → known speaker name from enrolled database                     | —           | ⚫         | Distinct task class. Requires enrolment-data pipeline that mlxk does not have.                                     |
| Speaker Verification       | Audio + speaker-X profile → yes/no                                    | —           | ⚫         | Same comment as above.                                                                                             |

### 4.2 Text-In → Audio-Out (generation tasks)

| Task                | Definition                                                       | Backend     | Status | Notes                                                                                              |
|---------------------|------------------------------------------------------------------|-------------|--------|----------------------------------------------------------------------------------------------------|
| TTS (Text-to-Speech)| Text → audio                                                     | mlx-audio   | ⚫     | mlxk's data model has no `tts` `model_type`, no audio-output CLI mode. Pipeline gap, not a label gap. |
| Voice-cloning TTS   | Text + reference audio → audio in cloned voice                   | mlx-audio   | ⚫     | VibeVoice-main is a candidate. Subset of TTS once TTS is supported.                                |
| Music generation    | Text prompt → music                                              | (other)     | ⚫     | MusicGen-class. Outside mlx-audio scope as currently defined.                                       |

### 4.3 Audio-In → Audio-Out

| Task                            | Definition                                          | Backend     | Status |
|---------------------------------|-----------------------------------------------------|-------------|--------|
| Voice Conversion                | Audio → audio with different voice                  | mlx-audio   | ⚫     |
| Speech Enhancement / Denoising  | Noisy audio → clean audio                           | mlx-audio   | ⚫     |
| Source Separation               | Mixed audio → component stems (vocals, drums, …)    | (other)     | ⚫     |
| Speech-to-Speech Translation    | Audio in language A → audio in language B           | mlx-audio   | ⚫     |

### 4.4 Audio → Vector

| Task               | Definition                                            | Backend     | Status |
|--------------------|-------------------------------------------------------|-------------|--------|
| Audio Embeddings   | Audio → fixed-dimension vector for retrieval/similarity | (other)   | ⚫ — text-embeddings are 2.0.7-roadmap, audio is further out. |

### 4.5 Implications

- **TTS is a pipeline gap, not a label gap.** Adding TTS support is not a matter of extending an enum; it requires an audio-output CLI mode, output sinks (file/player), and a `tts` `model_type` peer to `base`/`chat`/`audio`. To be flagged in the next planning round if a TTS test model enters the portfolio.
- **STT classification today is too narrow.** `STT_MODEL_TYPES` in `capabilities.py` lists only `{"whisper"}`. VibeVoice-ASR and (future) Voxtral / Qwen3-ASR are also ASR but currently misclassify as `base+audio` — see §5 bug class A.
- **Multimodal audio-in is *only* reachable on chat variants.** Base models with truthy `audio_config` cannot be invoked with `--audio` (no chat template → no `<audio>` placeholder). This is bug class D in §5; the architecture layer alone is misleading without the invocation layer.
- **English-only Whisper translate is a separate reachable feature, not an ASR sub-mode.** A Whisper variant that supports `<|translate|>` has both `audio-in (transcribe)` and `audio-translate-en` reachable. A `whisper.en` variant has only the former. They are listed as distinct features in the reachable set, and `--translate` against a model that has only `audio-in` rejects pre-execution per §3.2. This was not previously a separate feature in mlxk — the inventory exposed only `audio` as a single tag, conflating the two. Detection signal: `<|translate|>` special-token presence in the tokenizer (bytes-gated scan, analogous to existing `extract_chat_template`).
- **Free-target Speech-to-Text Translation is a different task class entirely** (Seamless-M4T family). Marked ⚫ in §4.1; not implementable with the current portfolio. Important to keep separate from English-only translate to avoid the language-pair illusion.

---

## 5. Bug Class Catalog (current)

> **Reading guide.** The four-class framework and each class's **Fix path**
> are `[TARGET]` specifications — provisional until validated by POC and
> stabilisation. The **Symptom** / **Cause** / **Affected** / **Empirical
> data** entries are `[CURRENT]` observations against live code and the
> portfolio at the time of writing; they are subject to drift.

Four orthogonal classes have been identified in this analysis. Each maps
to a distinct fix path; they cannot be collapsed into one patch.

### Class A — STT model_type misclassification

**Symptom.** ASR-only models are tagged `base+audio` instead of `audio`.

**Cause.** `STT_MODEL_TYPES = frozenset({"whisper"})` (`capabilities.py:119`) is too narrow, *and* `detect_model_type` step 4 uses exact set membership (line 202) while sibling `detect_audio_capability` (line 307) uses substring matching. Inconsistent matchers + narrow set → `vibevoice_asr` falls through to default `"base"`.

**Affected.** VibeVoice-ASR-{4bit, 8bit}. Future Voxtral, Qwen3-ASR.

**Fix path.** Extend `STT_MODEL_TYPES` to `{"whisper", "vibevoice", "voxtral"}` (substring tokens), align `detect_model_type` to substring matching, ensuring consistency with `detect_audio_capability`. Independent of any reachability logic.

**Fix path validated 2026-05-06 (2.0.6 surgical patch).** `STT_MODEL_TYPES` extended as planned. `detect_model_type` Step 4 (`common.py:202`) switched to substring matching. Effect verified on VibeVoice-ASR-{4,8}bit (now `audio` instead of `base+audio`); whisper-large-v3-turbo-{4,8}bit unchanged (was already matched via exact `whisper`).

### Class B — Audio capability false positive (key-existence vs truthy)

**Symptom.** Models with `audio_config: null` (key present, value null) are tagged with audio.

**Cause.** `detect_audio_capability` step 1 (`common.py:299`) tests `"audio_config" in config` rather than truthy-dict. Same key-existence pattern at line 328 for `audio_seq_length`.

**Affected.** Gemma4 family with stripped or stubbed audio block: `gemma-4-31b-6bit`, `gemma-4-31b-bf16`, `gemma-4-26b-a4b-it-4bit`. The same pattern likely affects vision detection where a converter packs an empty `vision_config` stub.

**Fix path.** Replace key-existence checks with truthy-dict / truthy-value checks. Apply symmetrically to `detect_vision_capability` for the Granite-Speech case (per existing memory `project_capability_label_bugs`).

**Fix path validated 2026-05-06 (2.0.6 surgical patch).** Step 1 (`common.py:299`) switched to `isinstance(audio_config, dict) and bool(audio_config)`. `detect_vision_capability` (`common.py:233`) symmetric pull: `isinstance(vision_config, dict) and bool(vision_config)` — resolves §A.3 (`vision_config: {}` empty-dict edge). Step 4 (`audio_seq_length` at `common.py:328`) **dropped entirely** after empirical correction: the Gemma4 processor template ships `audio_seq_length=750` for every variant including audio-null ones, so the truthy-value predicate alone was insufficient (the value *is* truthy even when the model has no audio tower). No real audio model in the portfolio relies on this signal as standalone — all caught by Step 1 (truthy `audio_config`) or Step 2 (model_type substring). Effect verified on gemma-4-{26b-a4b-it-4bit, 31b-bf16} (audio tag dropped); regression-anchors gemma-4-e4b-it-4bit (`chat+vision+audio` preserved via truthy `audio_config`) and gemma-3n-{E2B,E2B-it} (audio preserved). **Lesson for Iter 2/3:** the original truthy-predicate framing was sound for `audio_config` (a dict-typed marker) but insufficient for `audio_seq_length` (an int-typed processor parameter that template-inheritance can keep truthy independent of architecture). When future Class C/D fixes derive predicates from `[CURRENT]` empirics, validate against real config files before declaring the predicate complete.

### Class C — Loader gap (mlx-lm cannot load text-only)

**Symptom.** `mlxk run <model> "prompt"` fails with cryptic errors:

- `'model'` KeyError (gemma3n family — no loader registered)
- "Received N parameters not in model" (gemma4-e4b — loader exists but config shape exceeds it)

**Cause.** mlx-lm's text-tower loader does not handle every multimodal architecture. The discriminator is **not** purely `model_type`-based; empirically it depends on `(model_type, audio_config-truthy)` for the gemma4 family.

**Empirical data (this session):**

| Model                     | model_type | vision_config | audio_config | text-only |
|---------------------------|------------|---------------|--------------|-----------|
| gemma-4-31b-6bit          | gemma4     | absent/null   | null         | ✅       |
| gemma-4-26b-a4b-it-4bit   | gemma4     | dict          | null         | ✅       |
| gemma-4-e4b-it-4bit       | gemma4     | dict          | dict         | ❌ param mismatch |
| gemma-3n-E2B-4bit         | gemma3n    | dict          | dict         | ❌ KeyError 'model' |
| gemma-3n-E2B-it-4bit      | gemma3n    | dict          | dict         | ❌ KeyError 'model' |

→ Hypothesis: mlx-lm text-only is loadable iff `model_type ∈ MLX_LM_TEXT_LOADER_TYPES` (auto-discovered) AND `audio_config` is not a truthy dict. `vision_config`-truthy alone does not break the loader.

**Affected.** All VLMs whose text-tower mlx-lm cannot extract. Generalises to mllama (per [ADR-024](ADR/ADR-024-Pre-Execution-Capability-Mismatch-Reject.md)) which fails text-only outright.

**Fix path.** Implement reachability layer 2 as a probe that combines an auto-discovered `model_type` allowlist with config-shape filters. Use the same probe as the single source of truth for: (a) routing in `run.py`, (b) the health-aggregator's text-load gate in `common.py:645`, (c) the `text` capability label.

[ADR-024](ADR/ADR-024-Pre-Execution-Capability-Mismatch-Reject.md) is the implementation decision record for this. The Class A instantiation (STT / Embedding pre-execution-reject) shipped in 2.0.6 (`2de2f21`, smoke §P). Class C extends the same pattern with the `MLX_LM_TEXT_LOADER_TYPES` allowlist + show/run alignment in `common.py:645-647`.

**Slot:** DEFER 2.1 (tracking: [Issue #53](https://github.com/mzau/mlx-knife/issues/53)).

### Class D — Invocation gap (base + media-input is unreachable)

**Symptom.** A base model with `vision_config`/`audio_config` truthy cannot be successfully invoked with `--audio` or `--image`:

- `--audio`: mlx-vlm errors with "0 audio tokens in the text and N tokens from audio embeddings" — the prompt has no `<audio>` placeholder for the base model's tokenizer to align against.
- `--image`: mlx-vlm auto-injects `<image>` and produces output, but the output is base-continuation off whatever metadata header the prompt contained — not a useful answer.

**Cause.** Multimodal grounded chat presupposes a chat template that emits media placeholder tokens. Base models lack such templates. The architecture layer (config truthy) and loader layer (mlx-vlm supports model_type) succeed, but the invocation layer fails or degenerates.

**Affected.** Any base variant of a multimodal architecture: `gemma-3n-E2B-4bit`, `gemma-4-31b-bf16` (for media inputs; text-only is fine for 31b), other future base+vision/audio combinations.

**Fix path.** Reachability layer 3 must probe for chat-template-with-media-placeholder before reporting `vision-in` or `audio-in` as reachable. The result for base+multimodal is: text-only reachable (if loader passes), media-in not reachable. The `-it` sibling variant remains reachable for media.

**Slot:** DEFER 2.1 (tied to the Iter 2/3 reachability work — Class D rephrase from „reachable=∅" to „policy-rejected per ADR-023 §4"; no ADR yet, design lives in the Iter 2/3 plan).

---

## 6. Discriminator Summary

For the reachability probe of feature `text` against a model `M`:

```
reachable(M, "text") =
    M.model_type ∈ MLX_LM_TEXT_LOADER_TYPES               # layer 2 (loader)
    AND not is_truthy_dict(M.config.audio_config)         # layer 2 (loader, empirical)
    AND prompt_layer_supports("text", M)                  # layer 3 (trivially true for text)
    AND mlx_lm_available()                                 # layer 4 (backend)
```

For the reachability probe of feature `audio-in`:

```
reachable(M, "audio-in") =
    is_truthy_dict(M.config.audio_config)                 # layer 1 (architecture)
        OR M.model_type matches AUDIO_ONLY_TYPES           # STT class
    AND backend_loader_available_for_audio(M)             # layer 2
    AND chat_template_emits_audio_placeholder(M)          # layer 3 — fails for base
        OR M is STT-only (audio_in is the only modus)     # STT does not need chat template
    AND (mlx_vlm_available() OR mlx_audio_available())    # layer 4
```

For the reachability probe of feature `audio-translate-en` (Whisper-architecture
translate task — produces English output regardless of source language):

```
reachable(M, "audio-translate-en") =                      # detect_audio_translate_en_capability(name, config)
    M.model_type in WHISPER_DERIVED_TRANSLATE_TYPES       # layer 1+2 — PRIMARY signal (architecture + loader allowlist)
    AND NOT english_only_variant(M.name)                  # whisper-*.en lack the <|translate|> token (architectural)
    AND NOT turbo_variant(M.name)                          # turbo: token present but reduced decoder cannot translate (functional)
    AND audio_in_invocation_pathway_works(M)              # layer 3 — trivially true for STT
    AND mlx_audio_available()                              # layer 4 — verified to accept task='translate' (shipped 2.0.7)
```

Implementation note (shipped 2.0.7, #54): the discriminator is **config-first** —
`model_type` plus a name-pattern, not a tokenizer byte-scan. mlx-community Whisper
checkpoints ship only `config.json` + weights (sometimes `multilingual.tiktoken`),
no HF tokenizer JSON, so a `<|translate|>`-token scan is **not load-bearing** on the
current portfolio (Finding 7-A/7-B). A `whisper.en` variant is excluded on the
architectural basis (its english-only vocabulary lacks the token); a **turbo**
variant is excluded on a **functional** basis — the `<|translate|>` token is present
but the reduced 4-layer decoder emits non-English garbage (Finding 7-C). A non-Whisper
STT or non-STT model fails at layer 1+2. On the translate path mlxk additionally
applies a quality-relevant invocation default — `condition_on_previous_text=False` —
to break Whisper's long-form repetition loop (Finding 7-D). Per §3.2, a
`mlxk run M --translate` call (or `POST /v1/audio/translations`) against any
non-capable model rejects pre-execution with a hint (CLI error / server HTTP 422),
never silently transcribes instead.

Symmetric forms apply for `vision-in`, `audio-out` (when added), etc.

---

## 7. Cross-References

| Topic                                  | Anchor                                                                                                       |
|----------------------------------------|---------------------------------------------------------------------------------------------------------------|
| Current implementation state           | `docs/ARCHITECTURE.md` — Resolve → Probe → Policy → Load → Run pipeline as actually wired today. Companion to this document; expected to drift from it until the reachability refactor lands, then to converge. |
| Health (data integrity)                | Health-aggregator in `mlxk2/operations/common.py:645`; index validator. No dedicated ADR yet — code is the truth. |
| Workspace identity / change-detection  | `content_hash`. ADR-022 (Workspace-First Paradigm), ADR-025 (`content_hash` v2).                              |
| Reachability probe + capability labels | ADR-024 (Vision-only Pre-Execution Routing → extend to multimodal-only routing + capability-layer refactor). |
| Quality annotations                    | ADR-023 (Text-First + Verified Multimodal), `docs/MODEL-COVERAGE.md`.                                         |
| Convert / Repair workflows             | ADR-018 (Convert Operation, with planned `--repair` Phase 2 update).                                          |
| Audio backend architecture             | ADR-019 (Audio-Input Support), ADR-020 (Audio Backend Architecture).                                          |

---

## 8. Open Questions

The following are unresolved at the time of writing and must be addressed
before the reachability refactor lands. Each is recorded here so the
implementation can pick up exactly where this discussion stops.

1. **Reachability probe cost.** As §2 defines them, the four layers are in-memory checks: Layer 2 = `model_type ∈ allowlist` (not a real loader import), Layer 3 = bytes-gated tokenizer scan, Layer 4 = `importlib.util.find_spec`. For a `list` operation the cost is bounded — N × small config-file reads — and no probe-cache is needed today. If a future probe form becomes expensive enough to require caching, that is a workspace concern only: workspace sentinels (ADR-022 / ADR-025) are the natural home, with no central state store, and any cache key must include the mlx-stack version tuple so cached workspace listings stay equivalent to live-classified cache listings of the same content (§0.3 consistency requirement). Cache models stay re-classified per-call (§0.3 scope); there is nothing to invalidate that the next call would not naturally re-derive.

2. **31b-bf16 text-only verification skipped.** `gemma-4-31b-bf16` (62.6 GB) was not text-smoked due to RAM constraints (64 GB host). Result inferred from `gemma-4-26b-a4b-it-4bit` (same family, same `audio_config: null`, text works). Architectural inference is solid; the skipped smoke is noted for completeness.

3. **mllama Vision-only smoke pending second-vendor confirmation.** Class C was empirically verified on the gemma family. The original ADR-024 motivation case is mllama (Llama-3.2-Vision). A second-vendor smoke should confirm the discriminator generalises before the reachability probe is wired into routing. **Resolved 2026-06-02** (second-vendor: molmo2 / AllenAI, against mlx-vlm 0.6.0): `mlx_lm._get_classes` rejects `molmo2` exactly as it rejects `mllama`; vision-in is reachable via mlx-vlm, text-only is not — the gemma-derived discriminator (`model_type ∈ MLX_LM_TEXT_LOADER_TYPES` gates text reachability) generalises to a non-gemma/non-llama vendor. mllama-specific smoke remains nominally untaken, but the generalisation concern this question raised is addressed.
4. **Quality axis structure.** Currently per-feature annotations live free-form in `MODEL-COVERAGE.md`. As the Verified-list grows (ADR-023 trajectory), a more structured form may be required (per-feature `{quality: ok | partial | poor, notes: …}`). Out of scope for the first iteration of this document.

5. **`--healthy` filter flag.** Inverse of `--all`: filter models where `health = healthy` (regardless of reachability). Plausibly useful, but no concrete user request yet. Tracked as opportunistic.

6. **TTS pipeline.** §4.5 notes that TTS is a pipeline gap, not a label gap. The decision whether 2.0.7+ adds a `tts` `model_type` peer to `base`/`chat`/`audio` plus an audio-output CLI mode is open and orthogonal to this document.

---

## 9. Document Status

- **Source of authority:** this document is conceptual. It captures *intended* runtime-feature semantics. `docs/ARCHITECTURE.md` is the companion implementation-state document and captures what the code actually does today. The two are expected to differ during a refactor cycle and converge after it.
- **Update triggers:** new bug class identified; new audio task class enters mlxk's scope; reachability probe form changes; visibility policy changes.
- **Implementation triggers (separate sessions):** when the reachability refactor or any of the four bug-class fixes lands, `docs/ARCHITECTURE.md` is updated as part of the same patch series so its description of the live pipeline stays current.
- **Audience:** maintainers and contributors. Not user-facing. README and SERVER-HANDBOOK remain the user-facing references.
- **Provisional default:** target-side content in this document (probe forms, discriminator pseudocode, visibility policy, reachability semantics) is provisional pending POC implementation experience. Revision following observation of upstream behaviour and `mlxk`'s user-facing surface is the expected mode, not a regression. The marker convention defined in §0.2 distinguishes `[TARGET]` provisional content from `[CURRENT]` observations and `[POLICY]` ADR-anchored decisions; iteration progress is tracked in a separate internal iteration log.

---

## Appendix A — Iteration Backlog

The following four items are underspecified in this draft and are tracked
for follow-up iterations. Each is independent and can be addressed in any
order. They differ from §8 Open Questions in that each names a specific
gap in the conceptual model that blocks implementation, not a
forward-looking design choice.

### A.1 Layer-3 media-placeholder probe form (§2, §6)

The reachability discriminator references
`chat_template_emits_<media>_placeholder(M)` without specifying the probe
form. Options to evaluate:

- render the chat template with an empty conversation and grep stdout for
  `<image>` / `<audio>` placeholders;
- inspect the tokenizer's special-token list (analogous to the bytes-gated
  `<|translate|>` scan in §4.5);
- static parse of the template source.

Decision blocks Layer-3 implementation for `vision-in` and `audio-in`. The
same form applies to both modalities once chosen.

### A.2 HF-cache reachability-cache asymmetry (§8 #1) — resolved 2026-05-06

Resolved by §8 #1 update. The four-layer probe is in-memory and bounded,
so no cross-source caching mechanism is needed; cache models are
re-classified per-call from read-only on-disk inputs (§0.3 scope). The
asymmetry that motivated the question does not exist in the chosen probe
form. If a future probe layer ever becomes expensive enough to require
caching, the natural home is the workspace sentinel — cache models stay
re-classified per call, in line with their read-only role.

### A.3 Class B vision-detection precision (§5 Class B) — resolved 2026-05-06

**Resolved 2026-05-06.** Predicate `isinstance(x, dict) and bool(x)`
chosen, applied symmetrically to `audio_config` and `vision_config`.
The empty-dict edge (`vision_config: {}`, `audio_config: {}`) is now
filtered. See §5 Class B Fix-path note for the empirical correction
on the audio-side (Step 4 `audio_seq_length` dropped entirely after
the Gemma4 processor-template-stub finding — the typed predicate
applied to `audio_seq_length` would still have admitted `750`).

### A.4 Layer-2 "mlx-vlm verified set" substance (§2)

§2 Layer 2 references "mlx-vlm's verified set" without anchoring to a
concrete list. Candidates:

- `VISION_MODEL_TYPES` (current runtime list);
- `VISION_QUANTIZE_TYPES` (current convert-quantize list);
- a new auto-discovered list analogous to `MLX_LM_TEXT_LOADER_TYPES`;
- a per-version curated set with empirical refresh.

Open: which of these is the source of truth for Layer 2, and what triggers
refresh (per release, on dep-bump, on first probe failure)? Without this,
Layer 2 is the most under-specified gate in the four-layer probe.
