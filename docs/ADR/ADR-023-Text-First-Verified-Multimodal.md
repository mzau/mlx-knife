# ADR-023: Text-First CLI + Verified Multimodal List

**Status:** Accepted
**Created:** 2026-04-17
**Related:** ADR-018 (Convert Operation), ADR-020 (Audio Backend Architecture), ADR-022 (Workspace-First Paradigm)
**Target:** 2.0.5

---

## Context

### What mlx-knife actually is

mlx-knife is an **integration layer** over three independently versioned
upstream packages:

- `mlx-lm` — text models
- `mlx-vlm` — vision-language models
- `mlx-audio` — speech-to-text and audio-capable multimodal models

Each of these packages evolves on its own release cadence and its own bug
curve. Every mlx-knife release that follows them closely inherits their bug
surface: missing assets, removed APIs, broken tokenizers, unfinished model
types. The 2.0.4 → 2.0.5 release cycle showed this pattern repeatedly —
three monkey-patches live in `audio_runner.py` alone just to keep
mlx-audio's Whisper path usable (mlx-audio #479, #645).

### The silent-fallback bug class

During 2.0.5-beta.4 stabilisation, the `convert --quantize` path in
`mlxk2/operations/convert.py:_detect_quantize_backend` was tightened to a
small whitelist of `model_type` values that route to mlx-vlm. The original
`vision_config` heuristic was removed. The intent was defensive (keep
unsupported multimodal types out of mlx-vlm), but the effect was the
opposite:

> A multimodal model whose `model_type` is not on the whitelist falls
> through to the text backend (`mlx_lm.convert`). `mlx_lm.convert` strips
> `vision_config` from the saved config during its internal rewrite
> (`mlx_lm/utils.py`: `config.pop("vision_config", None)`). The resulting
> workspace is no longer multimodal — semantically broken, silently.

Two independent reviews confirmed this as a release-blocking regression.
The bug is not "our whitelist is too small" — the bug is that the code
has no explicit rule for unknown multimodal types and defaults to a
destructive fallback.

### The treadmill problem

If mlx-knife's ambition is to make every model type from upstream
"just work", then every upstream regression becomes an mlx-knife regression.
Every new mlx-vlm model_type becomes a whitelist-update chore. Every
missing upstream asset becomes a bundle-and-patch decision. The integration
layer accumulates workarounds that the maintainer must carry forward
indefinitely — while the actual unique mlx-knife features (convert,
workspace-first, JSON API, server) get starved of attention.

### The scope question

The same two reviews disagreed politely about what to do: the narrow
"fix just these two symptoms" view vs. the broader "write down the policy
so this class of bug cannot recur" view. This ADR takes the broader view.

---

## Decision

### 1. Text-first positioning

mlx-knife is primarily a **text-model CLI for Apple Silicon**. All standard
text models work via `list`, `clone`, `run`, `convert`, `quantize`,
and `serve` without curation. Text routes through `mlx-lm` and inherits
`mlx-lm`'s capability.

### 2. Verified multimodal list

Vision and audio support is **an explicitly curated list of model types**
maintained in `mlxk2/core/capabilities.py`:

| Frozenset | Purpose |
|---|---|
| `VISION_MODEL_TYPES` | Runtime-capable vision types (routes to mlx-vlm for `run`/`serve`) |
| `VISION_QUANTIZE_TYPES` | Convert-capable vision types (routes to mlx-vlm for `convert --quantize`) |
| `STT_MODEL_TYPES` | Runtime-capable STT types (routes to mlx-audio for `run`) |
| `AUDIO_MODEL_TYPES` | All audio-capable model types (including multimodal audio) |

These lists are **independently curated** — runtime-capable does not
automatically imply quantize-capable, and vice versa. Entries carry
per-line comments noting the upstream version they were verified against.

### 3. Explicit reject at destructive boundaries

`convert --quantize` dispatches via `classify_convert_target(config)`
with the following policy order:

1. `model_type` in `VISION_QUANTIZE_TYPES` → vision backend
2. `model_type` in `STT_MODEL_TYPES` → reject (STT quantization not implemented)
3. config carries `vision_config` or `audio_config` but type is unknown
   → **hard reject** with `ErrorType.UNSUPPORTED_MULTIMODAL` (HTTP 501)
4. otherwise → text backend

The reject fires **before any filesystem side-effect** on the target path.
No partial write is possible. The error envelope names the offending
`model_type` and links to this ADR and the coverage matrix.

**Scope of the policy:** the reject applies to `convert --quantize` only.
`clone` stays byte-preserving and does not inspect `model_type`.
`run`/`serve`/`list`/`health` keep their existing capability gates
(embedding blocker, Voxtral tokenizer block, video_processor block, etc.).

### 4. No silent degradation anywhere

If a model type is not supported for an operation, mlx-knife surfaces a
typed error. It never routes to a "close-enough" backend that would
produce a silently wrong result. If a model works in `list` and `run`
but not in `convert`, the coverage matrix in
[`docs/MODEL-COVERAGE.md`](../MODEL-COVERAGE.md) says so explicitly.

---

## Transition rule: 2.0.5 → 2.0.6

### 2.0.5 is a grandfather clause (one-time)

The 2.0.5 lists contain every model type that has worked at any point
during the 2.0.x line. `VibeVoice`, `Voxtral`, `gemma3n` stay listed even
where runtime is partial or buggy. This keeps 2.0.5 a minimal-disruption
release relative to 2.0.4 — users do not find features silently removed.

### From 2.0.6 onwards: list follows upstream reality

The grandfather clause ends at 2.0.6. From that release onward:

- Entries are added when `mlx-lm` / `mlx-vlm` / `mlx-audio` correctly
  support the type and mlx-knife has verified the integration.
- Entries are **removed** when upstream breaks the type and no
  acceptable fix exists in mlx-knife. Removal is not a regression —
  it is a correction of a misleading promise.
- The coverage matrix ([`docs/MODEL-COVERAGE.md`](../MODEL-COVERAGE.md))
  carries a "Verified in" column so the transition is visible to users.

### Workaround-Sunset Policy

Every existing workaround in the code tree gets a
`# WORKAROUND: <upstream-issue> — sunset-by 2.0.6` marker and an explicit
fallback decision recorded:

- If upstream has fixed the issue by the 2.0.6 freeze → workaround is
  removed, model stays on verified list.
- If upstream has NOT fixed it → workaround is removed anyway, and the
  affected model is removed from the verified list. No perpetual patch
  maintenance.

**No new workarounds are accepted without an upstream issue link AND a
sunset marker.** This is enforceable via `grep -rn "sunset-by" mlxk2/`.

### Non-goal: mlx-knife does not exceed upstream

mlx-knife does not provide model capabilities that upstream does not
provide. It is not a missing-feature shim, not a bundle-everything
distribution, and not a fix-everything fork. If `mlx-vlm` cannot
quantize a type, `mlxk convert --quantize` cannot either — and says so
clearly.

> **2026-06 pointer (non-normative).** The same logic rejects two ideas
> floated for a hypothetical future LoRA-surface: a cross-tool
> `lora convert` (no upstream MLX wrapee — mlx-knife would have to
> *originate* and maintain the layer-map matrix itself) and a
> training-config `validate-flow` lint. Both are missing-feature-shim /
> fix-everything-fork territory → out of scope as core verbs; they belong
> as `examples/` recipes or consumer-side.

---

## Consequences

### Freed capacity

With multimodal scope bounded and the workaround treadmill capped, the
mlx-knife maintainer backlog can refocus on features that only live in
mlx-knife:

- Embeddings (ADR-015, #26)
- Reasoning API (#40)

### Visible trade-offs

- Users who installed a previous beta expecting silent convert-to-text
  of any config will now see an explicit reject. The error message
  names the model type and the policy — this is intentional.
- The coverage matrix in [`docs/MODEL-COVERAGE.md`](../MODEL-COVERAGE.md)
  becomes a living document. Entries move between ✅ / ⚠ / ❌ as upstream
  versions change.

### Guards against regression

- Two test files lock the policy contract: `test_convert_multimodal_reject.py`
  (behavior) and `test_capabilities_invariants.py` (list hygiene +
  `classify_convert_target` algorithm).
- The duplicate `_VISION_QUANTIZE_TYPES` literal in `convert.py` is gone.
  The single source of truth is `mlxk2/core/capabilities.py`.

---

## Implementation

See CHANGELOG.md entry for 2.0.5 for the concrete code changes. The
verification path is:

```bash
# Multimodal reject fires, no side-effect
mlxk convert /path/to/gemma-3n-source /tmp/out --quantize 4 --json
# -> error.type == "unsupported_multimodal", /tmp/out does not exist

# Verified vision still dispatches correctly
mlxk convert /path/to/gemma3-bf16 /tmp/out --quantize 4 --json
# -> success via mlx-vlm

# Pure text unchanged
mlxk convert /path/to/llama-bf16 /tmp/out --quantize 4 --json
# -> success via mlx-lm
```
