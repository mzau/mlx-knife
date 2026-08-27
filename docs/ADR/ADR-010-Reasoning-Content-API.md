# ADR-010: Reasoning Content Schema & Shape-Derived Segmentation (Issue #40)

**Status:** Accepted — scoped. All decisions taken; blocked on implementation only.
**When:** the next slot that can absorb a runner refactor plus an additive API field.
Pulled by a live consumer symptom — reasoning traces reach `content`, and downstream
clients strip them with fragile marker matching.
**Date:** 2026-08-27 (measured revision; replaces the 2026-06-20 scope, see below)
**Depends On:** nothing. The formerly-planned 2.0.8 "detection bite" is **withdrawn** —
see *Withdrawal*.
**Affects:** `core/runner/__init__.py`, `core/runner/reasoning_format.py`,
`core/runner/stop_tokens.py`, `core/reasoning.py`, `operations/run.py`,
Server API (`/v1/chat/completions`).
**Design line:** ADR-023 (Text-First, Verified Multimodal) — follow upstream reality,
keep the internal maintenance footprint small, no curation database. Extended here to
**wire contracts**: do not be an ecosystem outlier on a field name or a message shape.

> **Supersession note (2025-10-21 → 2026-06-20).** The original ADR-010 draft tied
> reasoning structuring to name-substring `model_type` detection, a hard dependency on
> #33 (system prompts), and a "no breaking changes" constraint on the inline
> `**[Reasoning]**` / `**[Answer]**` markdown. Empirical findings from the 2.0.6
> dep-update work invalidated all three: modern models emit reasoning markers natively
> via `chat_template` (so the #33 dependency drops), and the inline markdown is
> *replaced* by a structured field rather than preserved.
>
> **Second supersession (2026-06-20 → 2026-08-27).** The 2026-06-20 scope treated the
> problem as *N dialects* (a vocabulary problem) and proposed Jinja-template parsing to
> derive markers, with a 2.0.8 "detection bite" shipping first. Measurement across five
> model families invalidated four points of that scope: the structural boundary is
> **two shapes**, not N dialects; the marker vocabulary is available without parsing
> Jinja; the bite as specified would **de-detect gpt-oss**; and the CLI section proposed
> the opposite of what the evidence supports. Decision 3, left open in June, is closed
> here on measurement. The body below is the canonical ADR-010.

---

## Withdrawal: the 2.0.8 detection bite

The 2026-06-20 scope carved out a cheap precursor — replace the name-list in
`detect_model_type()` with template introspection, ship it separately, keep this ADR for
"everything the bite cannot do". **That carve-out is withdrawn.** Three measurements make
it unsafe as an independent step:

1. **It de-detects gpt-oss.** `mlx_lm.tokenizer_utils._infer_thinking` recognises
   `<think>`, `<longcat_think>` and the `<|channel>` / `<channel|>` pair. Harmony uses
   `<|channel|>` — pipes on both sides — so `has_thinking` is **`False`** for
   `gpt-oss-20b`, and its `chat_template` contains no `enable_thinking`. Any bite that
   swaps the name-list for marker introspection loses the one family that works today.
2. **The formatting it would make "honest" lives in the wrong layer** and is itself
   being retracted (see Decision 5 and *CLI surface*).
3. **It cannot be verified in isolation.** The acceptance property established here
   (Decision 6) compares CLI output against the server's segmentation; a bite that
   changes only one of the two has nothing to be checked against.

Consequence: the reasoning subsystem is changed **once**, here, not in two releases.

---

## Context

Reasoning models emit chain-of-thought inline with the answer. The current state, measured
2026-08-27 on the development stack (mlx-lm 0.31.3, mlx-vlm 0.6.10, transformers 5.14.1):

**mlx-knife is currently an outlier on four counts, none of them decided:**

- **Terminal markdown reaches the API.** `serve` calls `runner.generate_streaming()`
  (`core/server/streaming.py:77`, `:234`) and `runner.generate_batch()`
  (`core/server/handlers/chat.py:190`, `core/server_base.py:681`) **without**
  `hide_reasoning`, which defaults to `False` = *show, formatted*. For the gpt-oss family
  the runner therefore emits `**[Reasoning]**\n…\n---\n\n**[Answer]**\n…` into the
  OpenAI `content` field. No other OpenAI-compatible server does this, and no client can
  anticipate it.
- **No reasoning field exists.** `reasoning_content` appears in the tree only as a local
  variable name (`core/reasoning.py:235`, `core/runner/reasoning_format.py:33`).
- **The chat template's own default is overridden, in opposite directions.**
  `mlx_lm/tokenizer_utils.py:336-337` fills `enable_thinking = self.has_thinking` when the
  caller is silent; `mlx_vlm/prompt_utils.py:783-787` fills `False` under the same
  condition. mlx-knife passes the kwarg nowhere (`core/runner/chat_format.py:12`, `:31`;
  `core/vision_runner.py:215`), so **the same model reasons or does not depending on
  whether an image is attached.**
- **Dialect detection is name-based** (`core/reasoning.py:54-65`), where the neighbouring
  implementation in the same ecosystem derives from config and vocabulary with an explicit
  note that repository names must not be used.

### Downstream consequence (not cosmetic)

An OpenAI-API consumer that parses `content` is handed the reasoning as the answer. In one
client session a GLM trace contained several re-drafted project trees (the model redrew a
file tree 4–5× while thinking) and the client's project-structure detector fired on each →
multiple spurious "bulk download" affordances from one logical reply. Separating the trace
removes this at the source.

Note: semantic re-drafting ≠ repetition loop — the redrawn trees differ token-by-token, so
an n-gram / `no_repeat` guard would miss them and false-positive on legitimate long code
output. The lever is reasoning *segmentation*, not generation-length heuristics.

### Measured landscape

Five families, two independent axes. The **shape** decides how the answer is recovered;
the **controls** decide whether reasoning can be switched off at all.

| Model | Shape | on/off | level | template default |
|---|---|---|---|---|
| `Qwen3.8-27B-8bit` | delimited, **prompt-opened** `<think>` | `enable_thinking` | `reasoning_effort` `xhigh\|medium\|low` | **on**, `xhigh` |
| `GLM-4.7-Flash-8bit` | delimited, **prompt-opened** `<think>` | `enable_thinking` | — | **on** |
| `gemma-4-e4b-it-4bit` | delimited `<\|channel>thought` / `<channel\|>` | `enable_thinking` (**opt-in**) | — | **off** |
| `gpt-oss-20b` | **channel-selected** (Harmony) | — | `reasoning_effort` `low\|medium\|high` | **on**, `medium` |
| `QwQ-32B` | delimited `<think>` | — | — | on |

Two consequences that the June scope did not have:

- **The structural boundary is the shape, not the dialect.** `gemma-4`'s markers are
  lexically unusual but structurally identical to `<think>`: trace between a start and an
  end marker, answer = everything after. The genuinely distinct case is Harmony, where the
  answer is *selected by channel label* and structure continues **after** it (`<|return|>`,
  a `commentary` channel for tool calls). That is why the current gpt-oss path needs four
  `skip_tokens` plus a `conditional_skip` (`core/reasoning.py:30`, `:32`).
- **`QwQ` is not Harmony.** `core/reasoning.py:64-65` maps `qwq → 'gpt-oss'` with the
  comment *"QwQ uses similar format to GPT-OSS"*. Measured against both
  `mlx-community/QwQ-32B-4bit` and `Qwen/QwQ-32B`: QwQ uses `<think>` / `</think>`, i.e.
  the `deepseek` entry. The mapping is a **false positive** — detected, then parsed with
  regexes that can never match, plus `<|return|>` added as a stop token the model does not
  have. It is worse than the undetected case because it looks handled.

---

## Decision 1 — API schema: one value, two field names, both transports

Follow the convention the ecosystem converged on. A separate field, never folded into
`content`.

The compatibility problem is **not** which marker dialect to expose — it is which *field
name*, because two conventions circulate: `reasoning_content` (DeepSeek) and `reasoning`.
The neighbouring server in this ecosystem resolves it by emitting **both**, with the same
string value (`mlx_vlm/server/openai.py:1676-1677`), and accepting both on input
(`:148-149`). mlx-knife does the same. Cost: one duplicated string per response.

**Non-streaming** — `choices[].message`:

```jsonc
{
  "role": "assistant",
  "content": "Here is the project structure …",        // final answer only
  "reasoning_content": "The user wants a snake game …", // omitted when absent
  "reasoning": "The user wants a snake game …"          // same value, alias
}
```

**Streaming (SSE)** — distinct deltas, never interleaved into one field, and **both names
on the delta too**, or a streaming client sees a different world than a batching one:

```jsonc
{"choices":[{"delta":{"reasoning_content":"The user wants","reasoning":"The user wants"}}]}
{"choices":[{"delta":{"content":"Here is"}}]}
```

**Rules:**

- The field is **omitted** (not `null`, not `""`) when there is no segmented reasoning.
- It is populated **only when segmentation succeeded**. A model whose markers are not
  recognised gets a clean `content` passthrough — better an unsegmented answer than a
  mis-split. Fail loud in tests, never in the schema.
- Reasoning deltas precede content deltas for a given choice; the parser must not buffer
  the whole trace before the first content delta beyond what segmentation requires.
- **The absent field carries three cases, and the schema does not distinguish them:** the
  model cannot reason · it can but did not this time · its shape is unrecognised. A client
  that needs the difference reads it from the capability surface (Decision 4's non-goal,
  → #51), not from the field.

**Out of scope:** OpenAI's Responses API shape (`{"type":"reasoning"}` output items) is a
different endpoint. mlx-knife serves `/health`, `/v1/models`, `/v1/chat/completions`,
`/v1/completions`, `/v1/embeddings`, `/v1/audio/transcriptions`,
`/v1/audio/translations` — there is no `/v1/responses`, so that shape does not apply.

---

## Decision 2 — two readers, selected by a checked condition

The June scope proposed deriving markers by inspecting the Jinja `chat_template`, and named
that its main implementation risk ("the marker grammar is Jinja embedded … may need a small
per-grammar normaliser"). That risk is largely avoidable: the markers are available from the
**vocabulary**, and the structural split is binary.

One contract, two implementations:

```
segment(raw_completion, shape) -> (reasoning | None, answer)
```

| Reader | Selected when | Recovers the answer by |
|---|---|---|
| **delimited** | a start/end marker pair is known for the model | everything after the end marker |
| **channel-selected** | the vocabulary carries `<\|channel\|>` + `<\|message\|>` + `<\|start\|>` | the segment labelled `final` |
| *(none)* | neither | passthrough — no field emitted |

**Marker source, in order:**

1. `mlx_lm.tokenizer_utils.TokenizerWrapper` — `has_thinking` / `think_start` /
   `think_end`, inferred from the vocabulary. Verified to resolve `<think>`/`</think>`
   (Qwen3.8, GLM-4.7, QwQ) **and** `<|channel>thought`/`<channel|>` (gemma-4). This is the
   dialect the June scope called the expensive one; it needs no template parsing.
2. `PATTERNS` (`core/reasoning.py:22-51`) as a **fallback and cross-check** for dialects the
   vocabulary probe misses. It is corrected to three *distinct* entries — `gpt-oss` and
   `qwq` are not one dialect (see Context).
3. Neither → passthrough. A sixth dialect falls through honestly rather than being
   mis-split.

**`starts_in_thinking` is a required input, not an optimisation.** When the template opens
the block in the *prompt* (Qwen3.8, GLM-4.7), the completion carries only the closing
marker, and a `<think>(.*?)</think>` regex never fires. The delimited reader must be told
whether the prompt already opened the block. Upstream hit the same class twice and named
the signal `prompt_has_open_thinking` (Blaizzy/mlx-vlm#1811, #1912).

**Rejected:** harmonising all shapes into one marker dialect inside `content`, as an
alternative to the field. It removes the API change but fabricates tokens the model never
emitted — indistinguishable from the outside, lossy (Harmony's `commentary` channel has no
delimiter equivalent), not round-trippable, and it still leaves the client parsing markers,
just fewer of them.

---

## Decision 3 (CLOSED) — strip display, do not suppress generation

`--no-reasoning` and any future request flag **strip display only**. Generation-side
suppression is a separate, separately-named capability (see Non-Goals).

The June lean was "strip-only, to avoid one flag with two meanings". Measurement shows it
would carry **three**:

| | `enable_thinking` | what "suppress where advertised" would do |
|---|---|---|
| Qwen3.8 / GLM-4.7 | present, default **on** | suppresses |
| gemma-4-e4b | present, default **off** (opt-in) | **nothing** — already off |
| gpt-oss | absent (only `reasoning_effort`) | **nothing** — no such control |

And the display use case does not need it: a client that renders the trace collapsed lets
the user expand or ignore it, with no server involvement. Suppression is a token-cost
optimisation, not a rendering requirement, and it is the only part of this ADR where the
five families and their two axes matter at all.

---

## Decision 4 — do not override the chat template's default

When the caller expresses no preference, mlx-knife renders the prompt with the template's
**own** default. It does not inherit `mlx_lm`'s forced `True` nor `mlx_vlm`'s forced
`False`.

This is not a new capability; it is the removal of an unintended one. Today mlx-knife
silently requests maximum reasoning on the text path — Qwen3.8's template applies
`reasoning_effort|default('xhigh')` when thinking is enabled — and silently disables it on
the vision path. On `gemma-4`, whose template defaults to **off**, the text path *turns
reasoning on that the model author left off*.

Implementation note: `TokenizerWrapper.apply_chat_template` fills the kwarg only when the
key is **absent**, so "pass nothing" is not reachable through the wrapper. Reaching the
template's own default requires rendering outside it. This is why Decision 4 is part of
this ADR and not a one-line fix.

**Non-goal here:** exposing the controls (`enable_thinking`, `reasoning_effort`) as a
request parameter, and reporting per model which of them are honourable. That belongs with
the capability surface (#51) — without it, a UI offers a switch that silently does nothing
on two of the five families measured.

---

## Decision 5 — the runner delivers raw; segmentation sits beside it

Today the segmentation is **fused into the runner** (`core/runner/__init__.py:461-465`,
`:814`), so every consumer inherits the CLI's formatting whether it wants it or not — which
is how terminal markdown reached the API field.

| Layer | Responsibility |
|---|---|
| `ModelRunner.generate_batch` / `generate_streaming` | return **raw** — what the model emitted, unchanged |
| segmenter (Decision 2), callable, not embedded | `raw → (reasoning, answer)` |
| `operations/run.py` | raw by default; `--no-reasoning` calls the segmenter |
| `serve` | calls the segmenter, fills `content` + the two reasoning fields |

The same applies per backend: `VisionRunner` (mlx-vlm) is a second path with its own raw
output. The rule is per path, and there are two.

---

## Decision 6 — acceptance: the model is the oracle

Because the runner returns raw and `run` shows it unchanged, the CLI output **is** the
reference for what the model emitted. That yields an acceptance property with no stored
ground truth:

> For the same prompt at `--temperature 0`, the server's `content` and `reasoning_content`
> must **reconstruct** the CLI output exactly, modulo the separator this ADR defines.
> If they do not, the segmenter is wrong.

Checkable per portfolio model, no fixture to maintain, and it fails for the right reason
today: Decision 4's defect means CLI and server do not even build the same prompt.

The property is void if any consumer reformats — which is the second reason the gpt-oss
formatter is retracted below. An oracle with one exception is not an oracle.

---

## CLI surface

`mlxk run` prints the model's output **verbatim** by default. `--no-reasoning` invokes the
segmenter and prints the answer only.

This inverts the June scope, which proposed rendering reasoning "visually distinct by
default (e.g. dim, prefixed)". Three reasons:

- **The inline `**[Reasoning]**` / `**[Answer]**` markdown was built for
  markdown-rendering frontends**, not for a terminal. That consumer is served properly by
  Decision 1; the CLI was standing in for it.
- **`--no-reasoning` becomes loud instead of silent.** When raw is the default and the
  segmenter runs only on request, a failure is visible: you asked to hide, the trace is
  still there. Today it fails silently, because handled and unhandled look identical.
- **Verbatim streams trivially**; segmentation requires buffering.

### The gpt-oss formatter is retracted as a failed attempt

`core/runner/reasoning_format.py`'s formatting branch, the `skip_tokens` /
`conditional_skip` machinery (`core/reasoning.py:30`, `:32`, `:325-390`) and the
`stop_tokens.py:103-104` gpt-oss special case are **deleted**, not repaired.

Deleted with them: `ReasoningExtractor.format_for_display()`
(`core/reasoning.py:139-160`), which has **no caller anywhere in the tree or the tests**
and carries a *third* rendering style (`═══ Reasoning ═══`) beside the two live ones. It is
the origin of a documentation error — ADR-014's CLI-symmetry table proposed a
`--show-reasoning` flag that never existed, evidently read off this dead function's
parameter. Three display formats for one concept, one of them unreachable, is the same
improvisation signature.

Grounds for the retraction:

- it fires on a **name match**, so it also claims QwQ, where its regexes cannot match and
  it adds a stop token the model does not have;
- it does its one job unreliably — the channel segmentation over-collects, producing
  duplicated reasoning under some prompts;
- it needs four skip tokens, a conditional skip and a dedicated stop token — the
  complexity signature of fighting a format rather than reading it;
- it is the only formatting in the product, making one of five families look different for
  no principled reason;
- and it voids Decision 6 for exactly that family.

Accepted cost: raw Harmony (`<|channel|>analysis<|message|>…`) in a terminal when no flag
is given. In exchange, gpt-oss gets a *working* answer-only mode for the first time, and
the pretty rendering moves to the client where it belongs.

---

## Non-Goals

- **Generation-side suppression** — a separate capability, separately named, gated on the
  control-reporting surface (#51). Decision 3 closes the flag semantics; it does not open
  this.
- **System-prompt-based activation** — #33, orthogonal, unblocked from this work.
- **Function/tool-call reasoning traces** — #39 / ADR-027. Harmony's `commentary` channel
  is recognised by the reader but not interpreted here.
- **User-defined custom tags** — markers come from the model, not user injection.
- **Generation-length / repetition heuristics** — explicitly rejected as the lever.
- **OpenAI Responses API** — different endpoint, not served.

---

## Acceptance anchor

1. **Reconstruction** (Decision 6) across the portfolio's reasoning models, covering both
   shapes: `Qwen3.8` / `GLM-4.7` (delimited, prompt-opened), `gemma-4-e4b` (delimited,
   unusual markers), `gpt-oss-20b` (channel-selected), `QwQ` (delimited, the former false
   positive).
2. **Prompt parity** — text and vision paths render the same thinking discipline for the
   same model; neither overrides the template default (Decision 4). Offline, tokenizer and
   processor only, no model load.
3. **Field discipline** — both names present with the same value on batch and SSE; field
   absent, not empty, when segmentation did not run.
4. **Passthrough honesty** — a model with an unrecognised shape yields clean `content` and
   no reasoning field, and `--no-reasoning` visibly does nothing rather than silently
   claiming to have worked.
