# Model Routing — single-node task → model selection (POC)

Pick the right local model for a request, using mlx-knife's capability
self-description. A standalone consumer of `mlxk list --json`.

**Status:** ✅ Runnable
**Runnable against:** mlxk 2.0.6 (JSON-API ≥ 0.1.5: `capabilities` + `runtime_compatible`)
**Requires:** `mlxk` on PATH with a populated model cache; Python 3.8+, stdlib only
**Run:** `./route.py "Explain quicksort"` → prints the chosen model name

> This is a **POC for broke-cluster**, not broke's architecture. It prototypes the
> *model-routing* half of routing as a consumer example. The productized,
> cross-node version belongs in broke-cluster — see "Boundary" below.

---

## What problem this is (and isn't)

broke-cluster's `Prompt → {node, model}` hides **two separable dimensions**:

1. **Node routing** — "which *node* has model X?" (cross-node; always broke's job)
2. **Model routing** — "which *model* for this task?" (the task→model decision)

**Special case: a single control node and nothing else.** Then dimension 1
collapses to null — there is no other node to route to. Only **model routing**
remains, and *that* is what this example implements. It needs no cluster, no
commissioning, no SSH, no daemon — just one mlx-knife node.

So this is the cheapest usable slice of broke functionality you can run *today*,
while broke-cluster itself is still being defined.

---

## How it decides (capability-driven, not learned)

Selection consumes the Contract (`mlxk list --json`) and chooses only from models
that are **both** capability-matching **and** `runtime_compatible: true` — the
*declared ∩ runnable* rule. A model that declares a capability but cannot run is
reported, never executed.

| Signal | Source | Determinism |
|--------|--------|-------------|
| Modality (`--image` → `vision`, `--audio` → `audio`) | the request itself | deterministic |
| Task category (`--task coding`) → capability + preferred model | `routing.json` | declarative |
| Pinned `--model` | the request | passthrough, routing skipped |

**Deliberately NOT here:** inferring intent ("is this a coding prompt?") from the
prompt text. That is a learned-classifier problem, and broke-cluster already hit
that wall and exiled it to a separate project (its `adr-016-ml-classifier-exodus`).
This POC stays capability + declarative-default only. Keep it that way.

---

## Usage

```bash
./route.py "Explain quicksort"                  # chat (default)
./route.py --task coding "write a bubble sort"  # task category from routing.json
./route.py --image photo.jpg "what is this?"    # requires 'vision'
./route.py --audio clip.wav  "transcribe this"  # requires 'audio'
./route.py --model org/Some-Model "hi"          # pinned: passthrough, no routing
./route.py --json "hi"                          # full decision as JSON
./route.py --task coding --exec "..."           # route, then run via `mlxk run`
```

The chosen model name goes to **stdout** (reasoning to stderr), so it composes:

```bash
mlxk run "$(./route.py --task coding 'write a bubble sort')" "write a bubble sort"
```

Edit `routing.json` `defaults` with model names from your own `mlxk list` to make
selection deterministic per category; otherwise the first runnable
capability-matching model is used.

---

## The `declared ∩ runnable` reality (why this is honest)

On a real cache, declaring a capability is not the same as being able to run it.
Live example output for `--image`:

```
[route] task=vision capability=vision -> mlx-community/pixtral-12b-4bit
[route] picked first runnable 'vision' model.
[route]   declared-but-not-runnable: ...Llama-3.2-11B-Vision... (Model type mllama not supported.)
[route]   declared-but-not-runnable: ...gemma-3n-E2B-it-4bit  (Index/shard mismatch (#624) — Fix: mlxk convert --repair-index)
[route]   declared-but-not-runnable: ...nanoLLaVA-1.5-8bit    (missing preprocessor_config.json)
```

7 models declare `vision`; only 1 is runnable. The example routes to the runnable
one and reports the rest *with mlx-knife's own reasons* (including repair hints).
This is the point: a model-router must trust `runtime_compatible`, not just
`capabilities`.

---

## Why this lives here: it dogfoods mlx-knife's Contract

To route by task you must consume mlx-knife's capability self-description. Building
this POC is the cheapest way to find out **what that Contract must expose** — i.e.
it dogfoods the capability surface (the JSON-API `capabilities` + `runtime_compatible`,
and the future Capabilities-API, Issue #51). Things this example makes concrete:

- `capabilities` is coarse today (`text-generation`/`chat`/`vision`/`audio`/`embeddings`).
  STT vs. audio-generation, or "coding-strong", are **not** distinguishable from
  the Contract — they have to come from `routing.json`, not from mlx-knife.
- `runtime_compatible` + `reason` are exactly what a consumer needs to avoid
  routing to a broken model. This example is a live test that they suffice.

Per the authoritative list, see `mlxk2/core/capabilities.py` `Capability` enum.

---

## Boundary (so this stays a POC, not a broke spec)

mlx-knife describes and executes *itself*; it never decides about other nodes.
This example is a **consumer** of that self-description — the intelligence lives in
`route.py`, not in mlx-knife. What it intentionally leaves to broke-cluster:

- **node routing / location-routing** (which node has the model)
- **commissioning** (deploying models to worker nodes, `scp`/SSH)
- **the scheduler-brain** (complexity → tier, historical performance, learned routing)

When broke-cluster is defined, the productized cross-node version moves there and
wraps this model-routing slice with node routing + commissioning. This example
*informs* that design; it does not specify it.

---

## License

Same as MLX Knife (Apache 2.0).
