# Pipes — Unix-style CLI workflows

Pipe-based workflows that compose `mlx-run` with standard Unix tools.

**Status:** ✅ Runnable
**Runnable against:** mlxk ≥ 2.0.4 (Unix Pipe Integration, beta)
**Requires:** `export MLXK2_ENABLE_PIPES=1`; one or more local models in your HF cache
**Run:** see per-script usage below

> These are best-effort consumer demos, not part of mlxk core. Some drift is
> expected as the pipe surface evolves.

---

## `mlx-tee.py` — Broadcast stdin to multiple commands in parallel

Send the same prompt to several models (or machines) at once and collect the
outputs in stable order.

```bash
export MLXK2_ENABLE_PIPES=1

# Compare two local models
echo "Explain quicksort" | python mlx-tee.py \
    "mlx-run phi-3 -" \
    "mlx-run llama-3 -"

# Distribute across machines (SSH placeholder)
echo "Explain quicksort" | python mlx-tee.py \
    "@mac_mini:mlx-run phi-3 -" \
    "@mac_studio:mlx-run llama-3 -"
```

**Validation level:** the broadcast/fan-out mechanism is smoke-tested
(`echo … | python mlx-tee.py "cat" "tr a-z A-Z"` returns both branches in order).
The model leg uses mlxk's public `mlx-run <model> -` stdin support — substitute
models you actually have cached.

**Note:** Remote execution uses SSH as a placeholder. For production distributed
workflows without SSH complexity, see
[broke-cluster](https://github.com/mzau/broke-cluster).

---

## `vision_pipe.sh` — Vision → text archive pipeline

Caption a folder of images one at a time (Metal-safe loop), assign stable
`Image N` numbers, then pipe the captions into a chat-only model for synthesis
(table, tags, summary).

```bash
export MLXK2_ENABLE_PIPES=1
VISION_MODEL=pixtral TEXT_MODEL=Qwen3-Next ./vision_pipe.sh /path/to/photos/*.jpeg
```

**Privacy:** point this at photos that live **outside** the project tree. The
script writes only to a `mktemp` scratch file that is cleaned up on exit.

**Validation level:** all dependencies are public (pipe beta + vision beta);
run it against your own vision model and images. With ADR-012 Phase 1c (CLI
internal batch processing) the per-image loop can collapse into a single
`mlxk run --image …` invocation while keeping global Image numbering.
