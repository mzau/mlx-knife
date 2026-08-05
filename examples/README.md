# MLX Knife — Examples

Validated, real-world use cases for consumers of mlx-knife. Each example shows
how to *use* mlx-knife from the outside — CLI pipes, RAG, photo workflows — not
how mlx-knife is built internally.

These are best-effort consumer demos, kept deliberately small and principled.
Some drift is expected; each example states what it is runnable against.

---

## Catalog

| Example | Status | Runnable against | Shows |
|---------|--------|------------------|-------|
| [pipes/](pipes/)         | ✅ Runnable | mlxk ≥ 2.0.4 (pipe beta) | Broadcast stdin to N models in parallel; vision → text archive pipeline |
| [model-routing/](model-routing/) | ✅ Runnable | mlxk 2.0.6 (`list --json` capabilities) | Single-node task → model selection (POC for broke-cluster's model-routing dimension) |
| [rag-server/](rag-server/) | ✅ Runnable | mlxk ≥ 2.0.7 (`embed` experimental, alpha-gated) | Pipe-based RAG toolbox + OpenAI-compatible RAG server |
| [photo-rag/](photo-rag/) | ✅ Runnable | mlxk ≥ 2.0.7 (`mlxk serve` vision; `embed` alpha-gated) | A family album made searchable, with nothing leaving the machine; resumable multi-day batch |

**Status legend**
- ✅ **Runnable** — runs today against the named released mlxk.
- 🟡 **Preview** — code is present but depends on an unreleased feature.
- 📋 **Planned** — use-case is defined; code lands with the feature it needs.

---

## Conventions (for adding an example)

1. **One example = one self-contained subdirectory** with its own `README.md`. An example whose
   README has grown past being readable in one sitting may carry **one** companion document
   beside it, and no more — `photo-rag/` is the precedent: `README.md` is the way in and stays
   short, `MANUAL.md` holds everything past the first run. One companion, never a
   documentation tree.
2. Each example README opens with a metadata block:

   ```
   **Status:** ✅ Runnable | 🟡 Preview | 📋 Planned
   **Runnable against:** <released mlxk version, or the unreleased feature it needs>
   **Requires:** <dependencies, env flags>
   **Run:** <a one-line invocation, or "not yet" for planned>
   ```

3. **Private data stays out of the tree.** Inputs (photos, documents) are read
   from an external directory and outputs are written to an external directory —
   never into `examples/`. There is no in-tree backstop and none is relied on:
   the rule is that nothing private is ever *in* the tree to begin with, so
   external paths are required rather than defaulted.

   Deliberately published sample material is the other case and is not an
   exception to this: it is chosen to be publishable, lives under `tests_2.0/assets/`
   with the rest of the fixtures, and an example must be built so that a reader's
   own data never joins it.
4. **English only.** Track files with explicit paths (never `git add -A`).
5. Examples are not part of mlxk core and carry no stability guarantee.

---

## Release coupling

Examples are published as their underlying feature ships:

- **pipes/** rides the public Unix Pipe Integration (beta since 2.0.4).
- **model-routing/** dogfoods the capability Contract (`list --json` +
  `runtime_compatible`, future Capabilities-API #51) and is a POC for
  broke-cluster's model-routing dimension.
- **rag-server/** is live as of 2.0.7 — `mlxk embed` ships experimental
  (alpha-gated); the embeddings work doubled as its dogfooding / acceptance artifact.
- **photo-rag/** rides nothing. It consumes released 2.0.7 and needs **no change to
  mlx-knife**, so it is coupled to no later release and waits for none. Its
  `geo-test-run.py` grades the whole pipeline against the photographs shipped with the
  project, which is how that claim stays checkable rather than asserted.

---

## License

Same as MLX Knife (Apache 2.0).
