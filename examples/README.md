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

**Status legend**
- ✅ **Runnable** — runs today against the named released mlxk.
- 🟡 **Preview** — code is present but depends on an unreleased feature.
- 📋 **Planned** — use-case is defined; code lands with the feature it needs.

---

## Conventions (for adding an example)

1. **One example = one self-contained subdirectory** with its own `README.md`.
2. Each example README opens with a metadata block:

   ```
   **Status:** ✅ Runnable | 🟡 Preview | 📋 Planned
   **Runnable against:** <released mlxk version, or the unreleased feature it needs>
   **Requires:** <dependencies, env flags>
   **Run:** <a one-line invocation, or "not yet" for planned>
   ```

3. **Private data stays out of the tree.** Inputs (photos, documents) are read
   from an external directory and outputs are written to an external directory —
   never into `examples/`. `examples/.gitignore` is only a defensive backstop.
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

---

## License

Same as MLX Knife (Apache 2.0).
