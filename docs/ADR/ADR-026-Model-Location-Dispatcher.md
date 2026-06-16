# ADR-026: Unified Model Resolution — a Model-Location-Dispatcher

**Status:** Draft (provisional — captures the direction; decision not yet finalized)
**Created:** 2026-06-16
**Related:** ADR-022 (Workspace-First Paradigm — *parent*), ADR-018 Phase 0c (workspace-path support in operations — *superseded*), ADR-012 / ADR-016 / ADR-020 / ADR-024 (*consumers*), ADR-015 §Code-Structure (run.py size tech-debt), ADR-025 (content_hash identity)
**Target:** 2.1 (NOT 2.0.7)

---

> **Provisional note.** This ADR is a Draft capturing insights surfaced during ADR-015 Slice C
> (embeddings capability honesty, 2026-06-16). It records the *direction* so it is canonical in the
> repo; the concrete design + decision happen in the dedicated 2.1 refactor session. Nothing here is
> implemented yet beyond the interim seed noted below.

## Context

The workspace-vs-cache **location resolution** is re-implemented across operations rather than
living behind one dispatcher. There is no single `model_spec → (model_path, cfg, framework,
identity, is_workspace)` handle that all verbs + the server consume. Scattered sites:

- `operations/run.py::run_model` — inline cache-dir math (`models--*/snapshots/<hash>`) + a separate
  workspace branch; the cache branch additionally carries a fail-fast pre-flight the workspace
  branch lacks.
- `operations/common.py::build_model_object` — branches internally on `is_workspace_path` for health
  (`health_check_workspace` vs `is_model_healthy`) and workspace metadata.
- `core/server/handlers/models.py::handle_list_models` — **two** discovery loops (workspace scan +
  cache scan) doing nearly the same per-model build + filter.
- Loose resolvers: `resolve_model_for_operation`, `resolve_model_dir`, `model_display_identity`,
  scattered `is_workspace_path` checks. Partial seams exist; no uniform handle.

**How this surfaced (ADR-015 Slice C):** Slice C made config-first encoder embedders (`model_type:
bert`, e.g. bge) classify as `embedding` and surface as runnable. `mlxk run` on such a *cache*
model returned mlx-lm's cryptic *"Model type bert not supported"* instead of the honest *"is an
embedding model — use `mlxk embed`"* — because the cache branch's early `check_runtime_compatibility`
ran *before* the path-agnostic Class-A reject and shadowed it. The workspace branch has no such
early check, so it was already honest. **The bug was an artifact of the location-path asymmetry, not
of capability logic.**

## Decision (proposed)

Introduce a single **model-location-dispatcher**: resolve `model_spec` once into a uniform handle
`(resolved_name, model_path, cfg, framework, identity, is_workspace, commit_hash)`. All downstream
per-model logic — capability detection + rejects (ADR-024), memory gates (ADR-016), audio backend
routing (ADR-020), vision health (ADR-012), identity stamping (ADR-025) — runs **once on the
handle**, path-agnostically. Operations and the server consume the handle instead of each
re-deriving location.

This is a **resolution substrate (its own axis)**, distinct from capability honesty (ADR-024, a
*consumer*). Dependency shape: **parent** ADR-022 (which created the workspace/cache duality);
**supersedes** the scattered per-op branching added by ADR-018 Phase 0c; **consumers** ADR-012 /
016 / 020 / 024 all run on the handle.

**Same initiative as the `run.py` size-refactor** (the "run.py ~850 LOC accreted by inlining
handlers" tech-debt, ADR-015 §Code-Structure): the clean way to split `run.py` is to extract
exactly the dispatcher + per-modality handlers. So they are done together, in 2.1 — not piecemeal.

## Scope & non-goals

- **NOT 2.0.7.** Cross-cutting (touches every operation + server), deserves its own migration; doing
  it inside 2.0.7 would reopen the scope explosion the maintainer deliberately avoided. 2.0.7 ships
  solid tests via *local* unifications + workspace-first test conventions.
- **Seed (interim, shipped in Slice C):** `run.py`'s text/LLM pre-flight skips embedding models so
  they reach the honest Class-A reject — a localized fix, the first concrete instance of the
  path-uniform direction this ADR generalizes.

## Open questions

- Two resolution worlds — "operate on one named model" (run/embed) vs "scan/list all"
  (list/show/serve). Unifying *within* each is more tractable than a grand unification; decide
  whether one dispatcher serves both or they share a lower-level locator.
- Migration order and how much of the per-modality pre-flight (vision memory gate, audio routing)
  moves behind the handle vs stays modality-specific.
- Whether this lands as its own ADR or folds into an ADR-022 revision.
