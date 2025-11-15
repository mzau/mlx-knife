# ADR-012 — Vision / Multimodal Support Roadmap (Draft)

- **Status:** Draft (internal evaluation)
- **Authors:** mlx-knife maintainers
- **Date:** 2024-11-11

## Context

Community feedback (e.g., Reddit comments comparing `mlx_lm.server + llama swap`) highlights the lack of multimodal support in MLX-Knife 2.0. The 2.0 CLI deliberately focused on LLM lifecycle management (cache, health, server, JSON automation). Before publicly committing to a multimodal feature set we need to document what “vision support” actually means for mlx-knife and how it fits with existing abstractions.

## Goals

1. Deliver an alpha-level `mlxk run` flow that accepts still images (one-shot prompt or chat) without breaking current UX patterns.
2. Extend JSON API schemas so callers can submit/receive multimodal payloads predictably once the runner core is solid.
3. Keep HF cache semantics identical to text-only models (no bespoke asset folders) and rely on upstream `mlx_lm` vision support.
4. Preserve CLI-first ergonomics—no GUI dependencies; the sample Web client only mirrors server capabilities.

## Proposed Phases

### Phase 0 — Research & Scoping
- Inventory MLX models on Hugging Face that expose vision/multimodal configs.
- Validate mlx-lm backend hooks for image preprocessing (tokenizer, pixel transforms).
- Capture UX expectations from comparable tools (Ollama `/api/chat` with `images`, OpenAI `responses` API).

### Phase 1 — `mlxk run` Alpha (highest priority)
- Implement `--image <path>` (repeatable) for `mlxk run` one-shot prompts. Optional text input keeps behaving exactly like today.
- Inject a default system prompt (e.g., “You are a vision assistant. Describe the image.”) when the user supplies only `--image` so models never silently no-op.
- Define chat semantics: the first turn may include both the image and text; follow-up turns stay text-only but retain the original image context.
- Allow pure image invocations to respond immediately (default description) instead of forcing batch mode, unless other batch flags are set.

### Phase 2 — Runner Hardening & Metadata
- Wire preprocessing (resize/normalize) into `mlxk2/operations/run.py` in a model-agnostic way so upgrades track upstream `mlx_lm`.
- Emit capability metadata (`capabilities: ["text","vision-understanding"]`) in `mlxk list --json` and `mlxk show` so automation can distinguish “image understanding” from text-only or potential future text→image cases.
- Update health checks to verify auxiliary assets (tokenizers, image processors) are present in the HF snapshot.
- Extend `docs/json-api-specification.md` with alpha notes for `inputs.images[]`, the “image-on-first-turn” contract, and explicit output modality indicators (text vs future image generation).

### Phase 3 — Server & Web Hooks
- Teach `mlxk serve` the OpenAI-style `content` format (`[{type:"input_text"}, {type:"input_image", image_url:{...}}]`) so off-the-shelf OpenAI clients and connectors can upload base64 images.
- Introduce optional text-file uploads first (low-risk rehearsal) to validate multipart handling, temp storage, and streaming responses before turning on images.
- Update the sample Web client to support drag-and-drop text files (and later, images) without adding GUI dependencies to the CLI core.

### Phase 4 — Tooling & Docs
- Document workflows in README (new “Vision Models” section) and add regression tests under `tests_2.0/vision/`.
- Provide sample HF models + scripted downloads for contributors.
- Evaluate need for GPU/ANE configuration toggles specific to large pixel encoders.

## Non-Goals

- Shipping a GUI or image annotation tool.
- Supporting arbitrary video/audio inputs in the same release (separate ADR if needed).

## Open Questions

1. Do we gate vision support behind `MLXK2_ENABLE_ALPHA_FEATURES`, or carve out a dedicated `MLXK2_ENABLE_ALPHA_VISION=1` gate so releases can graduate independently?
2. How do we represent mixed-mode outputs (e.g., returning base64 images) in the JSON schema without breaking OpenAI compatibility?
3. Should we expose preprocessing hooks so advanced users can override transforms, or keep the runner opinionated?
4. When (if ever) do we allow mid-chat image updates, or do we enforce “image only on first turn” permanently?

## Next Steps

1. Complete Phase 0 (catalog MLX vision-ready models, verify `mlx_lm` support status).
2. Implement Phase 1 + 2 behind `MLXK2_ENABLE_ALPHA_VISION` and cut a 2.0.3 alpha release focused on end-user feedback.
3. Collect real-world reports (CLI-only) to validate preprocessing, default prompts, and capability metadata.
4. Schedule Phase 3 (server + web hooks) as a 2.0.4/2.0.5 beta milestone once alpha feedback is digested.
5. Promote to 2.1 stable after Phase 4 (docs/tests) and when the env flag can be removed confidently.
