# Changelog

## [2.0.4-beta.6] - 2026-01-07

### Highlights

**Complete Local Development Workflow:** Beta.6 completes the workspace story started in beta.5. Full local development cycle without HuggingFace round-trips:
1. Clone a model to a local workspace (`mlxk clone`)
2. Repair or modify it locally (`mlxk convert --repair-index` fixes index/shard mismatches)
3. Use it directly without pushing:
   - Run inference: `mlxk run ./workspace "prompt"`
   - Inspect metadata: `mlxk show ./workspace`
   - Start a dev server: `mlxk server --model ./workspace`

Enables model experimentation, repair (mlx-vlm #624 affected models), and testing before publishing to HuggingFace.

**Vision Batch Processing:** Production-ready vision processing with automatic chunking for large image collections. Default: process one image at a time for maximum stability (prevents Metal OOM + model-specific hallucination). Server now accepts unlimited images with safe chunking. Complete Vision→Text pipe workflow support. Metadata now appears before model output (vision models can reference filenames, GPS, dates in their analysis).

**Enhanced Reliability:** Resumable clone operations with `--force-resume`, deterministic temp cache naming for better debugging, and improved test architecture (umbrella marker convention) for multi-Python compatibility (3.9-3.14). 161 Wet Umbrella tests passing (includes new Vision→Geo pipe integration tests).

---

### Added

- **Workspace Operation Support (ADR-018 Phase 0c):**
  - Complete local workflow: `clone → convert → run/show/server` without HF push
  - `mlxk run ./workspace "prompt"` - Direct inference on workspace paths
  - `mlxk show ./workspace` - Metadata inspection for workspaces
  - `mlxk server --model ./workspace` - Local dev server with workspace models
  - Server `/v1/models` endpoint lists workspaces with `"owned_by": "workspace"`
  - MLXRunner + VisionRunner now support workspace paths
  - Central implementation via `resolve_model_for_operation()` + `is_workspace_path()`
  - Files changed: 11 files (~250 LOC)
  - See ADR-018.md Phase 0c for details

- **Path Resolution Convention (README.md):**
  - `model-name` → Cache resolution (expands to `mlx-community/...`)
  - `./model-name` → Workspace path (local directory)
  - `/abs/path` → Workspace path (absolute)
  - `.` or `..` → Workspace path (current/parent directory)
  - Explicit path prefixes required to treat local directories as workspaces
  - Prevents ambiguity when workspace name matches cache model name

- **Health command workspace support:**
  - `mlxk health ./workspace` - Integrity checks for workspace directories
  - Returns minimal format per JSON API Schema 0.1.5: `{name, status, reason}`
  - Uses same integrity checks as list/show commands (consistency)

- **Resumable Clone with --force-resume (ADR-018 Phase 0b):**
  - `mlxk clone <model> <workspace> --force-resume` - Skip confirmation prompt for partial downloads
  - Deterministic temp cache naming: `.mlxk2_temp_{hash(model+target)}` (SHA256-based, replaces PID random names)
  - Resume detection: Verifies health + download completion marker before resuming
  - Conditional cleanup: Keeps complete downloads on failure (debugging/retry), cleanup incomplete
  - Helper functions: `_get_deterministic_temp_cache_name()`, `_create_temp_cache_same_volume()`
  - Files: `operations/clone.py` (~100 LOC)

- **Portfolio Discovery Blacklist (Testing Infrastructure):**
  - `KNOWN_BROKEN_MODELS` config in `tests_2.0/live/test_utils.py`
  - Filters models with upstream runtime bugs (pass static health checks, fail at runtime)
  - Policy: Add only when static health ✅ but runtime initialization ❌ due to verified upstream bug
  - First entry: `mlx-community/MiMo-VL-7B-RL-bf16` (mlx-vlm NoneType iteration bug)
  - Portfolio Discovery now filters against blacklist (27 → 26 models)
  - Files: `tests_2.0/live/test_utils.py` (+29 LOC blacklist config)

- **Vision Batch Processing (ADR-012 Phase 1c, #47):**
  - Automatic chunking for large image collections (prevents Metal OOM crashes)
  - CLI: `mlxk run model --image *.jpg --chunk N` (N = images per batch)
  - Server: `POST /v1/chat/completions {"chunk": N, ...}` + `mlxk serve --chunk N`
  - Model loaded once per chunk (isolation guarantees no state leakage between batches)
  - Global image numbering preserved (Image 1..N across all chunks)
  - Batch context markers in prompt: `[Processing batch 1/2: Images 1-4 (chunk_size=4, 8 total)]`
  - Collapsible batch info in output: `📸 Batch 1/2: Images 1-4` (WebUI-friendly)
  - Environment variable: `MLXK2_VISION_BATCH_SIZE` sets default chunk size
  - Server unlimited images: Removed MAX_IMAGES_PER_REQUEST limit, added MAX_SAFE_CHUNK_SIZE=5 validation
  - Chunk size validation: CLI + Server validate against MAX_SAFE_CHUNK_SIZE (HTTP 400 if chunk>5)
  - Files: `cli.py`, `operations/run.py`, `operations/serve.py`, `core/server_base.py`, `core/vision_runner.py`, `tools/vision_adapter.py` (~200 LOC)

- **Vision Chunk Isolation (Prevents State Leakage):**
  - Fresh VisionRunner created per chunk (no KV-cache or model state persistence)
  - Fixes chunk hallucination bug where model "remembered" images from previous batches
  - Trade-off: Model load overhead (~2-3s per chunk) vs guaranteed isolation
  - Alternative approach deferred: Model reuse + cache clearing (2.0.5-beta, requires mlx-vlm API support)

- **Flexible Prompt Argument Order (UX Improvement):**
  - Added `--prompt` flag as alternative to positional argument
  - Use case: Test different prompts with same parameters: `mlxk run model --image *.jpg --chunk 5 --prompt "prompt"`
  - Solves argparse limitation where `--image` with `nargs='+'` consumes following arguments
  - Both forms supported: `mlxk run model "prompt" --image file.jpg` OR `mlxk run model --image file.jpg --prompt "prompt"`
  - Backward compatible: Positional prompt still works when placed before flags
  - Pipe mode integration: `mlx-run pixtral ... | mlx-run qwen - --prompt "..."` combines stdin with additional prompt
  - File: `cli.py` (+10 LOC)

- **Vision→Geo Pipe Integration Tests (Wet Umbrella Phase 4):**
  - Comprehensive smoke tests for Vision→Text pipe workflows
  - Test suite: `tests_2.0/live/test_pipe_vision_geo.py` (3 tests, marker: `live_vision_pipe`)
  - **Test 1:** Vision batch processing with `--chunk 1` (validates ADR-012 Phase 1c)
  - **Test 2:** Complete Vision→Geo pipe (validates stdin + --prompt fix)
  - **Test 3:** Chunk isolation (validates no state leakage between chunks)
  - Test assets: `tests_2.0/assets/geo-test/` (9 publishable Stockholm photos with EXIF)
  - Model selection: Hardcoded pixtral (vision) + Portfolio discovery for text model (largest eligible)
  - Wet Umbrella integration: 161 total tests passed (Phase 1: 152, Phase 2: 3, Phase 3: 3, Phase 4: 3)
  - Graceful skip: Tests skip cleanly if vision/text models unavailable
  - **NOT a quality benchmark:** Pure workflow validation (pass = exit 0, no GOLD reference)

- **Test Infrastructure Improvements (Umbrella Marker Convention):**
  - Added `pytest.mark.live` umbrella marker for scalable test exclusion
  - All 11 live test files decorated with umbrella + specific markers
  - `test-multi-python.sh` simplified: `pytest -m "not live"` (future-proof)
  - Documentation: TESTING-DETAILS.md "Writing New Live Tests: Umbrella Marker Convention"
  - Rationale: Blacklist pattern (excluding individual markers) doesn't scale with new test categories

### Changed

- **BREAKING: Vision processing now defaults to processing one image at a time (ADR-012 Phase 1c, #47)**
  - Previous: All images processed in single batch (could OOM crash with many images)
  - New: Process one image at a time by default (maximum safety on all systems)
  - Override with `--chunk N` (CLI) or `"chunk": N` (API) for faster batching when system can handle it
  - Rationale: Safety over speed - prevents Metal OOM + model-specific hallucination with large batches
  - **Important:** Chunk isolation (fresh VisionRunner per chunk) prevents state leakage but adds ~2-3s model load overhead per chunk
  - Migration: Users with large image workflows should use `--chunk 5` or higher for old behavior
  - Environment variable: `MLXK2_VISION_BATCH_SIZE=N` sets default (e.g., export `MLXK2_VISION_BATCH_SIZE=5`)
  - **Model-specific hallucination triggers:**
    - Larger chunks with global context hints (e.g., "8 total images" when batch only has 4)
    - Plural prompts mismatched to actual count (e.g., "describe these images" with chunk=1)
    - Default chunk=1 with singular prompts provides maximum robustness

- **Model metadata `cached` field:** Now `false` for workspace paths, `true` for cache models
  - Workspace paths are NOT in HuggingFace cache → `cached: false`
  - Cache models remain `cached: true`
  - Semantic distinction: cache-managed vs user-managed models

### Fixed

- **Vision model detection (False Negatives):**
  - Fixed: Models with `vision_config` dict but `skip_vision: true` incorrectly marked as non-vision
  - Root cause: `skip_vision` flag misinterpreted as "no vision support" (actually means "optional for text-only")
  - Solution: Presence of `vision_config` dict now reliably indicates vision capability
  - Impact: Mistral-Small 3.1, DeepSeek-OCR now correctly detected as vision models
  - File: `operations/common.py:detect_vision_capability()`

- **Health check improvements:**
  - **processor_config.json support:** Vision models can use EITHER `preprocessor_config.json` OR `processor_config.json`
    - Different models use different naming conventions (e.g., DeepSeek-OCR)
    - Previously: Only `preprocessor_config.json` accepted → false "unhealthy" status
    - Impact: More vision models pass health checks
  - **mlx-vlm #624 specific error message:** Index/shard mismatch now distinguished from incomplete downloads
    - Message: "Index/shard mismatch (mlx-vlm #624). Index references N shards but found M different files. Fix: mlxk convert <model> <output> --repair-index"
    - Previously: Generic "Missing weight shards" message without fix guidance
    - Impact: Users get actionable repair instructions
  - **File Integrity Definition documented:** 5-point definition in `_check_snapshot_health()` docstring
    - Clarifies: Required files, weight completeness, corruption markers, auxiliary assets
    - Emphasizes: Integrity (health) vs Runtime (runtime_compatible) separation
    - Ensures: Consistent checks across list/show/health commands
  - File: `operations/health.py`

- **Workspace health check bug:** `build_model_object()` now uses `health_check_workspace()` for workspace paths
  - Previously: Workspace paths triggered cache-based health check → "Model not in cache"
  - Now: Correct health reasons for workspaces (e.g., "Missing weight shards")
  - Impact: `mlxk show ./workspace` displays accurate health status

- **Path resolution priority:** Explicit path detection prevents cache bypass
  - Previously: Any local directory matching model name would bypass cache resolution
  - Now: Only paths with `./ ../ /` prefixes or `.` `..` are treated as workspaces
  - Impact: `mlxk show Mistral-Small` tries cache first (even if local dir exists)

- **Sentinel version fix:** Workspace metadata now uses dynamic version
  - Previously: Hardcoded `"mlxk_version": "2.0.4"` in clone/convert operations
  - Now: Dynamic import from `mlxk2.__init__.__version__`
  - Impact: Sentinel metadata automatically tracks current version
  - Files: `operations/clone.py`, `operations/convert.py`, `operations/workspace.py`

- **Clone operation: Ctrl-C now preserves temp cache for resume:**
  - Previously: KeyboardInterrupt during `mlxk clone` deleted partial downloads → resume impossible
  - Root cause: Cleanup logic checked `result["status"] == "success"` first (initial value)
  - Fix: `cancelled_by_user` flag + inner/outer KeyboardInterrupt handlers + correct cleanup order
  - Impact: Ctrl-C during clone preserves temp cache, user can resume operation
  - Files: `operations/clone.py` (lines 59, 139-143, 242-246, 279-302)

- **Clone operation: Partial downloads now resumable automatically:**
  - Previously: `--force-resume` only worked for "complete but unhealthy" models, not after Ctrl-C
  - Root cause: Partial downloads (no `.mlxk2_download_complete` marker) treated as non-resumable
  - Fix: `_check_temp_cache_resume()` now treats partial downloads as resumable
  - Impact: `mlxk clone` automatically resumes after Ctrl-C (HuggingFace Hub native resume)
  - Files: `operations/clone.py` (lines 393-424, 447-469)
  - Note: Resume validation overhead (checksum) takes 5-10 min for large models (HF Hub behavior)

- **Vision server scalability:**
  - Server now accepts unlimited images (removed 5-image hardcoded limit from beta.5)
  - Added safe chunk size validation (max 5 images per chunk for Metal stability)
  - Impact: Large image collections work with server API
  - Files: `tools/vision_adapter.py`, `operations/serve.py`, `core/server_base.py`

- **Vision metadata positioning:**
  - Metadata table now appears BEFORE model output (was: after in beta.5)
  - Key benefit: Vision model can reference metadata in its analysis (filename, GPS, date visible in prompt context)
  - Secondary benefit: Clearer association with output when processing multiple chunks
  - File: `core/vision_runner.py`

### Testing

- **Unit tests:** 550 passed, 56 skipped (0 failures)
  - New: `tests_2.0/test_model_resolution_workspace.py` (9 tests - workspace path resolution)
  - Extended: `tests_2.0/test_workspace_sentinel.py` (+32 tests - sentinel version, health checks)
  - Extended: `tests_2.0/test_clone_operation.py` (+10 tests - resumable clone, deterministic naming)
  - Coverage: workspace detection, path resolution, health checks, resumable clone, sentinel metadata

- **Vision pipe integration tests:** 3 new smoke tests for Vision→Text workflows
  - Test suite: `tests_2.0/live/test_pipe_vision_geo.py` (marker: `live_vision_pipe`)
  - Test assets: 9 publishable Stockholm photos with EXIF in `tests_2.0/assets/geo-test/`
  - Coverage: Batch processing (`--chunk 1`), complete pipe workflow, chunk isolation
  - Model selection: Hardcoded pixtral + portfolio discovery for text model
  - **Note:** Smoke tests only (exit 0 = pass), not quality benchmarks

- **Wet Umbrella test integration:** 161 tests total across 4 phases
  - Phase 1: 152 tests (live_e2e marker)
  - Phase 2: 3 tests (live_pull marker)
  - Phase 3: 3 tests (live_clone marker)
  - Phase 4: 3 tests (live_vision_pipe marker)
  - Single entry point: `scripts/test-wet-umbrella.sh`
  - RAM requirements: 64GB recommended (M2 Max tested), 32GB untested

- **Test marker architecture improvements:**
  - Added umbrella marker `pytest.mark.live` to all 11 live test files
  - Simplified exclusion: `pytest -m "not live"` (future-proof, scalable)
  - Old approach: Blacklist individual markers (doesn't scale with new test categories)
  - Updated: `test-multi-python.sh` uses umbrella marker for clean isolation
  - Documentation: TESTING-DETAILS.md "Writing New Live Tests: Umbrella Marker Convention"

- **Test marker refactoring:** `live_resumable` → `live_pull` for consistency
  - Renamed marker to match `live_push`, `live_list` pattern
  - Updated Wet Umbrella script: 4-Phase separation (wet, live_pull, live_clone, live_vision_pipe)
  - Files: `pytest.ini`, `conftest.py`, `scripts/test-wet-umbrella.sh`, `TESTING-DETAILS.md`

### Documentation

- **README.md:** Model reference path conventions documented
- **operations/health.py:** File Integrity Definition documented in `_check_snapshot_health()` docstring

---

## [2.0.4-beta.5] - 2025-12-31

### Added

- **Workspace Infrastructure (ADR-018 Phase 0a):**
  - Managed workspace detection via `.mlxk_workspace.json` sentinel
  - Workspace health checks support managed and unmanaged workspaces
  - All cloned workspaces now include provenance metadata
  - Backward compatible with unmanaged workspaces (pre-2.0.4)
  - See ADR-018.md Phase 0a for details

- **Convert Operation (ADR-018 Phase 1):**
  - `mlxk convert <src> <dst> --repair-index` rebuilds safetensors index from shards
  - Fixes mlx-vlm #624 affected models (7+ models: Qwen2.5-VL, gemma-3, Mistral-Small, etc.)
  - Cache sanctity: Hard blocks writes to HF cache
  - APFS CoW optimization for instant, space-efficient cloning
  - See ADR-018.md Phase 1 for details

- **Resumable Pull Support:**
  - Auto-detect partial downloads with user confirmation prompts
  - `--force-resume` flag for non-interactive automation
  - Health-aware logic for damaged downloads

- **Wet Umbrella Test Integration:**
  - Single entry point: `scripts/test-wet-umbrella.sh` runs all real model tests
  - Auto-marker assignment via pytest hook
  - Memory-optimized options prevent pytest RAM spikes
  - See TESTING-DETAILS.md for architecture

- **Vision Model Benchmark Reporting:**
  - Auto-reporting fixture adds Vision metadata to benchmark JSONL
  - Schema v0.2.0 compliance for memplot integration

### Changed

- **Clone operation:** Now produces managed workspaces (sentinel written after APFS clone, enables provenance detection)

### Fixed

- **Vision Portfolio Discovery:** Fixed KeyError when discovering Vision models. Issue: Vision models share `model_type="chat"` with TEXT. Solution: Filter by capabilities instead. Result: 3 Vision models now detected.

- **Mistral Tokenizer Bug (Issue #49):** Fixed BPE space markers `Ġ` (U+0120) appearing in output instead of spaces. Affected: 3+ Mistral-family models (DeepHermes, Mistral-Small-3.2, DeepSeek-R1). Root cause: Broken `tokenizer.json` PreTokenizer regex patterns. Solution: (1) Post-load tokenizer regex patching for encoding fix, (2) `_decode_tokens()` helper using `tokenizer.detokenizer` for proper BPE space marker conversion in both streaming and batch modes. Impact: Correct encoding/decoding, 15% context window waste eliminated. All generation modes fixed (streaming, batch, server). Related: HuggingFace discussion https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503/discussions/84

- **Memory Cleanup Hook:** Fixed 60GB swap during wet tests. Hook now triggers for both `live_e2e` and `wet` markers.

- **Missing Model Skip:** Regression test now skips cleanly when hardcoded model not in cache.

- **pytest CLI option location:** Moved `--report-output` hook from subdirectory to root conftest.py (pytest requirement).

- **Cache resolution in pull:** Dynamic `get_current_model_cache()` instead of module-level import (fixes test isolation).

- **Test Regression:** Fixed 4 unit tests after BPE detokenizer changes. Created `MockDetokenizer` class matching BPEStreamingDetokenizer API.

### Testing

- **Benchmark Quality Metrics (OS-agnostic):** Switched from swap-based to RAM-based quality flags. DeepHermes-3-24B steady-state baseline analysis revealed OS-specific memory behavior: Tahoe maintains ~24 GB free RAM with aggressive swap (1-55 GB), Sequoia maintains ~40 GB free with minimal swap (0 MB). Despite 16 GB RAM difference, both OSs achieve similar performance. Identified `ram_free < 5 GB` as universal degradation threshold (extreme memory pressure). Result: 97.8% clean on Tahoe, 96.9% clean on Sequoia. Report generator now recalculates quality flags from raw metrics, ignoring stored flags. Created memory profiling tooling (memmon.py, memplot.py). MLX 0.30.x cleanup verified functional.

### Community

- **Repair workflow:** 7+ mlx-community VLM models fixable with `mlxk clone <model> ./ws && mlxk convert ./ws ./ws-fixed --repair-index`. Discord announcement pending mlx-vlm 0.3.10 release.

---

## [2.0.4-beta.4] - 2025-12-25

### Fixed

- **Pixtral text-only regression (upstream mlx-vlm):**
  - Updated mlx-vlm dependency to commit `c536165df2b3b4aece3a795b2e414349f935e750` (Blaizzy/mlx-vlm)
  - **Issue:** mlx-vlm commit d7d73de (Batch Generation #538) broke text-only requests on Vision models without `pad_token`
  - **Fix:** Upstream PR merged - sets `pad_token = eos_token` when missing during text-only padding
  - **Impact:** Pixtral and other Vision models now work correctly for both text-only and image requests
  - **Installation:** `pip install mlx-knife[vision]` now auto-installs patched mlx-vlm from git
  - **Future:** Will switch to PyPI mlx-vlm v0.3.10 when released
  - **Details:** See `docs/ISSUES/pixtral-pad-token-regression.md`
  - **Upstream contribution:** Issue #643, PR merged into Blaizzy/mlx-vlm main
  - Files: `pyproject.toml:69`, `README.md`

### Documentation

- **mlx-vlm installation guidance:** Updated to use commit c536165df2b3b4aece3a795b2e414349f935e750 with Pixtral fix
  - Files: `README.md`, `pyproject.toml`

## [2.0.4-beta.3] - 2025-12-23

### Added

- **Benchmark Infrastructure v1.0 (ADR-013 Phase 0):**
  - Template-based report generator: `benchmarks/generate_benchmark_report.py`
  - Per-model statistics, per-test statistics, system health summary
  - Schema validation: `benchmarks/validate_reports.py`
  - Documentation: `benchmarks/README.md`, `benchmarks/TESTING.md`
  - Quality tracking: Schema v0.2.0 with system_health (swap, RAM, zombies, quality_flags)
  - Page-size fix: Corrected Apple Silicon 16KB page size (RAM values were 4x too low)
  - Files: `tests_2.0/live/conftest.py`, `test_utils.py`, `test_vm_stat_parsing.py`

- **Memory Timeline Visualization:**
  - Interactive HTML visualizer: `benchmarks/tools/memplot.py` (500+ lines)
  - Memory monitor enhanced: `benchmarks/tools/memmon.py` (memory pressure capture)
  - Visual legend: Activity Monitor colors, memory pressure, test regions, model markers
  - Documentation: Complete interpretation guide in `benchmarks/README.md`
  - Schema learnings: Server test attribution problem + log-parsing solution documented
  - File: `benchmarks/schemas/LEARNINGS-FOR-v1.0.md`

### Fixed

- **Server model switch log timing:** "Switched to model" now emitted only after successful load (past tense reflects completed action)
  - File: `mlxk2/core/server_base.py:230`

- **Unified model filter (Server + CLI):** Both `/v1/models` and `mlxk list` now use `build_model_object()` as single source of truth
  - Filter: `healthy AND runtime_compatible` (no more code duplication)
  - Framework gate: Non-MLX models (PyTorch, GGUF) now correctly marked `runtime_compatible=false`
  - WebUI clients get consistent, runnable model lists
  - Files: `mlxk2/core/server_base.py`, `mlxk2/output/human.py`, `mlxk2/operations/common.py`

- **transformers 5.0 compatibility for vision models:** Removed `fix_mistral_regex` parameter from mlx-vlm load call
  - transformers 5.0.0rc1 changed tokenizer initialization - `fix_mistral_regex` no longer accepted as kwarg
  - Error was: `TypeError: _patch_mistral_regex() got multiple values for keyword argument 'fix_mistral_regex'`
  - Removed deprecated parameter from vision model loading - all vision models now work with transformers 5.0
  - File: `mlxk2/core/vision_runner.py:101`

- **huggingface-hub 1.x compatibility:** Updated preflight test mocks for hub 1.x exception API changes
  - Hub 1.x changed exception signatures: `GatedRepoError/RepositoryNotFoundError` now require `response` parameter
  - Added `_create_mock_response()` helper to create proper httpx.Response objects for test mocks
  - **Test-only changes** - preflight production code works unchanged with hub 0.x and 1.x
  - **Result:** mlx-knife now fully compatible with mlx 0.30.x, mlx-lm 0.30.0, transformers 5.0, hub 1.x
  - All 494 unit tests pass, vision models functional with newest dependencies
  - Files: `tests_2.0/test_issue_30_preflight.py`, `mlxk2/core/vision_runner.py`

- **EXIF GPS 0° coordinate handling:** Fixed truthiness checks in `VisionRunner._extract_exif` that incorrectly dropped valid GPS coordinates
  - Equator (0° latitude) and Prime Meridian (0° longitude) now correctly preserved
  - Changed latitude/longitude negation checks from `if lat` to `if lat is not None`
  - Changed EXIF retention check from `not any([...])` to `all(x is None for x in [...])`
  - Ensures 0.0 is treated as valid coordinate, not as missing data
  - File: `mlxk2/core/vision_runner.py:259-262, 283`

- **Framework/Type detection for non-mlx-community models (Issue #48):**
  - `detect_framework()`: Now reads front-matter internally and checks config.json `quantization` key (MLX-specific)
  - `detect_model_type()`: Added `probe` parameter and checks for `chat_template.json` file (reliable chat indicator)
  - Removed redundant PR #42 code from server_base.py (cleaner architecture)
  - Fixes: Models like locally converted quantized models now correctly show "MLX" + "chat" instead of "PyTorch" + "base"
  - Files: `mlxk2/operations/common.py:118-157, 180-208`, `mlxk2/core/server_base.py:114-120`

- **Video model detection and exclusion:**
  - Video models (require PyTorch/Torchvision) now excluded from vision capability detection
  - mlx-vlm only supports image vision models, not video models
  - Video indicators: `video_preprocessor_config.json`, `temporal_patch_size`, `AutoVideoProcessor`
  - Video models fall back to mlx-lm for text-only (consistent with vision architecture)
  - Example: `mlx-community/MiMo-VL-7B-RL-bf16` now classified as "chat" (not "chat+vision")
  - Files: `mlxk2/operations/common.py:211-266`, `mlxk2/core/capabilities.py:169-238`

### Documentation

- **mlx-vlm beta.3 install guidance:** Recommend upstream commit `c4ea290e47e2155b67d94c708c662f8ab64e1b37` until mlx-vlm 0.3.10 is released
  - Files: `README.md`, `docs/SERVER-HANDBOOK.md`

## [2.0.4-beta.2] - 2025-12-16

**PyPI-only release** - Fixes Git dependency issue for PyPI compatibility. Not tagged on GitHub.

### Fixed

- **PyPI compatibility:** Changed `mlx-vlm` dependency from Git URL to PyPI version `mlx-vlm>=0.3.9`
  - PyPI does not allow Git dependencies
  - mlx-vlm 0.3.9 is available on PyPI
  - File: `pyproject.toml:69`

### Documentation

- **Installation instructions:** Added Vision-specific installation to README.md
  - Clear separation: Text models (Python 3.9+) vs Vision models (Python 3.10+)
  - Installation command: `pip install mlx-knife[vision]`
  - Updated all version references from 2.0.4-beta.1 → 2.0.4-beta.2

## [2.0.4-beta.1] - 2025-12-16

**Focus:** Unix Pipe Integration + Vision Support + Memory-Aware Loading + Python 3.14

### Added

- **Unix Pipe Integration (ADR-014 Phase 1):**
  - `mlx-run` wrapper command for streamlined model execution
  - Stdin support: `mlxk run <model> -` reads from stdin, concatenates trailing CLI text with `\n\n` separator
  - Auto-batch mode: Non-TTY stdout disables streaming for clean pipe output
  - SIGPIPE handler and BrokenPipeError handling for Unix convention compliance
  - Gate: `MLXK2_ENABLE_PIPES=1` (beta, stable API)
  - Example: `examples/mlx-tee.py` for parallel model execution

- **Vision Support (Issue #45, ADR-012 Phase 1-3):**
  - CLI (Phase 1+2): Vision model detection, `--image` flag with multiple images, VisionRunner (mlx-vlm 0.3.9+), health checks
  - Server (Phase 3): VisionHTTPAdapter for OpenAI-compatible Base64 image requests
  - Limits: 5 images max, 20 MB per image, 50 MB total (prevents Metal OOM crashes)
  - Requires: Python 3.10+ (mlx-vlm dependency)

- **Memory-Aware Model Loading (ADR-016 Phase 1+2):**
  - JSON API 0.1.6: `system.memory_total_bytes` in `mlxk --version --json`
  - Pre-load memory checks: Vision models >70% RAM → abort with error (CLI) or HTTP 507 (server)
  - Text models >70% RAM → warning only (backwards compatible, swap-tolerant)

- **Python 3.14 Support:**
  - mlx-lm 0.28.4+ with official Python 3.14 support
  - Tested: Python 3.9.6 - 3.14 (all versions verified)

- **Server `/v1/models` Enhancements:**
  - Preloaded model listed first (when `--model` specified), then alphabetically
  - Resolved name stored for sorting (e.g., "qwen" → "mlx-community/Qwen2.5-0.5B-Instruct-4bit")
  - Filters to healthy MLX models only (runtime compatibility check deferred to P2 refactoring due to code duplication)

### Fixed

- **`mlxk serve --log-json` fully fixed (Issue #44):** Clean JSON logs without duplicates or mixed formats
  - **Fixed mixed plain-text/JSON logs:** Added `__main__` entrypoint; supervised mode now calls `python -m mlxk2.core.server_base` with JSON config
  - **Fixed duplicate uvicorn logs:** Added `propagate: False` to `uvicorn.error` logger config
  - **Result:** 100% parseable JSON output for log aggregation and monitoring systems
  - **Config:** Passed via environment variables (`MLXK2_HOST`, `MLXK2_PORT`, `MLXK2_LOG_LEVEL`, `MLXK2_LOG_JSON`)

### Documentation

- ADR-012: Vision Support Phase 1-3 (CLI complete, Server Phase 3 in progress)
- ADR-014: Unix Pipe Integration Phase 1 complete
- ADR-016: Memory-Aware Model Loading complete
- ARCHITECTURE.md: Core principles documented (no silent fallbacks, fail fast/clearly, memory gates)

## [2.0.3] - 2025-11-17

**Stable Release**: Benchmark infrastructure + Unix stderr fix + reasoning control + dependency hardening.

### Features

- **Benchmark Reporting Infrastructure**:
  - `--report-output` flag in E2E tests: Writes JSONL benchmark reports with model metadata
  - `report_benchmark()` fixture: Easy model metadata reporting (family, variant, size_gb, stop_tokens, skip_reason)
  - Model family detection: `_parse_model_family()` helper extracts family/variant from model IDs
  - Schema validation: `benchmarks/schemas/report-v0.1.schema.json` + validation script
  - **Validated**: 17 models with full metadata (Phi-3, Qwen, Llama, Mistral, DeepSeek, etc.)
  - **Files**: `tests_2.0/live/conftest.py`, `test_cli_e2e.py`, `test_server_e2e.py`
  - **Documentation**: `benchmarks/README.md`, `benchmarks/TESTING.md`, `benchmarks/schemas/MIGRATIONS.md`

- **Reasoning Model Control** (`--no-reasoning` flag):
  - CLI toggle to hide reasoning output (Issue #40 Option 1 - partial implementation)
  - Works in both streaming and batch modes (single-shot + interactive)
  - Default: Show reasoning (backward compatible)
  - **Limitations**: Only works with models that auto-generate reasoning tags (GPT-OSS, QwQ-32B via chat templates)
  - **Not supported**: DeepSeek-R1, Qwen3, and most other reasoning models (require system prompts from Issue #33)
  - **Issue #40 remains open**: Requires structured API (Option 2) and depends on #33 (System Prompts) for broad model support
  - **Technical fix**: - Flag now correctly propagates through `interactive_chat()` in both streaming and batch modes
  - **Files**: `mlxk2/cli.py`, `mlxk2/operations/run.py`, `mlxk2/core/runner/__init__.py`, `mlxk2/core/runner/reasoning_format.py`

### BREAKING CHANGES

**Error Output to stderr (Human Mode Only)**

Errors are now printed to stderr instead of stdout in human mode. This follows Unix conventions and enables clean pipe workflows. JSON mode remains unchanged (all output to stdout) for scripting/automation use cases.

**What changed:**
- **Human mode**: Errors → stderr (was stdout)
- **JSON mode**: Unchanged - all output to stdout (errors + success, for scripting)
- Exit codes: Unchanged (0=success, 1=error)
- **Affected commands**: list, health, show, pull, rm, clone, push, run (all commands)

**Migration:**

```bash
# Human mode: Capture both stdout and stderr if needed
OUTPUT=$(mlxk list 2>&1)

# Recommended: Separate success and error streams (human mode)
if mlxk pull model > output.txt 2> error.log; then
    echo "Success: $(cat output.txt)"
else
    echo "Error: $(cat error.log)"
fi

# JSON mode: No change needed (all output still on stdout)
OUTPUT=$(mlxk show model --json)
echo "$OUTPUT" | jq .status  # Works as before
```

**Not affected:**
- Interactive terminal users (stderr visible)
- JSON mode users (all output on stdout)
- Exit code checks
- Pipe workflows (actually **fixed** in human mode)

### Bug Fixes

- **stdout/stderr separation** (Issue #43): Errors now correctly go to stderr (human mode only)
  - **Central implementation**: `print_result()` helper in `cli.py` for consistent error handling
  - **Run command**: 5 error print statements changed to `file=sys.stderr` in `operations/run.py:57, 88, 102, 132, 229`
  - **All other commands**: Unified via `print_result()` (list, health, show, pull, rm, clone, push)
  - **Human mode errors**: Generic format `command: Error: message` (stderr, consistent across all commands)
  - **JSON mode errors**: Structured JSON on stdout (unchanged, for scripting/jq workflows)
  - **Rationale**: JSON is for automation/scripting (not piping), human mode is for interactive + pipes
  - **Test updates**: 2 interactive mode tests updated to check stderr
    - `tests_2.0/test_interactive_mode.py`: 2 assertions (template fallback, generation error recovery)

- **huggingface-hub 1.x incompatibility** (Critical dependency fix):
  - **Problem**: `huggingface-hub>=0.34.0` allowed upgrades to 1.x, breaking `transformers` compatibility
  - **Impact**: All models showed `healthy*` (integrity OK, but runtime failed)
  - **Fix**: Pin `huggingface-hub>=0.34.0,<1.0` in dependencies
  - **File**: `pyproject.toml:30`

### Testing Improvements

- **Streaming parity tests refactored to use portfolio discovery**:
  - **Problem**: Tests had hardcoded model IDs (e.g., `Llama-3.2-3B-Instruct-4bit`) that may not exist in user's cache
  - **Impact**: Tests failed with cryptic "mock path" errors when models not downloaded
  - **Fix**: Tests now use portfolio discovery (`mlxk list --json`) to select 2-3 available small models (<6GB)
  - **Selection strategy**: Smallest models first, exclude reasoning models (known batch/stream inconsistency, fixed in ADR-010)
  - **Result**: Tests automatically adapt to available models, no more hardcoded dependencies
  - **File**: `tests_2.0/live/test_streaming_parity.py`
  - **Custom hook**: `pytest_generate_tests()` parametrizes tests over discovered models at collection time

### Known Issues

- **Large model downloads (>30 GB) can fail overnight**: Connection resets during multi-hour downloads
- **External SSD I/O deadlocks with parallel downloads**: `huggingface-cli` default `max_workers=8` causes stalls at 99%. Workaround: `--max-workers 1`

---

## [2.0.2] - 2025-11-15

**Stable Release**: Test infrastructure hardening, stop token validation with 17 models, and web API improvements.

This release completes the 2.0.2 recovery plan (Issue #32) with extensive empirical validation, architecture decisions, and community contributions. Highlights: 73/81 E2E tests passing, stop token bugs fixed, web API framework detection for all MLX organizations.

### Bug Fixes

- **Test collection regression** (E2E test suite, ADR-011):
  - **Problem**: `pytest tests_2.0/live/` failed with "fixture 'model_key' not found" without `-m live_e2e` marker
  - **Root Cause**: `conftest.py:64-70` returned early without parametrizing when marker missing
  - **Fix**: Added fallback parametrization with `["_skipped"]` - tests now collect and skip gracefully
  - **Impact**: Collection works without markers (22 tests), with marker discovers 17 models (81 tests)
  - **File**: `tests_2.0/live/conftest.py:68-72`

- **Stop token ordering bugs** (batch AND streaming modes, ADR-009):
  - **Problem**: Both `generate_batch()` and `generate_streaming()` filtered stop tokens by **list order** instead of **text position**
  - **Impact**: Models generating multiple EOS tokens (e.g., Phi-3-mini: `<|end|><|endoftext|>`) could leak stop tokens into output
  - **Evidence**: Phi-3-mini generates two token IDs: `32007='<|end|>'` then `32000='<|endoftext|>'`
  - **Old behavior**: Checked stop tokens list `['<|endoftext|>', '</s>', '<|end|>']` → found `<|endoftext|>` first (position 146) → left `<|end|>` (position 139) in output
  - **New behavior**: Finds **earliest stop token in text** → cuts at position 139 → clean output
  - **Affected**: All models that generate multiple EOS tokens
  - **Files**: `mlxk2/core/runner/__init__.py` (streaming: 441-466, batch: 619-631)
  - **Validation**: 73/81 tests passing with diverse portfolio (Phi-3, DeepSeek-R1, GPT-oss, Llama, Qwen, Mistral, Mixtral)

- **E2E test temperature flakiness** (Test reliability fix):
  - **Problem**: CLI E2E tests used default `temperature=0.7` → non-deterministic outputs → flaky test results
  - **Fix**: Added `temperature=0.0` to all CLI E2E tests for reproducible results
  - **Rationale**: E2E tests validate code logic (stop token filtering), not model quality
  - **Files**: `tests_2.0/live/test_cli_e2e.py`, `tests_2.0/live/test_utils.py` (TEST_TEMPERATURE constant)

- **Web API framework detection** (PR #42 by @limey, fixes Issue #41): `/v1/models` endpoint now correctly lists MLX models from all organizations, not just `mlx-community/*`

- **E2E test marker fix**: `pytest -m show_model_portfolio` now works for diagnostic model discovery

### Architecture

- **mlx-lm API evaluation** (ADR-009):
  - **Question**: Migrate to `BatchGenerator(stop_tokens=...)` or keep manual implementation?
  - **Research**: Source code analysis of mlx-lm 0.28.3 (`generate.py`, `BatchGenerator`)
  - **Critical Finding**: BatchGenerator uses **token-ID based** stop detection (`set[int]`)
  - **Fundamental Blockers**:
    1. Cannot handle multi-token sequences like `"\nHuman:"` (required for Issue #14 chat turns)
    2. No streaming support (we need SSE for `/v1/chat/completions`)
    3. No "earliest position" logic (Phi-3-mini dual EOS breaks)
    4. No reasoning parser integration (MXFP4 support breaks)
  - **Historical Proof**: Issue #14 (1.x) validated text-based approach (114 tests passing, 1.0.4)
  - **Decision**: **Keep manual text-based implementation** (migration impossible)
  - **Impact**: No code changes needed, validation simplified

- **Stop token workaround evaluation** (ADR-009):
  - **Workaround 1** (Line 49): `<|end|>` special handling for Phi-3-mini
    - **Validated**: 2 Phi-3 variants in portfolio (discovered_11, discovered_12)
    - **Rationale**: Fixes `eos_token_id=null` bug, empirically stable
    - **Decision**: **Keep** (0 failures, production stable)
  - **Workaround 2** (Line 98): `reasoning_end` removal for DeepSeek-R1
    - **Validated**: DeepSeek-R1-Distill-8B in portfolio (discovered_01)
    - **Rationale**: Reasoning models need full output until final marker
    - **Decision**: **Keep** (supports ADR-010 reasoning roadmap)
  - **Workaround 3** (Line 100): `<|return|>` addition for GPT-oss
    - **Validated**: gpt-oss-20b-MXFP4 in portfolio (discovered_16)
    - **Rationale**: GPT-oss reasoning format requires special marker
    - **Decision**: **Keep** (future-proof for larger reasoning models)
  - **Evidence**: All 3 workarounds validated with 73/81 tests passing, 0 failures
  - **File**: `mlxk2/core/runner/stop_tokens.py`

### Testing

- **Portfolio Discovery validation** (ADR-009, Issue #32):
  - **Scope**: 17 models discovered, 15 testable (60% RAM budget), 2 skipped (RAM constraints)
  - **Results**: 73/81 tests passing, 0 failures
  - **Portfolio Families**: Phi-3, DeepSeek-R1, GPT-oss, Llama, Qwen, Mistral, Mixtral
  - **Validation**: All 3 stop token workarounds actively used and validated
  - **Performance**: 7:55 minutes (64GB M2 Max, sequential execution, no system freeze)
  - **Command**: `HF_HOME=/path/to/cache pytest -m live_e2e tests_2.0/live/ -v`

- **E2E test infrastructure hardening** (ADR-011):
  - **TOKENIZERS_PARALLELISM=false**: Prevents fork warnings and potential deadlocks
  - **Active cleanup polling**: Waits for actual process termination (not blind timeout)
  - **Explicit garbage collection**: `gc.collect()` + 2s Metal memory buffer prevents RAM overlap
  - **Conservative timeout**: 45s max wait for very large models (>40GB), polls every 500ms
  - **Sequential execution warning**: TESTING-DETAILS.md documents parallel execution risks (Lines 91-128)
  - **Files**:
    - `tests_2.0/live/conftest.py` - TOKENIZERS_PARALLELISM + fallback parametrization
    - `tests_2.0/live/server_context.py` - Active polling + gc.collect()

### Documentation

- **ADR-009 Outstanding Work completed**:
  - **Portfolio Discovery**: Implemented (2.0.1), validated (2.0.2-beta.1)
  - **Workaround Evaluation**: Completed with empirical evidence (3/3 kept)
  - **Empirical Validation**: Expanded from 3 → 15 models tested
  - **File**: `docs/ADR/ADR-009-Stop-Token-Detection-Fix.md` Lines 180-206

- **TESTING-DETAILS.md harmonization**:
  - **Portfolio Discovery section** (Line 24): Updated with current validation (17 models, 73/81 tests)
  - **E2E Test Architecture section** (Lines 465-493): Updated with model_key fix, collection warning, current results
  - **De-versioned**: Changed from "Validation (2.0.2)" to "Current validation" (timeless guide, not version-specific)
  - **ADR references maintained**: Architecture context preserved

- **CLAUDE.md accuracy audit**:
  - **ADR-009 status**: Updated to reflect 2.0.1 + 2.0.2 completion timeline
  - **ADR-011 status**: Updated to 73/81 tests passing, 17 models discovered
  - **Roadmap**: Updated with recovery plan progress
  - **All claims evidence-based**: No false "completed" claims

**Production Code Changes**:
- `mlxk2/__init__.py` - Version 2.0.2
- `mlxk2/core/runner/__init__.py` - Stop token ordering fix (streaming + batch modes)
- `mlxk2/core/runner/stop_tokens.py` - Workarounds validated and documented
- `mlxk2/core/server_base.py` - Web API framework detection (PR #42)

**Test Infrastructure & Documentation**:
- `tests_2.0/live/*` - New E2E test infrastructure (conftest, server_context, test suites)
- `docs/ADR/ADR-009-Stop-Token-Detection-Fix.md` - Outstanding Work completed

---

## 2.0.1 — 2025-11-8

**Bug Fix & Enhancement Release**: CLI exit code propagation fixes + Portfolio Discovery for stop token validation.

### Fixed

- **CLI `run` command exit codes** (GitHub Issue #38): `mlxk run` now correctly returns exit code 1 when model execution fails, enabling proper error detection in shell scripts and automation workflows
  - **Both modes fixed**: Text mode and JSON mode now properly propagate errors
  - **JSON mode**: Returns `{"status": "error", "error": {...}}` with exit code 1
  - **Text mode**: Prints `"Error: ..."` message and returns exit code 1
  - **Affects**: Shell scripts using `mlxk run && next_step`, batch processing, model validation workflows
  - **Root cause**: `run_model()` returned `None` in text mode instead of error strings; CLI had no way to detect text-mode failures
  - **Fix**:
    - Modified `mlxk2/operations/run.py` to return `"Error: ..."` strings in both modes (lines 50-86, 125-129)
    - CLI error detection in `mlxk2/cli.py:273-288` now catches errors for both modes
  - **Examples of fixed scenarios**:
    - Nonexistent model: `mlxk run bad-model "hi"` → exit 1 (was: exit 0)
    - Incompatible model: Runtime version mismatch → exit 1 (was: exit 0)
    - Runtime exceptions: OOM, loading failures → exit 1 (was: exit 0)

- **Stop token validation Portfolio Discovery** (GitHub Issue #32, ADR-009): Live stop token tests now support dynamic model discovery and HF_HOME-optional testing
  - **Portfolio Discovery**: Auto-discovers all MLX chat models via `mlxk list --json` (filter: MLX + healthy + runtime_compatible + chat)
  - **Refactored in 2.0.2**: Now uses production command instead of duplicating cache logic (~70 LOC eliminated)
  - **RAM-Aware Testing**: Progressive RAM budgets (40-70%) prevent OOM during multi-model validation
  - **Empirical Reporting**: Generates `stop_token_config_report.json` with cross-model stop token findings
  - **Fallback Support**: Tests work without `HF_HOME` using 3 predefined models (MXFP4, Qwen, Llama)
  - **Marker-Required**: Tests excluded from default suite, use `pytest -m live_stop_tokens` to run
  - **Implementation**: `tests_2.0/test_stop_tokens_live.py` (~50 LOC for discovery + RAM gating)

### Testing

- 306 passed, 20 skipped (including 4 live stop token tests, marker-required)
- **New test files**:
  - `tests_2.0/test_cli_run_exit_codes.py` validates both text and JSON mode exit codes (+9 tests)
  - `tests_2.0/test_stop_tokens_live.py` implements Portfolio Discovery with HF_HOME-optional fallback (+4 tests)
- **Updated tests**: `tests_2.0/test_run_complete.py` reflects new error contract
- Zero regressions in full test suite

---

## 2.0.0 — 2025-11-06

**Stable Release**: MLX Knife 2.0 replaces 1.x as the primary version. Full feature parity with 1.1.1 achieved plus major enhancements.

### License Change

- **MIT → Apache License 2.0**: Better patent protection, industry-standard licensing
- See [MIGRATION.md](MIGRATION.md) for details on license change and user impact

### Highlights

- **Full 1.x Feature Parity**: All commands from 1.1.1 available (`list`, `show`, `pull`, `rm`, `run`, `server`, `health`)
- **JSON API**: Machine-readable output for automation (`--json` flag on all commands)
- **Enhanced Error Handling**: Structured errors with request IDs, logging levels, JSON logs
- **Runtime Compatibility Checks**: Pre-flight validation prevents loading incompatible models
- **Improved Stop Token Detection**: Multi-EOS support (MXFP4, Qwen, Llama)
- **Better Human Output**: Improved formatting, relative timestamps, runtime status

### Package Changes

- **Package name**: `mlx-knife` (unchanged from 1.x)
- **Primary command**: `mlxk` (replaces `mlxk2` from beta)
- **Aliases**: `mlxk-json`, `mlxk2` (backwards compatibility)

### Breaking Changes

- **Lock file handling**: `mlxk rm` requires `--force` flag when models have active locks (safety improvement)
- See [MIGRATION.md](MIGRATION.md) for complete migration guide from 1.x

### Installation

```bash
# PyPI (recommended)
pip install mlx-knife

# GitHub release
pip install https://github.com/mzau/mlx-knife/releases/download/v2.0.0/mlx_knife-2.0.0-py3-none-any.whl

# Upgrade from 1.x
pip install --upgrade mlx-knife
```

### Testing

- 297 passed, 20 skipped (317 total tests)
- Python 3.9-3.13 compatibility verified
- Apple Silicon (M1/M2) tested

---

## 2.0.0-beta.6 — 2025-10-22

### Fixed
- **Stop token detection for multi-EOS models** (Issue #32, ADR-009): MXFP4 and Qwen models no longer generate visible stop tokens (`<|end|>`) or chat template markers in output
- **Private/org MLX model detection** (Issue #37): `mlxk run` now correctly detects MLX models outside `mlx-community/*` namespace
- **Commit-pinned compatibility checks**: Models with `@commit_hash` syntax now correctly validated before inference
- **Packaging dependencies** (P0): `pip install -e .` now installs all required dependencies (`mlx-lm`, `mlx`, `fastapi`, etc.) via `pyproject.toml`

### Documentation
- Simplified installation instructions in README.md and TESTING.md (consistent `pip install -e ".[dev,test]"` recommendation)

### Testing
- 297 passed, 20 skipped (317 total)
- Added 6 new tests: 4 stop token validation tests (opt-in), 2 compatibility check tests

## 2.0.0-beta.5 — 2025-10-20

**Enhanced Error Handling & Logging (ADR-004)**: Unified error envelope, structured logging with JSON support, and request correlation.

**Legacy Model Format Detection**: Models with outdated weight file formats are detected and marked as runtime-incompatible (Issue #37).

### Added

- **Error envelope and structured logging** (ADR-004 Phase 1):
  - Unified error envelope for CLI/Server: `{"status": "error", "error": {"type", "message", "detail", "retryable"}, "request_id"}`
  - Request correlation via `request_id` (UUID4) in all server responses and logs
  - HTTP status mapping: 400 (validation), 403 (access denied), 404 (not found), 500 (internal), 503 (shutdown)
  - Structured logging with INFO/WARN/ERROR/DEBUG levels (replaces ad-hoc print statements)
  - Optional JSON logs via `MLXK2_LOG_JSON=1` for machine-readable output
  - **Log-level control**: `--log-level` (debug/info/warning/error) controls MLXKLogger, root logger, and Uvicorn access logs
  - **`--log-json` CLI flag**: User-friendly alternative to `MLXK2_LOG_JSON=1` environment variable
  - **Uvicorn JSON formatting**: Access logs (`GET /v1/models`, etc.) also formatted as JSON when `--log-json` is used
  - **Root logger JSON formatting**: External libraries (mlx-lm, transformers) also log as JSON in JSON mode
  - Automatic redaction of sensitive data (HF tokens, user paths)
  - Error rate limiting (max 1 error per 5s for duplicate errors)
  - New modules: `mlxk2/errors.py`, `mlxk2/logging.py`, `mlxk2/context.py`
  - FastAPI middleware: Request ID injection, custom exception handler
  - **User documentation**: README.md "Logging & Debugging" section (log levels, JSON format, redaction examples)
  - Test coverage: 22 new tests in `test_adr004_error_logging.py`

- **Legacy format detection in runtime compatibility check** (Issue #37):
  - Gate 2 in `check_runtime_compatibility()`: Validates weight file naming conventions
  - Detects legacy patterns: `weights.*.safetensors` (e.g., `weights.00.safetensors`), `pytorch_model-*.safetensors`
  - Accepts modern patterns: `model.safetensors`, `model-XXXXX-of-YYYYY.safetensors`
  - Clear error message: `"Legacy format not supported by mlx-lm"`
- **Pre-flight check in `run` command**:
  - Validates runtime compatibility before attempting model load
  - Prevents cryptic mlx-lm errors: `"ERROR:root:No safetensors found in..."`
  - Returns user-friendly error: `"Model 'X' is not compatible: Legacy format not supported by mlx-lm"`
  - Best-effort check: gracefully skips if model not in cache (preserves test compatibility)

### Changed
- **Runtime compatibility validation extended**:
  - Gate 1: Framework check (MLX vs GGUF/PyTorch) - from Beta.4
  - Gate 2: **NEW** - Weight file format check (modern vs legacy patterns)
  - Gate 3: Model type support check (mlx-lm compatibility) - from Beta.4
- **CLI description**: "HuggingFace model management for MLX" (removed "JSON-first" and version number)
- **README reorganization**: Better section flow, merged duplicate sections, removed beta-specific content (550 lines)

### Fixed
- **Legacy format detection** (Issue #37, bug):
  - Models with legacy weight file formats (`weights.*.safetensors`, `pytorch_model-*.safetensors`) now correctly detected as runtime-incompatible
  - Health output: `healthy` (file integrity OK) but `runtime_compatible: false`
  - `reason` field describes incompatibility: `"Legacy format not supported by mlx-lm"`
  - Human output: `healthy*` in compact mode, `healthy | no | Legacy format...` in verbose mode
  - Pre-flight check in `run` command prevents cryptic mlx-lm errors
- **CLI error handling** (regression since 19a6667): Running `mlxk2` without arguments now shows help text (like git/docker) instead of JSON error, `--json` flag properly respected for automation
- **Code quality**: Removed 7 unused imports, ruff checks pass

### Implementation
- `mlxk2/operations/health.py`:
  - `check_runtime_compatibility()` Gate 2 implementation (lines 272-304)
  - Regex patterns for legacy format detection
  - Mixed legacy/modern: prefers modern if both present
- `mlxk2/operations/run.py`:
  - Pre-flight runtime compatibility check (lines 45-89)
  - Clear error messages before mlx-lm loading

### Testing
- **Current Status**: 293 passed, 14 skipped, 1 warning (urllib3/LibreSSL)
- **New Tests** (25 total):
  - `tests_2.0/test_adr004_error_logging.py` (22 tests):
    - Error envelope structure and serialization
    - Error type to HTTP status mapping (8 error types validated)
    - Request ID generation and propagation (UUID4 validation, context nesting)
    - Log redaction (HF tokens, home directory paths)
    - Structured logging (plain text vs JSON modes, log levels, rate limiting)
  - `tests_2.0/test_legacy_formats.py` (3 tests):
    - `test_weights_numeric_safetensors_is_runtime_incompatible`: Validates `weights.00.safetensors` detection
    - `test_pytorch_model_numeric_safetensors_is_runtime_incompatible`: Validates `pytorch_model-*.safetensors` detection
    - `test_modern_model_safetensors_passes_legacy_gate`: Ensures modern formats are not rejected
- **Regression**: All existing tests pass (zero breaking changes)

### Known Issues
- **Missing tests for Issue #36** (Beta.4 gap):
  - No dedicated tests for Gate 1 (framework check)
  - No dedicated tests for Gate 3 (model_type support)
  - Runtime compatibility tested indirectly via Issue #37 tests and schema validation
  - TODO: Add explicit tests for Beta.4 runtime compatibility feature

### User Experience Example
```bash
# Before (Beta.4): Cryptic mlx-lm error
$ mlxk2 run TinyLlama-1.1B-Chat-v1.0-4bit "Hello"
ERROR:root:No safetensors found in /Volumes/.../snapshots/01a7088...

# After (Beta.5): Clear error message
$ mlxk2 run TinyLlama-1.1B-Chat-v1.0-4bit "Hello"
Error: Model 'mlx-community/TinyLlama-1.1B-Chat-v1.0-4bit' is not compatible: Legacy format not supported by mlx-lm

# Health status shows details
$ mlxk2 show TinyLlama-1.1B-Chat-v1.0-4bit
Health: healthy (files OK, runtime incompatible)
Reason: Legacy format not supported by mlx-lm
```

### Notes
- Legacy models are file-complete (healthy integrity) but use outdated naming conventions incompatible with modern mlx-lm
- Pre-flight check improves UX by catching incompatibility before expensive model loading

---

## 2.0.0-beta.4 — 2025-10-18

**Health Check Enhancement**: Separate integrity and runtime compatibility validation (Issue #36).

### Changed
- **JSON API 0.1.5 specification**:
  - Added `runtime_compatible: boolean` field to `modelObject` (always present)
  - Added `reason: string | null` field to `modelObject` (describes first problem found)
  - `list`/`show` JSON output performs both integrity and runtime compatibility checks
  - Gate logic: Runtime check requires integrity check first; `reason` shows first problem (integrity > runtime priority)
- **Health check concepts documented**:
  - Integrity Check (`health` field): File-level validation (required files, no LFS pointers, valid JSON)
  - Runtime Compatibility Check (`runtime_compatible` field): MLX framework + architecture validation with mlx-lm
  - Framework detection: GGUF/PyTorch models marked as runtime-incompatible
  - Architecture detection: Unsupported model types (e.g., `qwen3_next` with mlx-lm < 0.28.0) detected
  - Respects `MODEL_REMAPPING` for aliased architectures (e.g., `mistral` → `llama`)

### Implementation Status
- ✅ **Phase 1 Complete**: JSON API Specification 0.1.5
  - `docs/json-api-schema.json` updated with new fields
  - `docs/json-api-specification.md` extended with health check concepts and examples
- ✅ **Phase 2 Complete**: JSON Implementation
  - `mlxk2/spec.py` bumped to 0.1.5
  - `mlxk2/operations/health.py`: `check_runtime_compatibility()` with gate logic
  - `mlxk2/operations/common.py`: `build_model_object()` always computes `runtime_compatible` + `reason`
  - mlx-lm API compatibility: Supports both 0.27.x (`mlx_lm.utils._get_classes`) and 0.28.x APIs
  - Log suppression: mlx-lm ERROR logs redirected to `reason` field only
- ✅ **Phase 3 Complete**: Human Output Specification
  - Compact mode: `healthy` / `healthy*` / `unhealthy` (single column)
  - Verbose mode: "Integrity" | "Runtime" | "Reason" (split columns)
  - ASCII-only output (no UTF-8 symbols for parsing compatibility)
  - README.md fully documented with examples and design philosophy
  - JSON examples verified for consistency with schema and code
- ✅ **Phase 4 Complete**: Human Output Implementation in `mlxk2/output/human.py`

### Dependencies
- **mlx-lm requirement updated**: `>=0.27.0` → `>=0.28.3`
  - Now uses official mlx-lm 0.28.3 release with Python 3.9 compatibility fixes for `qwen3_next`
  - Adds support for newer architectures (Klear, qwen3_next, etc.)
  - Git pin removed in favor of stable PyPI release

### Validation
- ✅ All 256 tests pass (9 skipped)
- ✅ Runtime compatibility correctly detects:
  - GGUF/PyTorch models → `runtime_compatible: false` (framework mismatch)
  - Supported MLX models → `runtime_compatible: true`
  - Unsupported architectures → `runtime_compatible: false` with descriptive `reason`
  - Klear-46B verified working with mlx-lm 0.28.2


### Notes
- Human output columns controlled by CLI flags (documentation in README.md, separate from JSON spec)
- This addresses the root cause discovered in Issue #36: GGUF models show "healthy" but are not executable with mlx-lm

## 2.0.0-beta.3 — 2025-09-18

**Feature Complete**: Full 1.1.1 parity achieved with Clone implementation (ADR-007 Phase 1) and APFS filesystem detection fixes.

### Added
- **Clone command implementation** (MAJOR):
  - Complete `mlxk2 clone` with ADR-007 Phase 1: Same-Volume APFS strategy
  - APFS Copy-on-Write optimization for instant cloning
  - Isolated temp cache with user cache safety
  - Health check integration via `health_from_cache`
  - Feature-gated behind `MLXK2_ENABLE_ALPHA_FEATURES=1`
- **JSON API 0.1.4 specification**:
  - Clone operation schema and documentation
  - Complete schema validation coverage for all 10 JSON commands
  - Schema tests for `list`, `show`, `health`, `pull`, `rm`, `clone`, `version`, `push`, `run`, `server`

### Fixed
- **APFS filesystem detection**: SMB/network mounts now correctly detected as Non-APFS
- **Push APFS warnings**: Non-APFS cache setups now display filesystem warnings

### Testing
- **Comprehensive test coverage**: 254/254 tests passing, 11 skipped
- **Clone operation tests**: 43 tests covering APFS, volume detection, health integration
- **Live validation**: 3 live clone + push tests with real HuggingFace models

## 2.0.0-beta.3-local — 2025-09-14

**Feature Complete Beta**: 1.x parity achieved. All core functionality implemented with clean experimental separation.

### Added
- **Run command implementation** (MAJOR):
  - Complete `mlxk2 run` with interactive and single-shot modes
  - Streaming and batch generation with parameter controls (`--temperature`, `--top-p`, `--max-tokens`)
  - Chat template integration and conversation history tracking
  - Interrupt handling (Ctrl-C) with graceful recovery and session reset
  - Enhanced run with future features (system prompts, reasoning model support)
- **MLXRunner core engine** (ported from 1.x):
  - `mlxk2.core.runner` package with modular architecture
  - Dynamic token limits (full context for run, half-context for server)
  - Stop token filtering and reasoning model detection
  - Thread-safe model loading, memory management, and cleanup
- **Server implementation**:
  - OpenAI-compatible endpoints (`/v1/completions`, `/v1/chat/completions`, `/v1/models`, `/health`)
  - SSE streaming with SIGINT-robust supervisor mode (deterministic shutdown/restart)
  - Model hot-swapping and thread-safe memory management
  - Half-context token limits for DoS protection
- **Experimental feature separation**:
  - Push command hidden behind `MLXK2_ENABLE_EXPERIMENTAL_PUSH=1` environment variable
  - Clean beta/experimental boundaries for stable release classification

### Changed
- **Feature status**: All core commands now complete
  - README/docs updated: Run status "Pending" → "Complete"
  - Feature parity with 1.x stable releases achieved
  - Stable version reference updated to 1.1.1
- **Test architecture**:
  - Default suite: **184 passed, 30 skipped** (stable features only)
  - Experimental: **205 passed, 9 skipped** (with `MLXK2_ENABLE_EXPERIMENTAL_PUSH=1`)
  - Clean separation ensures beta testing covers stable features only
- **Runner architecture**:
  - Modular design with focused helpers: `token_limits.py`, `chat_format.py`, `reasoning_format.py`, `stop_tokens.py`
  - API compatibility preserved for existing integrations and test patches

### Fixed
- **Pull operation cache pollution (Issue #30)**:
  - Added preflight access check with `preflight_repo_access()` to validate repository accessibility
  - Prevents cache pollution from attempting downloads of gated/private/missing repos
  - Surfaces clear "Access denied" guidance with `HF_TOKEN` hints before any download
  - Robust error handling across different `huggingface_hub` versions
- **Test stability**:
  - Pull network timeout test fixed for environments without `HF_TOKEN`
  - All push tests now properly gated behind environment variable (no unexpected failures)
  - Default test runs require no external dependencies or credentials
- **Documentation accuracy**:
  - Feature status corrected across README/TESTING to reflect actual implementation
  - Test count documentation updated to reflect stable vs experimental separation

### Implementation Milestones
- **Complete 1.x parity**: All core functionality (list, health, show, pull, rm, run, serve) fully implemented
- **Production ready**: Comprehensive testing across Python 3.9-3.13 with isolated cache system
- **Clean architecture**: Experimental features properly isolated, beta definition clarified
- **GitHub issues resolved**: Run implementation, interactive mode, streaming support, feature parity

### Tests & Docs
- **Comprehensive test coverage**: 31+ tests for run command (interactive, parameters, error handling)
- **TESTING.md**: Clear guidance on stable (184) vs experimental (+21) test runs
- **Multi-Python verification**: All tests passing across supported Python versions
- **Skip breakdown documented**: 21 push tests, 1 live test, 8 other opt-in tests

### Notes
- 2.0.0-beta.3 represents **complete feature parity** with 1.x stable releases
- Ready for production use as comprehensive 1.x alternative
- Experimental features cleanly separated for future development

## 2.0.0-alpha.3 — 2025-09-08

Port Issue #31 (lenient MLX detection) to 2.0; refine human list behavior.

Hard split: 1.x code and tests have been removed from this branch to avoid confusion and license duality. Use the `main` branch for 1.x (MIT).

### Added
- Detection helpers (README front‑matter + tokenizer):
  - Framework=MLX when README front‑matter `tags` includes `mlx` or `library_name: mlx`, in addition to `mlx-community/*`.
  - Type=chat when tokenizer has `chat_template`, or name hints (`instruct`/`chat`), or `config.model_type == 'chat'`.
  - Unified `build_model_object(...)` used by `list` and `show` to ensure consistent fields.
- Tests:
  - Offline: front‑matter and tokenizer detection for both `list` and `show`.
  - Human output: verifies default/verbose/all filtering semantics.
  - Live (opt-in): `tests_2.0/live/test_list_human_live.py` checks human list variants against a real HF cache (marker `-m live_list`).
  - Push (offline): branch-missing tolerance and retry on "Invalid rev id" with `--create`.

### Changed
- Human list (default): shows only MLX chat models (safer for run/server selection).
- Human list `--verbose`: shows all MLX models (chat + base).
- Human list `--all`: shows all frameworks (MLX, GGUF, PyTorch).
- `show` uses the same detection helpers as `list`; respects `HF_HOME` via `get_current_model_cache()`.

### Docs
- SECURITY.md: clarified experimental push scope and network behavior (explicit only; no background traffic).
- README.md: added “Privacy & Network” bullet; updated version strings to alpha.3.
 - README.md: noted hard split — 1.x lives on `main` (MIT), this branch is 2.x (Apache‑2.0).

### Notes
- No JSON API schema changes; spec remains 0.1.3.
 
### Fixed
- Push: tolerate missing target branches; with `--create`, proactively create the branch and retry the upload once. No‑op uploads still create the branch when `--create` is provided.

## [1.1.1-beta.2] - 2025-09-06

### Feature: Lenient MLX Detection for Private Repos (Issue #31)
- Problem: `run` only accepted `mlx-community/*` models; private/cloned MLX repos (in MLX format) appeared as "PyTorch | base" and were rejected.
- Solution: Added README/tokenizer-based detection to recognize MLX/chat models outside `mlx-community`.
- Details:
  - Tokenizer: If `tokenizer_config.json` contains a non-empty `chat_template` → Type = `chat` (highest priority).
  - README front matter (YAML, lenient parse):
    - `tags` contains `mlx` OR `library_name: mlx` → Framework = `MLX`.
    - `pipeline_tag: text-generation` OR `tags` contain `chat`/`instruct` → Type = `chat`.
    - `pipeline_tag: sentence-similarity` OR `tags` contain `embedding` → Type = `embedding`.
  - Fallback unchanged: `.gguf` → `GGUF`; else `safetensors/bin` → `PyTorch`; else `Unknown`. Type fallback by name substrings (`instruct/chat` → chat; `embed` → embedding; else base).

### CLI Behavior (Schema Unchanged)
- `mlxk show` now displays `Type: <chat|embedding|base>` when detected.
- `mlxk list --all` includes a `TYPE` column; default `mlxk list` now shows chat-capable MLX models only (strict view).
- `mlxk run` now accepts MLX repos identified via README (not only `mlx-community/*`).

### Implementation
- New helper: `mlx_knife/model_card.py` (no deps) to read README front matter and tokenizer hints; fully fail-safe.
- Updated detection in `mlx_knife/cache_utils.py`:
  - `detect_framework(...)` consults README hints before file-type fallback.
  - New `detect_model_type(...)` implements priority order.
  - `run_model(...)` imports runner module for easier test monkeypatching.

### Tests
- Added unit tests: `tests/unit/test_model_card_detection.py`.
- Server test stability and safety improvements:
  - RAM-aware model gating now combines size-token heuristics with `mlxk show` data (disk size + quantization) for more reliable estimates.
  - Fixed MoE size parsing (prefers tokens like `8x7B` over partial `7B` matches).
  - Robust server process guard ensures clean shutdown on Ctrl-C/SIGTERM (prevents orphaned Python processes using excessive memory).
  - Configurable safety/estimation factors via environment variables (see TESTING.md).
- All tests passing locally on Apple Silicon across Python 3.9–3.13: 166/166.

Note: GitHub tag/version uses `1.1.1-beta.2`. PyPI release uses PEP 440 `1.1.1b2`.

## 2.0.0-alpha.2 — 2025-09-05

Experimental `push` (upload only) and documentation/testing refinements.

### Added
- `push` (experimental, M0): Upload a local folder to Hugging Face using `upload_folder`.
  - Safety: `--private` required in alpha.
  - Quiet JSON: With `--json` (without `--verbose`) suppress progress bars/console logs; hub logs are captured in `data.hf_logs`.
  - No-op detection: Prefer hub signal (“No files have been modified… Skipping…”). Sets `no_changes: true`, clears `commit_sha/commit_url`, and sets `uploaded_files_count: 0`.
  - Offline preflight: `--check-only` analyzes the local workspace and returns `data.workspace_health` (index/weights/LFS/partials) without network.
  - Dry-run planning: `--dry-run` computes a plan vs remote (uses `list_repo_files`), returns `dry_run: true`, `dry_run_summary {added, modified:null, deleted}`, and sample `added_files`/`deleted_files` (up to 20). Honors default ignores and merges `.hfignore`.
  - Uploaded file count: Remains `null` when hub does not return per-file operations; no heuristic guessing.

### Docs
- TESTING.md: Added “Reference: Push CLI and JSON”, `--dry-run` examples, and a mini matrix (default vs markers/opt-in).
- CLAUDE.md: Updated Current Focus/Decisions + session summary for push quiet mode, no-op, `--dry-run`.

### Tests
- Offline push tests added/extended, including dry-run planning; live push remains opt-in via `wet`/`live_push` markers and required env vars.

## [1.1.1-beta.1] - 2025-09-01

### Fix: Strict Health Completeness for Multi‑Shard Models (Issue #27)
- Problem: Health reported some multi‑part downloads as OK with missing/empty shards (false positives).
- Solution: Backported 2.0 health rules to 1.x with index‑aware validation, pattern detection, and robust corruption checks.
- Details:
  - Config validation: `config.json` must exist and be a non‑empty JSON object.
  - Index‑aware: If `model.safetensors.index.json` or `pytorch_model.bin.index.json` exists, every referenced shard must exist, be non‑empty, and not be a Git LFS pointer file.
  - Pattern fallback policy: If pattern shards like `model-XXXXX-of-YYYYY.*` are present but no index file exists, the model is considered unhealthy (parity with 2.0 policy).
  - Partial/tmp markers: Any `*.partial`, `*.tmp`, or names containing `partial` anywhere under the snapshot mark the model as unhealthy.
  - LFS detection: Recursive scan flags suspiciously small files (<200B) that contain the Git LFS pointer header.
  - Single‑file weights: Non‑empty `*.safetensors`, `*.bin`, or `*.gguf` without pattern shards remain supported and healthy if not LFS pointers.
- Impact: “Healthy” now reliably means “complete and usable” for automation and CLI workflows.
- Tests: Added `tests/unit/test_health_multishard.py` covering complete/missing/empty shards, pointer detection, pattern‑without‑index policy, partial markers, and PyTorch index parity.

Note: GitHub tag/version uses `1.1.1-beta.1`. PyPI release uses PEP 440 `1.1.1b1`.

## 2.0.0-alpha.1 — 2025-08-31

- New JSON-first CLI (`mlxk2`, `mlxk-json`); `--json` for machine-readable output (new vs 1.0.0).
- Human output by default: improved formatting, new Type column, relative Modified; MLX-only compact view with `--all`, `--health`, `--verbose` flags.
- Stricter health checks for sharded models (Issue #27); robust model resolution (fuzzy, `@hash`); `rm` cleans whole model and locks.
- Packaging/tooling: dynamic versioning; multi-Python test script; Python 3.9–3.13; timezone-aware datetimes.
- **Not included yet: server and run** (use 1.x).

## [1.1.0] - 2025-08-26 - **STABLE RELEASE** 🚀

### Production Readiness & Enhanced Testing 🧪
- **First Stable Release Since 1.0.4**: Comprehensive beta testing cycle complete
- **Isolated Test System**: 150/150 tests passing with pristine user cache protection
  - **3-Category Test Strategy**: Isolated cache (78 tests) + Server tests (@pytest.mark.server) + Future framework diversity
  - **User Cache Protection**: Tests use temporary isolated caches - user cache stays completely clean
  - **Real Model Validation**: End-to-end tests using `hf-internal-testing/tiny-random-gpt2` (~12MB) in isolation
  - **Automatic Test Downloads**: No manual model setup required for standard test suite
  - **Parallel Testing**: No cache conflicts between test runs, improved CI reliability
- **Multi-Python Support**: Full compatibility verified for Python 3.9, 3.10, 3.11, 3.12, 3.13
- **All Critical Issues Resolved**: Issues #21, #22, #23 thoroughly tested and production-ready

### Technical Improvements 🔧
- **Test Infrastructure Revolution**: Complete migration from mocked tests to isolated real-world validation
- **Cache Isolation System**: `temp_cache_dir` + `patch_model_cache` fixtures ensure test isolation
- **Performance Optimization**: Fast CI with small test models, comprehensive validation with server tests
- **Developer Experience**: Clean setup process - only Python + test dependencies required
- **Test Reliability**: Reproducible results independent of user's existing model cache

---

## [1.1.0-beta3] - 2025-08-25

### Critical Bug Fixes 🐛
- **Issue #21**: Empty Cache Directory Crash - **RESOLVED**
  - **Root Cause**: `mlxk list` crashed with `FileNotFoundError` on fresh installations  
  - **Fix**: Added `MODEL_CACHE.exists()` checks in `list_models()` function
  - **Impact**: MLX-Knife now works correctly on fresh installations without pre-existing cache
  - **Test Coverage**: Added `test_list_models_real_empty_cache()` regression test

- **Issue #22**: urllib3 LibreSSL Warning on macOS Python 3.9 - **RESOLVED**
  - **Root Cause**: Every MLX-Knife command showed SSL compatibility warning on macOS system Python
  - **Fix**: Central warnings suppression in `__init__.py` before any imports that use urllib3
  - **Impact**: Clean command output on macOS system Python 3.9 with LibreSSL
  - **Scope**: Only affects macOS system Python 3.9, no impact on other environments

- **Issue #23**: Double rm Execution Problem - **RESOLVED**
  - **Root Cause**: `mlxk rm model@hash` required two executions - first left broken state, second completed
  - **Fix**: Changed from partial `snapshots/<hash>` deletion to complete model directory removal
  - **Enhancement**: Added intelligent lock cleanup system with user-friendly prompts
  - **Impact**: Single execution removes models completely + optional HuggingFace lock cleanup
  - **Features**: Interactive confirmation, `--force` parameter, robust corrupted model handling

### Enhanced Cache Management 🧹
- **Lock Cleanup System**: Addresses upstream HuggingFace FileLock accumulation issue
  - User-friendly prompt: "Clean up cache files? [Y/n]" 
  - `--force` parameter skips all confirmations for automation
  - Robust error handling with warnings (never fails on lock cleanup issues)
- **Extended rm Command**: Now handles all model states (healthy, corrupted, empty snapshots)
- **Superior UX**: Cleaner cache management than official HuggingFace CLI tools

### Test Infrastructure Improvements 🧪
- **Test Count**: Updated to 140/140 tests passing (+5 new tests for Issue #23)
- **Regression Coverage**: New tests for empty cache, corrupted models, lock cleanup scenarios
- **Force Parameter Testing**: Comprehensive coverage of interactive vs force mode behavior
- **Integration Test Robustness**: All edge cases now covered with real model testing

### Documentation Updates 📚
- **Version Updates**: All documentation updated to reflect 1.1.0-beta3 status
- **Testing Guide**: Updated test counts and new test scenarios in TESTING.md
- **Issue Documentation**: Added HUGGINGFACE_LOCK_ISSUES.md with upstream context
- **Lock Cleanup Documentation**: Clear explanation of MLX-Knife's cache management advantages

## [1.1.0-beta2] - 2025-08-22

### Critical Bug Fixes 🐛
- **Issue #19**: Server Response Truncation at ~1000 Words - **RESOLVED**
  - **Root Cause**: Server hardcoded `--max-tokens 2000` overrode dynamic limits from 1.1.0-beta1
  - **Fix**: Changed CLI `--max-tokens` default from `2000` to `None`, enabling model-aware dynamic limits
  - **Impact**: Large context models (Qwen3-30B, Llama-3.3-70B) now work at full capacity by default
  - **Validation**: Server startup shows "model-aware dynamic limits" instead of hardcoded values

- **Issue #20**: End-Token Visibility in Non-Streaming Mode - **RESOLVED**  
  - **Root Cause**: `generate_batch()` lacked End-Token filtering present in `generate_streaming()`
  - **Fix**: Ported filtering logic with new `_filter_end_tokens_from_response()` method
  - **Affected**: `mlxk run model "prompt" --no-stream` and Server API `"stream": false`
  - **Impact**: No more end tokens appearing in the final output in non-streaming mode

### Enhanced
- Better default for `--max-tokens`: `None` → model-aware limits
- Improved consistency between streaming and non-streaming generation
- Clearer server logs indicating active token policies

### Technical
- 15 new tests across server and CLI to validate token policies
- Internal refactoring for token handling to avoid duplication

## [1.1.0-beta1] - 2025-08-21

### Added
- Dynamic model-aware token limits (context-length sensitive)
- CLI `--max-tokens` default changed to `None` (was 2000)
- Server leverages the same dynamic limits

### Improved
- End-token filtering consistency across streaming and non-streaming modes
- Robustness in model loading and memory management

### Tests
- 114/114 tests passing
- Server tests behind `@pytest.mark.server` (opt-in)

## [1.0.4] - 2025-08-19

### Fixed
- **Issue #14**: Interactive chat self-conversation bug resolved
  - MLX models no longer continue generating conversation turns after their response
  - Added context-sensitive chat stop tokens: `\nHuman:`, `\nAssistant:`, `\nYou:`, `\nUser:` 
  - Smart priority system: native model stop tokens checked first, chat tokens as fallback
  - Affects both `mlxk run` and `mlxk server` modes
  - Comprehensive regression test suite added with 15 tests across 7+ MLX models

### Enhanced
- **Web UI Complete Overhaul** (simple_chat.html):
  - 🦫 Branding update: Replaced 🔪 with 🦫 (Beaver) emoji for friendlier appearance
  - 💾 Model persistence: Selected model survives browser reload via localStorage  
  - 📚 Chat history persistence: Full conversation history preserved across sessions
  - 🔄 Smart model switching: Choice to keep or clear chat history when switching models
  - 🌐 Responsive design: Full viewport height utilization, optimized screen space usage
  - 🎯 Clear UX: "Clear Chat" instead of ambiguous "Clear" button
  - 🏴 English dialogs: Custom modal dialogs replace German OS dialogs

### Added
- **Automated Server Testing Infrastructure**:
  - RAM-aware model filtering: Automatic model selection based on available system RAM
  - Self-contained server management: Automatic MLX Knife server lifecycle in tests
  - macOS compatible: Graceful handling of permission restrictions
  - Opt-in testing: Server tests marked `@pytest.mark.server`, excluded from default `pytest`
  - Comprehensive testing guide with RAM-based model recommendations

### Technical
- Context-aware token decoding maintains backward compatibility
- Native model stop tokens preserved, chat tokens only as fallback
- Exception-safe server test infrastructure with automatic cleanup
- Complete TESTING.md documentation for server-based regression testing
- All existing tests continue to pass (114/114)

## [1.0.3] - 2025-08-18

### Added
- **Issue #13**: Hash-based disambiguation for ambiguous model names
  - Use commit hashes to disambiguate between multiple matching models
  - Example: `mlxk show Llama@de2dfaf5` automatically resolves to `mlx-community/Llama-3.3-70B-Instruct-4bit`
  - Pure local resolution, no external API calls, offline-capable
- **Issue #6**: Repository name length validation for HuggingFace Hub 96-character limit
  - Pre-validation with clear error message before attempting download
  - Better user experience with immediate feedback on invalid repository names

### Fixed
- **Issue #7**: Fixed health check inconsistency in show command with fuzzy model names
  - `mlxk show Phi-3` vs `mlxk show mlx-community/Phi-3-mini-4k-instruct-4bit` now show identical health status
  - Unified health check logic to use resolved model names for consistent results

### Enhanced
- Enhanced short commit hash support with local resolution
- Improved model name disambiguation logic
- Real user workflow support - see hashes in `mlxk list`, use directly in other commands

### Technical
- 9 new comprehensive test cases added (TestIssue6RepositoryNameValidation, TestShowModelHealthConsistency, TestIssue13HashDisambiguation)
- All 114 unit tests passing on Apple Silicon
- Improved error handling and user experience across all model resolution scenarios

## [1.0.2] - 2025-08-18

### Fixed
- **Issue #11**: Fixed HF_HOME environment variable handling - MLX Knife now correctly uses `$HF_HOME/hub` for model storage, consistent with HuggingFace standard
- **Issue #9**: Fixed silent failure when removing corrupted models with empty snapshots directories
- **Cache Consistency**: Unified cache path logic - both default (`~/.cache/huggingface/hub`) and custom (`$HF_HOME/hub`) paths now consistently use `/hub` subdirectory

### Enhanced  
- **Download Throttling**: Improved adaptive throttling for household-friendly downloads (512KB chunks, 2-3s delays for large files)
- **Migration Warning**: Added helpful warning when models are found in legacy cache locations with clear migration instructions
- **Memory Management**: Enhanced exception-safe resource cleanup and baseline tracking

### Technical
- **Dependencies**: Updated to latest tested versions (huggingface-hub 0.34.0+, mlx 0.28.0+, fastapi 0.116.0+)
- **Python Support**: Full compatibility verified on Python 3.9-3.13
- **Test Suite**: All 105 tests passing with real MLX models on Apple Silicon

## [1.0.1] - 2025-08-15

### Changed
- **Description Update**: Changed package description to "ollama-style CLI for MLX models on Apple Silicon"

## [1.0.0] - 2025-08-15

### Changed
- **STABLE RELEASE**: MLX Knife 1.0.0 officially stable and ready for production use
- **PyPI Publication**: Now available on PyPI for easy installation via `pip install mlx-knife`
- **CLI-Only Policy**: Officially designated as CLI-only tool (Python API access not officially supported)
- **Documentation**: Updated all documentation to reflect stable 1.0.0 release status

## [1.0-rc3] - 2025-08-14

### Added
- **Issue 1**: Partial name filtering for `mlxk list` command (e.g., `mlxk list Phi-3`)
- **Issue 2**: Fuzzy matching for single-model commands (`mlxk show Phi-3`, `mlxk run Phi-3`)
- **Issue 3**: Default `mlxk health` behavior (no `--all` flag required)
- Comprehensive test coverage for all new fuzzy matching features
- Smart ambiguity resolution with helpful error messages

### Enhanced
- All single-model commands now support partial name matching
- Case-insensitive model name searching
- Improved user experience with intelligent model resolution
- Expanded test suite from 96 to 104 tests (104/104 passing ✅)

### Fixed
- Health command now works without requiring `--all` flag
- Better error handling for ambiguous model specifications
- Enhanced fuzzy matching logic with fallback mechanisms

## [1.0-rc2] - 2025-08-13

### Enhanced
- Robust exception handling during model loading with guaranteed cleanup
- Protection against nested context manager usage 
- Safe cleanup that handles partial loading failures
- Exception-resilient cache clearing (won't fail if cache operations error)
- Safe tokenizer attribute access using getattr() with defaults
- Graceful memory stats handling when metrics unavailable
- Comprehensive unit test coverage for all memory management edge cases

### Fixed
- Memory management edge cases in MLXRunner context manager
- Exception safety during model loading and cleanup operations
- Improved error handling for partial model loading failures

## [1.0-rc1] - 2025-08-12

### Added
- Initial release candidate
- Full MLX model support for Apple Silicon
- OpenAI-compatible API server
- Web chat interface
- Multi-Python support (3.9-3.13)
- Comprehensive test suite (86/86 passing)

## Known Issues
- See GitHub Issues for tracking
 
## 2.0.0‑beta.3 (local)

- Server robustness and API polish
  - Supervisor default: Uvicorn runs as subprocess in its own process group; Ctrl‑C terminates deterministically and allows immediate restart.
  - HTTP mapping: 404 for unknown/failed model loads; 503 during shutdown; preserve HTTPException codes from helpers.
  - Streaming (SSE):
    - Happy path: initial chunk, per‑token chunks, final chunk, then `[DONE]`.
    - Interrupt path: on `KeyboardInterrupt` emit clear interrupt marker and close promptly.
  - Token limits: server mode uses half of context length; explicit `max_tokens` respected.
  - Noise reduction: chat streaming debug prints gated behind `MLXK2_DEBUG`.

- Testing
  - Added focused server API tests for `/v1/models`, 404/503 mapping, SSE happy/interrupt, and server‑side token limit propagation.
  - Global suppression of macOS Python 3.9 `urllib3` LibreSSL warning in tests; runtime already suppressed.

- Docs
  - README/TESTING touch‑ups pending flip; CLAUDE.md tracks SSE UX roadmap (anti‑buffering headers, optional heartbeats, status/interrupt endpoints).
