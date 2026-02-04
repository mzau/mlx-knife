# Schema Migrations

This document tracks schema evolution for MLX Knife test reports.

## Version History

### 0.1.0 (2025-11-16) - Phase 0 Initial

**Status:** Experimental, evolving organically

**Required fields:**
- `schema_version`: "0.1.0"
- `timestamp`: ISO 8601 datetime
- `mlx_knife_version`: SemVer string
- `test`: pytest nodeid
- `outcome`: passed|failed|skipped

**Optional fields:**
- `duration`: Test duration (seconds)
- `model`: Model metadata (id, size, family, variant)
- `performance`: Performance metrics (tokens/sec, RAM, duration)
- `stop_tokens`: Stop token data (configured, detected, workaround, leaked)
- `system`: Platform info (OS, Python, MLX, hardware)
- `metadata`: Catch-all for experiments (no constraints)

**Design rationale:**
- Minimal required fields keep reporting lightweight
- Optional sections allow gradual data collection improvement
- `metadata` object enables experimentation without schema changes

**Breaking changes from nothing:** N/A (initial version)

**Migration:** N/A

---

### 0.2.0 (2025-12-08) - Scheduling-Enhanced

**Status:** Stable (used in 2.0.4-beta.3+)

**Added fields:**
- `model.framework`: Model framework identifier (e.g., 'MLX', 'GGUF')
- `model.quantization`: Quantization format (e.g., '4bit', '8bit', 'fp16')
- `performance.model_load_time_s`: Model loading time (critical for scheduling)
- `performance.time_to_first_token_s`: User-perceived latency metric
- `performance.cleanup_time_s`: Resource release timing
- `performance.peak_ram_gb`: Peak RAM usage during inference
- `performance.stable_ram_gb`: Steady-state RAM after warmup
- `system.hardware_profile`: Detailed hardware profiling (Mac model, cores, GPU)
- `system_health`: System health metrics (swap, RAM, zombies, quality flags)
- `timeline`: Optional detailed execution timeline for bottleneck analysis

**Design rationale:**
- Enables memory-based scheduling decisions (ADR-016)
- Supports hardware profiling for benchmark clustering
- Quality assessment flags for benchmark validation
- Backward compatible: v0.1.0 reports remain valid

**Breaking changes:** None (additive only)

**Migration:** N/A (automatic upgrade)

---

### 0.2.1 (2026-01-09) - Inference Modality

**Status:** Stable (used in 2.0.4-beta.7+)

**Added fields:**
- `metadata.inference_modality`: Type of inference performed
  - Values: `"vision"` | `"text"` | `"audio"` | `"video"`
  - Purpose: Differentiate Vision/Text inference for multimodal models
  - Example: Pixtral can do both Vision (with --image) and Text (without --image)

**Design rationale:**
- Vision-capable models perform BOTH Vision and Text inference
- Benchmark reports need to differentiate these for accurate statistics
- Per-model stats can now show: "Vision: 136s (90%), Text: 15s (10%)"
- Future-proof for audio/video/multimodal inference types

**Automatic detection:**
- Vision inference: Tests with `vision_model_key` fixture OR `--image` CLI arg
- Text inference: Tests with `text_model_key` fixture OR no `--image` arg
- Pipe tests: Explicit per-phase tagging (e.g., `[vision_phase]`, `[text_phase]`)

**Backward compatible:**
- Old reports without `inference_modality` remain valid
- Tools gracefully degrade: show only total time for legacy entries
- Mixed data (old + new) shows "Unknown (legacy)" breakdown

**Breaking changes:** None (additive only)

**Migration:** N/A (automatic upgrade, optional field)

---

### 0.2.2 (2026-01-30) - Precise Test Timing

**Status:** Stable (used in 2.0.4-beta.9+)

**Added fields:**
- `test_start_ts`: Unix epoch timestamp (seconds) when test execution started
- `test_end_ts`: Unix epoch timestamp (seconds) when test execution ended

**Design rationale:**
- Enables **effective runtime analysis** by excluding idle periods (Memory Gates, setup, teardown)
- Accurate correlation with memmon samples (memory monitoring tool)
- Faster post-processing (no ISO 8601 parsing needed)
- Example: Test with 30s wall clock duration but only 22s compute time (8s Memory Gate)

**Implementation:**
- Captured via pytest hooks: `pytest_runtest_setup`, `pytest_runtest_teardown`, `pytest_runtest_makereport`
- Stored in `item.stash` during test execution, added to `user_properties` in report
- Both fields are optional for backward compatibility

**Use cases:**
- Filter memmon samples by test window: `test_start_ts <= sample.ts <= test_end_ts`
- Calculate effective duration: `effective_s = test_end_ts - test_start_ts` (excludes pytest overhead)
- Identify idle time: `idle_s = duration - (test_end_ts - test_start_ts)`

**Backward compatible:**
- Old reports without these fields remain valid
- Tools can fall back to: `timestamp` (ISO 8601 test end), `duration` (wall clock)
- Legacy test_start approximation: `parse_iso(timestamp) - duration` (±5s accuracy)

**Breaking changes:** None (additive only)

**Migration:** N/A (automatic upgrade, optional fields)

---

## Future Versions (Planned)

---

### 1.0.0 (TBD - Phase 3, community-ready)

**Proposed changes:**
- Stabilize all core fields (no more optional → required migrations)
- Add `contributor` object (for community submissions)
- Add digital signatures (for trust/verification)
- Formal deprecation policy (2-release grace period)

**Migration:**
- Full validation tooling (`mlxk report validate`)
- Automatic upgrades for old reports (`mlxk report migrate`)

---

## Schema Evolution Policy

### Phase 0 (current): Experimental
- Rapid iteration based on collected data
- Breaking changes allowed (no backward compatibility guarantees)
- Focus: Learn what data is useful

### Phase 1 (2.1+): Stabilization
- Core fields stabilize
- Backward-compatible additions only
- Deprecation warnings for breaking changes (2 releases ahead)

### Phase 2 (2.2+): Community-ready
- Strict versioning (SemVer for schemas)
- Migration scripts for all breaking changes
- Validation tooling (`mlxk report validate`)

### Phase 3 (2.3+): Production
- No breaking changes without major version bump
- Formal governance (review process, audit log)
- Long-term support (LTS) for stable schema versions

---

## Deprecation Process (Phase 2+)

1. **Announcement:** Deprecation warning in schema, docs, and CLI
2. **Grace Period:** 2 releases (e.g., 2.2 → 2.3 → 2.4)
3. **Migration Tools:** `mlxk report migrate` auto-upgrades
4. **Breaking Change:** New major version (e.g., 2.0.0 → 3.0.0)
5. **Legacy Support:** Old reports remain queryable (read-only)

---

## Contributing

Schema evolution is driven by **empirical data**:
1. Collect reports with current schema
2. Analyze: What fields are useful? What's missing?
3. Propose changes in GitHub issues (with data evidence)
4. Iterate in next schema version

**Rule:** Schema follows data, not speculation.
