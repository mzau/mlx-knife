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

## Future Versions (Planned)

### 0.2.0 (TBD - Phase 1, when model field stabilizes)

**Proposed changes:**
- Make `model.id` required when `outcome == "passed"` (enforce for model tests)
- Add `model.framework_version` (mlx-lm version for reproducibility)
- Standardize `stop_tokens.workaround` enum (based on collected data)
- Add `test_type` enum (stop_tokens, performance, health, etc.)

**Migration:**
- Scripts will backfill `model.framework_version` from git history
- `stop_tokens.workaround` will be normalized (free text → enum)
- Old reports remain valid (historical data preserved)

**Breaking changes:**
- TBD based on Phase 0 learnings

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
