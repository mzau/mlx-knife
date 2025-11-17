# MLX Knife Benchmarks

**Status:** Phase 0 - Organic Data Collection

## Architecture

This directory tracks empirical performance and compatibility data from mlx-knife's test suite.

### Phase 0 Goals (2.0.3+)

1. **Collect data organically** from E2E tests
2. **No perfect schema** - schema evolves with data
3. **Git-tracked reports** - historical trends
4. **Foundation for future** - community contributions, public database

### Directory Structure

- `reports/` - JSONL test reports (one file per release)
- `schemas/` - JSON Schema definitions (versioned)

### Current Schema

**Version:** 0.1.0 (Phase 0 - Minimal)

See `schemas/report-v0.1.schema.json` for details.

**Required fields:**
- `schema_version`, `timestamp`, `mlx_knife_version`, `test`, `outcome`

**Optional sections:**
- `model` - Model metadata
- `performance` - tokens/sec, RAM usage
- `stop_tokens` - ADR-009 validation data
- `system` - Platform info
- `metadata` - Extensible (anything)

### Generating Reports

```bash
# During E2E tests
pytest -m live_e2e tests_2.0/live/ \
  --report-output benchmarks/reports/$(date +%Y-%m-%d)-v$(mlxk --version | cut -d' ' -f2).jsonl
```

### Schema Evolution

As we collect more data, the schema will evolve:
- New fields added (backward compatible)
- Optional → Required (when stable)
- Breaking changes documented in `schemas/MIGRATIONS.md`

### Future Phases

- **Phase 1 (2.1+):** Schema formalization, validation tooling
- **Phase 2 (2.2+):** `mlxk report` CLI for manual submissions
- **Phase 3 (2.3+):** Public database, community contributions

See `docs/ADR/ADR-013-Community-Model-Quality-Database.md` for full roadmap.
