# ADR-006: mlxk2 clone Implementation - Revised Strategy

## Status
**Accepted** - 2025-09-15

**Supersedes:** ADR-005 (deprecated due to incorrect HuggingFace cache assumptions)

## Context

GitHub Issue #29 requests clone functionality for MLX Knife 2.0. After implementing ADR-005, critical findings revealed that HuggingFace Hub's `local_dir` parameter does not provide true cache isolation and can corrupt existing cache entries.

### Key Findings from ADR-005 Implementation

**Problem: HuggingFace Cache Behavior is Unreliable**
1. `snapshot_download(local_dir=target, local_dir_use_symlinks=False)` still interacts with global cache
2. Global cache corruption observed (models showing 0.0 KB after clone operations)
3. `local_dir_use_symlinks` parameter is deprecated but behavior remains unclear
4. Documentation promises cache isolation but implementation differs

**Evidence:**
```bash
# Before clone: Phi-3-mini shows 4.3 GB in cache
mlxk list --health  # Shows healthy model

# After clone with local_dir: Cache corrupted
mlxk list --health  # Shows 0.0 KB - corrupted cache entry
```

### Revised Strategy: Pull + APFS Copy + Cleanup

**Core Insight:** Instead of fighting HuggingFace Hub's undocumented cache behavior, leverage it robustly:

1. **Pull to Cache** (battle-tested, reliable)
2. **Copy Cache → Workspace** (APFS copy-on-write optimization)
3. **Delete Cache Entry** (automatic cleanup)

## Decision

Implement `mlxk2 clone` using a **Pull + Copy + Cleanup** strategy that provides robust workspace creation without relying on HuggingFace Hub's unreliable `local_dir` behavior.

## Implementation Strategy

### Core Workflow
```
1. Hidden Pull      → Download to cache (existing reliable logic)
2. Optional Health  → Validate model integrity before copy
3. APFS Copy        → Copy cache → workspace (copy-on-write efficient)
4. Cache Cleanup    → Delete cache entry (no user prompt needed)
```

### APFS Volume Optimization

**Key Advantage:** On APFS volumes (standard on macOS), file copies use copy-on-write:
- Initial copy: **No additional disk space** (metadata references only)
- Space usage: Only when files are modified in workspace
- Copy speed: Near-instantaneous for large models

**Volume Detection:**
```python
def is_same_apfs_volume(cache_path, workspace_path):
    # Check if both paths are on same APFS volume
    # Optimize copy strategy accordingly
```

### API Signature (Unchanged)
```bash
mlxk2 clone <org>/<repo>[@<revision>] <target_dir> [options]
```

**Options:**
- `--branch <branch>` - Clone specific branch/revision
- `--json` - JSON output mode
- `--quiet` - Suppress progress output
- `--no-health-check` - Skip optional health validation

### JSON Response Schema (API 0.1.4 - Unchanged)
```json
{
  "status": "success|error",
  "command": "clone",
  "data": {
    "model": "org/repo",
    "clone_status": "completed",
    "message": "Cloned successfully to ./workspace",
    "target_dir": "/abs/path/to/workspace",
    "cache_cleanup": true,
    "health_check": true
  },
  "error": null
}
```

## Implementation Details

### Phase 1: Core Clone Logic
```python
def clone_operation(model_spec, target_dir, health_check=True):
    # 1. Standard pull to cache
    pull_result = pull_operation(model_spec)
    if pull_result["status"] != "success":
        return error_response("Pull failed", pull_result["error"])

    # 2. Optional health check
    if health_check:
        health_result = health_check_cache(model_spec)
        if not health_result["healthy"]:
            return error_response("Model unhealthy", health_result)

    # 3. Copy cache to workspace
    cache_path = resolve_cache_path(model_spec)
    copy_result = apfs_optimized_copy(cache_path, target_dir)
    if not copy_result["success"]:
        return error_response("Copy failed", copy_result["error"])

    # 4. Cleanup cache entry
    cleanup_result = remove_cache_entry(model_spec)

    return success_response(copy_result, cleanup_result)
```

### Phase 2: APFS Optimization
```python
def apfs_optimized_copy(source_path, target_path):
    """Copy with APFS copy-on-write optimization where possible."""
    if is_same_apfs_volume(source_path, target_path):
        # Use APFS-optimized copy (clonefile on macOS)
        return apfs_clone_files(source_path, target_path)
    else:
        # Fall back to standard file copy
        return standard_copy(source_path, target_path)
```

### Phase 3: Cache Management
```python
def remove_cache_entry(model_spec):
    """Remove cache entry after successful workspace creation."""
    cache_path = hf_to_cache_dir(model_spec)
    if cache_path.exists():
        shutil.rmtree(cache_path)
    return {"cache_cleanup": True, "path": str(cache_path)}
```

## Benefits

1. **Robust Behavior:** Uses proven pull logic, avoids HF cache edge cases
2. **APFS Efficient:** No duplicate storage on same volume (copy-on-write)
3. **Clean Workspaces:** No cache artifacts (.cache folders, symlinks)
4. **Predictable:** No undocumented HF behavior dependencies
5. **Testable:** Each phase can be tested independently

## Security Classification

**Clone Operation: LOW RISK** (unchanged)
- Read-only operation with local file manipulation only
- No remote publication risk
- Workspace isolation maintained through file copying

## Risk Analysis

### Mitigated Risks (from ADR-005)
- ✅ **Cache Corruption:** Eliminated by using standard pull path
- ✅ **Undocumented Behavior:** No reliance on HF `local_dir` edge cases
- ✅ **Symlink Issues:** Pure file copying, no symlinks

### New Risks and Mitigations

**Risk:** Double storage usage during copy process
**Mitigation:** APFS copy-on-write optimization, volume detection

**Risk:** Cache cleanup removes model unexpectedly
**Mitigation:** Only cleanup after successful workspace creation

**Risk:** Interrupted copy leaves partial workspace
**Mitigation:** Atomic operations, rollback on failure

## Testing Strategy

### Test Categories
1. **Pull Integration:** Verify pull-phase works correctly
2. **Copy Operations:** Test APFS vs standard copying
3. **Cache Management:** Validate cleanup behavior
4. **Error Handling:** Test failure modes at each phase
5. **JSON Schema:** API 0.1.4 compliance validation

### Environment Variables
- `MLXK2_ENABLE_EXPERIMENTAL_CLONE=1` - Enable clone tests in CI
- `MLXK2_LIVE_CLONE=1` - Enable live network tests (opt-in)

## Timeline

**Target:** Complete within current session
- Implementation: 1-2 hours (reuse existing pull logic)
- Testing: 1 hour (focused on copy + cleanup logic)
- Documentation: 30 minutes

## Success Criteria

1. ✅ **Reliable Clone:** No cache corruption, predictable behavior
2. ✅ **APFS Optimized:** Minimal storage overhead on macOS
3. ✅ **Clean Workspaces:** No cache artifacts in target directories
4. ✅ **JSON API Compliance:** Full 0.1.4 schema validation
5. ✅ **Robust Error Handling:** Graceful failure at each phase

## References

- **Supersedes:** ADR-005 (retained for historical reference)
- GitHub Issue #29: Clone functionality request
- HuggingFace Hub Documentation: `snapshot_download` behavior analysis
- APFS Technical Reference: Copy-on-write filesystem optimization