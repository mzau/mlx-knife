# ADR-007: Clone Implementation Fixed Strategy

**Status:** Accepted
**Date:** 2025-01-16
**Supersedes:** ADR-006 (Clone Implementation Revised)

## Context

The clone implementation following ADR-006 has critical bugs that make it unsuitable for release:

1. **Destructive Cache Cleanup:** Always deletes user cache after copy, even when model pre-existed
2. **Commit Hash Mismatch:** Can copy outdated snapshots when remote HEAD differs from cached version
3. **Inconsistent Behavior:** User unexpectedly loses models from cache

Analysis revealed that the Pull+Copy+Cleanup strategy is fundamentally flawed for a "clone" operation, which should be non-destructive by nature.

## Decision

### Phased Implementation Strategy

**Core Principle:** Cache must be APFS (for optimization), workspace flexibility increases per phase.

### Phase 1: Same-Volume APFS (2.0.0-beta.3)

**Constraints:**
- Cache: APFS required
- Workspace: APFS required, same volume as cache
- Optimization: Direct APFS copy-on-write

**Workflow:**
```
1. Validate cache and workspace both on same APFS volume
2. Create isolated temp cache on same volume as workspace
3. Pull model to temp cache (isolated from user cache)
4. APFS clone temp cache → workspace (instant, zero space initially)
5. Delete temp cache (cleanup)
```

### Phase 2: Cross-Filesystem Support (eventually, when clone and push is non-Alpha)

**Constraints:**
- Cache: APFS required (for temp cache optimization)
- Workspace: Any filesystem supported
- Optimization: APFS CoW for temp cache, standard copy to workspace

**Workflow:**
```
1. Validate cache on APFS (workspace can be any filesystem)
2. Create isolated temp cache on APFS volume (cache volume)
3. Pull model to temp cache via APFS optimization
4. Copy temp cache → workspace (standard copy if cross-filesystem)
5. Delete temp cache (cleanup)
```

## Implementation Matrix

### Filesystem Compatibility Table

| Cache FS | Workspace FS | Same Volume | Phase 1 Support | Phase 2 Support | Copy Method | Performance |
|----------|--------------|-------------|------------------|------------------|-------------|-------------|
| APFS | APFS | Yes | ✅ Supported | ✅ Supported | APFS CoW Direct | ⚡ Instant |
| APFS | APFS | No | ❌ Error | ✅ Supported | Temp+Standard | 🐌 2x Copy |
| APFS | HFS+ | No | ❌ Error | ✅ Supported | Temp+Standard | 🐌 2x Copy |
| APFS | ExFAT | No | ❌ Error | ✅ Supported | Temp+Standard | 🐌 2x Copy |
| APFS | NFS | No | ❌ Error | ⚠️ Warning | Temp+Network | 🐌🐌 Slow |
| APFS | SMB/CIFS | No | ❌ Error | ⚠️ Warning | Temp+Network | 🐌🐌 Slow |
| HFS+ | Any | Any | ❌ Error | ❌ Error | N/A | N/A |
| NFS | Any | Any | ❌ Error | ❌ Error | N/A | N/A |
| SMB/CIFS | Any | Any | ❌ Error | ❌ Error | N/A | N/A |

### Data Flow Scenarios

#### Scenario A: Phase 1 Optimal (Same APFS Volume)
```
User Cache (APFS): /Users/me/.cache/huggingface/hub/
Target Workspace:   /Users/me/projects/mymodel/
Temp Cache:         /Users/me/.mlxk2_temp_12345/

Flow:
[Remote] --pull--> [Temp Cache] --APFS CoW--> [Workspace]
                        ↑              ↓
                   Zero space      Instant copy
```

#### Scenario B: Phase 2 Cross-Filesystem
```
User Cache (APFS):  /Users/me/.cache/huggingface/hub/
Target Workspace:   /Volumes/ProjectSSD/myapp/models/
Temp Cache:         /Users/me/.mlxk2_temp_12345/

Flow:
[Remote] --pull--> [Temp Cache] --Standard Copy--> [Workspace]
                        ↑              ↓
                   APFS CoW       Full copy
```

#### Scenario C: Phase 2 Network Workspace (NFS/SMB)
```
User Cache (APFS):  /Users/me/.cache/huggingface/hub/
Target Workspace:   /Volumes/NetworkShare/models/  (NFS or SMB/CIFS)
Temp Cache:         /Users/me/.mlxk2_temp_12345/

Flow:
[Remote] --pull--> [Temp Cache] --Network Copy--> [Network Workspace]
                        ↑              ↓
                   Fast local     Slow network
```

### Response Matrix (Phase 1 Implementation)

| Function | APFS Check Timing | Non-APFS Response | Response Type | JSON Example |
|----------|------------------|-------------------|---------------|--------------|
| `serve` | Never | Normal operation | Success | `{"status": "success", "command": "serve", ...}` |
| `list` | Never | Normal operation | Success | `{"status": "success", "command": "list", ...}` |
| `show` | Never | Normal operation | Success | `{"status": "success", "command": "show", ...}` |
| `health` | Never | Normal operation | Success | `{"status": "success", "command": "health", ...}` |
| `pull` | Never | Normal operation | Success | `{"status": "success", "command": "pull", ...}` |
| **`push`** | On success (Alpha only) | **Add APFS hint to message** | ⚠️ Success + Warning | `{"status": "success", "data": {"message": "Push successful. Clone operations require APFS filesystem."}}` |
| **`clone`** | On demand (lazy) | **Hard error, abort** | ❌ Error | `{"status": "error", "data": {"clone_status": "filesystem_error"}}` |

### Error Handling Matrix

#### Phase 1 Error Matrix (Same-Volume APFS Only)

| Scenario | Cache FS | Workspace FS | Same Volume | Error Type | Behavior | JSON Error | User Action |
|----------|----------|--------------|-------------|------------|----------|------------|-------------|
| ✅ **Supported** | APFS | APFS | Yes | None | Success | N/A | None |
| ❌ **Cache Requirement** | HFS+ | Any | Any | CacheFilesystemError | Hard Error | `{"status": "error", "data": {"clone_status": "filesystem_error"}}` | Migrate cache to APFS |
| ❌ **Cache Requirement** | ExFAT | Any | Any | CacheFilesystemError | Hard Error | `{"status": "error", "data": {"clone_status": "filesystem_error"}}` | Migrate cache to APFS |
| ❌ **Cache Requirement** | NFS/SMB | Any | Any | CacheFilesystemError | Hard Error | `{"status": "error", "data": {"clone_status": "filesystem_error"}}` | Use local APFS cache |
| ❌ **Workspace Requirement** | APFS | HFS+ | No | WorkspaceFilesystemError | Hard Error | `{"status": "error", "data": {"clone_status": "filesystem_error"}}` | Use APFS workspace |
| ❌ **Workspace Requirement** | APFS | ExFAT | No | WorkspaceFilesystemError | Hard Error | `{"status": "error", "data": {"clone_status": "filesystem_error"}}` | Use APFS workspace |
| ❌ **Workspace Requirement** | APFS | NFS/SMB | No | WorkspaceFilesystemError | Hard Error | `{"status": "error", "data": {"clone_status": "filesystem_error"}}` | Use APFS workspace |
| ❌ **Volume Requirement** | APFS | APFS | No | VolumeError | Hard Error | `{"status": "error", "data": {"clone_status": "filesystem_error"}}` | Move workspace to cache volume |

#### Phase 2 Error Matrix (Cross-Filesystem Support)

| Scenario | Cache FS | Workspace FS | Same Volume | Error Type | Behavior | JSON Response | User Action |
|----------|----------|--------------|-------------|------------|----------|---------------|-------------|
| ✅ **Optimal** | APFS | APFS | Yes | None | Success (CoW) | `{"status": "success", "data": {"clone_status": "success", "copy_method": "apfs_cow"}}` | None |
| ✅ **Standard** | APFS | APFS | No | None | Success (Standard) | `{"status": "success", "data": {"clone_status": "success", "copy_method": "standard_copy"}}` | None |
| ✅ **Standard** | APFS | HFS+ | No | None | Success (Standard) | `{"status": "success", "data": {"clone_status": "success", "copy_method": "standard_copy"}}` | None |
| ✅ **Standard** | APFS | ExFAT | No | None | Success (Standard) | `{"status": "success", "data": {"clone_status": "success", "copy_method": "standard_copy"}}` | None |
| ⚠️ **Network Warning** | APFS | NFS | No | NetworkWarning | Warning + Proceed | `{"status": "success", "data": {"clone_status": "success", "copy_method": "network_copy", "warning": "Network filesystem detected. Copy will be slower."}}` | Expect slower performance |
| ⚠️ **Network Warning** | APFS | SMB/CIFS | No | NetworkWarning | Warning + Proceed | `{"status": "success", "data": {"clone_status": "success", "copy_method": "network_copy", "warning": "Network filesystem detected. Copy will be slower."}}` | Expect slower performance |
| ❌ **Cache Requirement** | HFS+ | Any | Any | CacheFilesystemError | Hard Error | `{"status": "error", "data": {"clone_status": "filesystem_error"}}` | Migrate cache to APFS |
| ❌ **Cache Requirement** | ExFAT | Any | Any | CacheFilesystemError | Hard Error | `{"status": "error", "data": {"clone_status": "filesystem_error"}}` | Migrate cache to APFS |
| ❌ **Cache Requirement** | NFS/SMB | Any | Any | CacheFilesystemError | Hard Error | `{"status": "error", "data": {"clone_status": "filesystem_error"}}` | Use local APFS cache |

#### Error Message Examples

**⚠️ JSON Protocol Disclaimer:**
> All JSON response examples are provisional and based on specification v0.1.4. Field contents (e.g., `clone_status` values) and response structure may evolve during Phase 1 and Phase 2 implementation.

**Phase 1 Errors:**
```json
{
  "status": "error",
  "command": "clone",
  "data": {
    "clone_status": "filesystem_error",
    "target_dir": "/some/workspace"
  },
  "error": {
    "type": "FilesystemError",
    "message": "APFS cache required for clone operations."
  }
}
```

**Phase 2 Network Warnings:**
```json
{
  "status": "success",
  "command": "clone",
  "data": {
    "model": "microsoft/DialoGPT-small",
    "target_dir": "/Volumes/NASShare/models/dialog",
    "clone_status": "success",
    "message": "Cloned to /Volumes/NASShare/models/dialog",
    "expanded_name": "microsoft/DialoGPT-small"
  },
  "error": null
}
```

### Performance Characteristics

#### Phase 1 (Same APFS Volume)
```
4GB Model Clone Performance:
- Temp cache creation: ~30 seconds (network download)
- APFS CoW copy: ~0.1 seconds (metadata only)
- Temp cleanup: ~0.5 seconds
- Total time: ~30.6 seconds
- Total space: ~4GB (only in workspace after CoW)
```

#### Phase 2 (Cross-Filesystem)
```
4GB Model Clone Performance:
- Temp cache creation: ~30 seconds (network download)
- Standard copy: ~60 seconds (4GB copy)
- Temp cleanup: ~0.5 seconds
- Total time: ~90.5 seconds
- Peak space: ~8GB (temp + workspace during copy)
```

## Migration Strategy Between Phases

### Phase 1 → Phase 2 Upgrade
- **Breaking Change:** None (Phase 1 scenarios still work optimally)
- **New Capability:** Cross-filesystem support added
- **User Impact:** More flexible workspace placement
- **Performance:** Same for existing use cases, degraded for new cross-FS cases

### Implementation Flags
```python
# Alpha feature gate (existing)
MLXK2_ENABLE_ALPHA_FEATURES=1  # Required for clone and push operations

# Future Phase 2 flags (if needed)
# MLXK2_CLONE_ALLOW_CROSS_FILESYSTEM=1
```

## Implementation Details

### 1. Volume-Aware Temp Cache Creation

```python
def create_temp_cache_same_volume(target_workspace: Path) -> Path:
    """Create temp cache on same APFS volume as target for CoW optimization."""

    # Get target volume mount point via st_dev
    target_volume = get_volume_mount_point(target_workspace)

    # Create temp cache on same volume
    temp_cache = target_volume / f".mlxk2_temp_{os.getpid()}_{random.randint(1000,9999)}"
    temp_cache.mkdir(parents=True)

    # SAFETY: Create sentinel file to prevent accidental user cache deletion
    sentinel = temp_cache / ".mlxk2_temp_cache_sentinel"
    sentinel.write_text(f"mlxk2_temp_cache_created_{int(time.time())}")

    return temp_cache

def cleanup_temp_cache_safe(temp_cache: Path) -> bool:
    """Safely delete temp cache only if sentinel exists."""

    # SAFETY: Only delete if sentinel exists
    sentinel = temp_cache / ".mlxk2_temp_cache_sentinel"
    if not sentinel.exists():
        logger.warning(f"Refusing to delete {temp_cache} - no sentinel found")
        return False

    shutil.rmtree(temp_cache, ignore_errors=True)
    return True

def get_volume_mount_point(path: Path) -> Path:
    """Find mount point (volume root) for given path via st_dev changes."""
    abs_path = path.resolve()
    current = abs_path

    while current != current.parent:
        try:
            parent_stat = current.parent.stat()
            current_stat = current.stat()

            # Different st_dev = mount boundary
            if parent_stat.st_dev != current_stat.st_dev:
                return current
        except (OSError, PermissionError):
            pass
        current = current.parent

    return current  # Filesystem root
```

### 2. Shared APFS Filesystem Check

```python
def is_apfs_filesystem(path: Path) -> bool:
    """Simple APFS check - returns True/False only.

    Used by both clone (validation) and push (conditional warning).
    """
    try:
        import subprocess
        result = subprocess.run(['stat', '-f', '-c', '%T', str(path)],
                              capture_output=True, text=True)
        return result.stdout.strip() == 'apfs'
    except subprocess.CalledProcessError:
        return False  # Safe fallback

def validate_apfs_filesystem(path: Path) -> None:
    """Validate APFS requirement for clone operations.

    Called lazily - only on first clone operation, not at CLI startup.
    """
    if not is_apfs_filesystem(path):
        raise FilesystemError(
            f"APFS required for clone operations. "
            f"Path: {path}\n"
            f"Solution: Use APFS volume or external APFS SSD."
        )
```

### 3. Clone Operation Implementation

```python
def clone_operation(model_spec: str, target_dir: str) -> Dict[str, Any]:
    """Clone with isolated temp cache strategy."""

    target_path = Path(target_dir).resolve()

    # 1. Validate APFS requirement
    validate_apfs_filesystem(target_path.parent)

    # 2. Create temp cache on same volume as target
    temp_cache = create_temp_cache_same_volume(target_path)

    try:
        # 3. Pull to isolated temp cache
        with patch_hf_home(temp_cache):
            pull_result = pull_operation(model_spec)

        if pull_result["status"] != "success":
            return handle_pull_error(pull_result)

        # 4. Resolve temp cache snapshot path
        resolved_model = pull_result["data"]["model"]
        temp_snapshot = resolve_latest_snapshot(temp_cache, resolved_model)

        # 5. APFS clone to workspace (instant, CoW)
        target_path.mkdir(parents=True, exist_ok=True)
        clone_success = apfs_clone_directory(temp_snapshot, target_path)

        if not clone_success:
            return handle_clone_error()

        # 6. Success - temp cache auto-cleanup via context manager
        return {
            "status": "success",
            "command": "clone",
            "data": {
                "model": resolved_model,
                "target_dir": str(target_path),
                "clone_status": "completed",
                "cache_preserved": True,  # User cache never touched
                "copy_method": "apfs_cow"
            }
        }

    finally:
        # Cleanup temp cache
        shutil.rmtree(temp_cache, ignore_errors=True)
```

### 4. User Experience: Push Workflow Warning

```python
def push_operation(...) -> Dict[str, Any]:
    # ... normal push logic ...

    # Conditional APFS hint based on cache filesystem
    if not is_apfs_filesystem(get_hf_cache_dir()):
        message = "Push successful. Clone operations require APFS filesystem."
    else:
        message = "Push successful."

    result = {
        "status": "success",
        "command": "push",
        "data": {
            "repo_id": repo_id,
            "branch": branch,
            "message": message,
            # ... existing fields ...
        }
    }

    return result
```

### 5. APFS Copy-on-Write Implementation

```python
def apfs_clone_directory(source: Path, target: Path) -> bool:
    """Clone directory using APFS copy-on-write via clonefile."""
    try:
        for item in source.rglob("*"):
            if item.is_file():
                relative_path = item.relative_to(source)
                target_file = target / relative_path
                target_file.parent.mkdir(parents=True, exist_ok=True)

                # Use cp -c for clonefile (APFS CoW)
                subprocess.run(['cp', '-c', str(item), str(target_file)],
                             check=True, capture_output=True)
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"APFS clone failed: {e}")
        return False
```

## Pros and Cons

### Pros

1. **User Cache Preservation:** Never touches existing user cache
2. **Consistency:** Always gets latest/specified model version
3. **Performance:** APFS CoW provides instant copy with minimal space
4. **Isolation:** Temp cache prevents pollution of user environment
5. **Predictable:** Clone behaves like standard file copy operation
6. **Robust:** Clear filesystem requirements with early validation

### Cons

1. **APFS Requirement:** Users on non-APFS setups need migration
2. **Temporary Disk Usage:** Brief full model copy in temp cache before CoW
3. **Implementation Complexity:** Volume detection and temp cache management
4. **Platform Specific:** Relies on macOS/iOS APFS features

## Migration from ADR-006

### Breaking Changes

1. **Cache Behavior:** User cache is preserved (not deleted after clone)
2. **Filesystem Requirements:** APFS validation added
3. **Performance Profile:** May use more temporary disk space

### User Migration

**Before (ADR-006):**
```bash
mlxk2 clone org/model ./workspace  # Deleted model from cache
```

**After (ADR-007):**
```bash
mlxk2 clone org/model ./workspace  # Preserves model in cache
# User cache remains intact for other operations
```

### Error Handling

**Non-APFS Cache:**
```
Error: Filesystem 'nfs' not supported
MLX-Knife requires APFS for clone operations.

Current path: /Volumes/NetworkShare/cache
Solution: Use APFS volume:
  export HF_HOME="/Users/you/.cache/huggingface"
```

## Testing Strategy

### Unit Tests

1. **Volume Detection:** Verify mount point resolution across scenarios
2. **APFS Validation:** Test filesystem detection and error handling
3. **Temp Cache Creation:** Validate same-volume placement
4. **Copy-on-Write:** Test clonefile success and fallback behavior

### Integration Tests

1. **Cross-Volume Scenarios:** Cache on external APFS, workspace on internal
2. **Large Model Performance:** Verify CoW benefits with multi-GB models
3. **Error Recovery:** Temp cache cleanup on failures
4. **Concurrent Access:** Multiple clone operations

### Real-World Validation

1. **External APFS SSDs:** Thunderbolt/USB-C attached storage
2. **iOS Simulator:** Validate iOS filesystem assumptions
3. **Network Limitations:** Ensure clear errors for unsupported setups

## Implementation Timeline

### Phase 1: Same-Volume APFS (2.0.0-beta.3)
**Target:** Stable clone functionality with strict constraints
- ✅ Volume detection utilities (`get_volume_mount_point`, `is_same_volume`)
- ✅ APFS validation framework (`validate_apfs_filesystem`)
- ✅ Temp cache management on same volume
- ✅ APFS copy-on-write implementation (`apfs_clone_directory`)
- ✅ Error handling for unsupported scenarios
- ✅ Performance optimization for direct CoW path

**Success Criteria:**
- Clone works reliably when cache and workspace on same APFS volume
- Clear error messages for unsupported filesystem combinations
- Performance benchmarks show near-instant copy for large models

### Phase 2: Cross-Filesystem Support (eventually, when clone and push is non-Alpha)
**Target:** Flexible workspace placement with graceful degradation
- 🔄 Cross-filesystem copy implementation
- 🔄 Performance monitoring for different copy methods
- 🔄 Network filesystem handling and warnings
- 🔄 User experience improvements for mixed scenarios
- 🔄 Configuration flags for behavior control

**Success Criteria:**
- Clone works across all supported filesystem combinations
- Performance degradation is predictable and documented
- User guidance for optimal setup configurations

### Phase 3: Advanced Features (future, no version commitment)
**Target:** Production hardening and edge case handling
**Status:** Nice-to-have features, implement based on user demand

- 🔄 Incremental clone support (delta updates)
- 🔄 Resume capability for interrupted operations
- 🔄 Bandwidth limiting for network operations
- 🔄 Comprehensive logging and diagnostics
- 🔄 Advanced caching strategies

## Decision Rationale

This strategy addresses the fundamental flaws in ADR-006 while leveraging the strengths of the Apple Silicon ecosystem. By requiring APFS and using isolated temp caches, we achieve:

- **Correctness:** No data loss or inconsistent states
- **Performance:** Copy-on-write optimization
- **Simplicity:** Clear requirements and predictable behavior

The APFS requirement is justified given MLX's Apple Silicon dependency and the target use case focus on iOS development.

## Status

- **Implementation:** To be started
- **Testing:** Required before release
- **Documentation:** Needs update for filesystem requirements
- **Release:** Blocks 2.0.0-beta.3 until complete