# Commit Message Draft for MLX Knife 1.1.0-beta3

## Primary Commit Message

```
Release MLX Knife 1.1.0-beta3 - Critical Bug Fixes & Lock Cleanup Resolution

Major bug fixes addressing cache management and user experience issues:

**Issue #21: Empty Cache Directory Crash - RESOLVED**
- Fix: Added MODEL_CACHE.exists() checks in list_models() function  
- Impact: MLX-Knife now works correctly on fresh installations
- Files: cache_utils.py:459-462, cache_utils.py:478-481
- Test: Added test_list_models_real_empty_cache() regression test

**Issue #22: urllib3 LibreSSL Warning on macOS Python 3.9 - RESOLVED**
- Fix: Central warnings suppression before urllib3 imports
- Impact: Clean output on macOS system Python 3.9 with LibreSSL
- Files: __init__.py:7-9
- Scope: Only affects macOS system Python 3.9

**Issue #23: Double rm Execution Problem - FULLY RESOLVED**
- Problem: `mlxk rm model@hash` required two executions (first left broken state)
- Root Cause: Only deleted snapshots/<hash>, left refs/main pointing to deleted snapshot  
- Fix: Changed to delete entire model directory, not just specific snapshot
- Additional Fix: Corrected lock cleanup path bug discovered during implementation
- Impact: Single execution now completely removes models + cleans orphaned locks
- Files: cache_utils.py (whole model deletion + lock cleanup path correction)
- Tests: Added comprehensive integration tests covering full rm lifecycle

Technical improvements:
- Enhanced test coverage: 140/140 tests passing (up from 137)
- Fixed 3 unit tests broken by lock cleanup path correction
- Improved cache path consistency across all Python versions
- Better error handling for fresh installations and corrupted models

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

## Alternative Shorter Version

```
Release MLX Knife 1.1.0-beta3 - Critical Cache Management Fixes

Three major bug fixes for production readiness:

- Issue #21: Fix crash on fresh installations (empty cache directory)
- Issue #22: Suppress urllib3 LibreSSL warnings on macOS Python 3.9  
- Issue #23: Fix double rm execution bug - models now deleted in single command

Test improvements:
- 140/140 tests passing (up from 137)
- Added real integration tests for lock cleanup
- Fixed unit tests broken by path corrections

All known cache management issues resolved for stable release.

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

## Files Modified Summary

- `mlx_knife/__init__.py` - Issue #22: urllib3 warnings suppression
- `mlx_knife/cache_utils.py` - Issues #21, #23: empty cache fix + lock cleanup path
- `tests/integration/test_lock_cleanup_bug.py` - NEW: Issue #23 regression tests
- `tests/unit/test_cache_utils.py` - Updated mocks for corrected lock paths
- `CLAUDE.md` - Documentation updates for all three issues
- `TESTING.md` - Test structure and count updates