# ADR-005: mlxk2 clone Implementation for 2.0.0-beta.3

## Status
**DEPRECATED** - 2025-09-15

**Superseded by:** ADR-006 (Clone Implementation - Revised Strategy)

**Reason for Deprecation:** Critical findings revealed that HuggingFace Hub's `local_dir` parameter does not provide reliable cache isolation and can corrupt existing cache entries. The assumptions about cache isolation in this ADR were incorrect.

## Context

GitHub Issue #29 requests Ollama CLI-like "push" functionality for MLX Knife. The push function was successfully implemented in 2.0.0-alpha.1, but analysis revealed a critical workflow gap: there is no `clone` command to create writable workspaces from HuggingFace models.

### Current Workflow Limitations

**Missing Link in Author Workflow:**
```bash
# Desired workflow - currently incomplete
mlxk2 clone org/model@revision ./workspace    # ❌ Missing
mlxk2 health ./workspace                      # ✅ Exists
mlxk2 push ./workspace org/my-model --private # ✅ Exists
```

**Two Key Use Cases Identified:**
1. **Fork-Modify-Push:** `clone` existing HF model → edit → `push` to new repo
2. **Author-Generated Models:** Native MLX training → workspace → `health` → `push`

### Technical Analysis Results

**MLX Model Compatibility:** ✅ No additional work needed
- Native MLX models use identical structure to HuggingFace models (config.json + .safetensors)
- Existing `_analyze_workspace()` in push.py already validates MLX-native models correctly
- No .npz/.mlx extensions - MLX uses .safetensors with metadata={"format": "mlx"}

**Implementation Effort:** Very Low (~2 hours)
- Can reuse 90% of existing `pull.py` logic (snapshot_download)
- Only difference: download to custom local_dir instead of HF cache
- Test patterns already established for push (21 tests with offline/online/spec coverage)

### JSON API Schema Impact

**Required Changes for JSON API 0.1.4:**
- Schema update: Add "clone" to command enum in `docs/json-api-schema.json:9`
- Version bump: `mlxk2/spec.py` → `JSON_API_SPEC_VERSION = "0.1.4"`
- Documentation update: `docs/json-api-specification.md` → Version 0.1.4
- **No new schema definition needed** - clone reuses existing pull schema

## Decision

We will implement `mlxk2 clone` for 2.0.0-beta.3 to complete the GitHub Issue #29 feature request and provide a comprehensive workspace-based workflow, including full JSON API 0.1.4 compliance.

## Implementation Plan

### Phase 1: Core Implementation + JSON API (Session 1)
- **Time Estimate:** 1-1.5 hours (simplified - no new schema needed)
- **Files to Modify:**
  - `mlxk2/operations/clone.py` - New file, ~80 lines (reuse pull.py patterns)
  - `mlxk2/cli.py` - Add clone command integration
  - `mlxk2/spec.py` - Version bump to 0.1.4
  - `docs/json-api-schema.json` - Add "clone" to command enum only
  - `docs/json-api-specification.md` - Version update + clone documentation
  - Basic test coverage: CLI args, validation, JSON output schema

### Phase 2: Complete Test Suite (Session 2)
- **Time Estimate:** 1-2 hours
- **Test Structure:** Mirror existing push test patterns from TESTING.md
  - Offline tests: target directory validation, CLI argument parsing
  - Online tests: live clone with opt-in env vars (MLXK2_LIVE_CLONE=1)
  - Spec tests: JSON schema validation for clone command output (JSON API 0.1.4)
  - Integration: Add to existing test matrix in TESTING.md

### Phase 3: Issue #29 Feedback
- Request user testing from feynon (Swift/iOS porting use case)
- Validate workflow completeness for both identified use cases

## Implementation Details

### API Signature
```bash
mlxk2 clone <org>/<repo>[@<revision>] <target_dir> [options]
```

**Options:**
- `--branch <branch>` - Clone specific branch/revision
- `--json` - JSON output mode
- `--quiet` - Suppress progress output

### Critical Cache Behavior Requirements

**IMPORTANT:** Session 1 initial implementation used `snapshot_download(local_dir=target)` which creates symlinks to HF cache. This violates the core requirements below.

**Required Implementation:**
```python
snapshot_download(
    repo_id=model_name,
    local_dir=str(target_path),
    local_dir_use_symlinks=False  # CRITICAL: Force actual file copies
)
```

**Cache Isolation Validation:**
- Clone target must contain real files, not symlinks
- HF cache (`~/.cache/huggingface/hub/`) must not be populated during clone
- Target directory should be completely self-contained workspace

### JSON Response Schema (API 0.1.4)
```json
{
  "status": "success|error",
  "command": "clone",
  "data": {
    "model": "org/repo",
    "download_status": "completed",
    "message": "Cloned successfully to ./workspace",
    "target_dir": "/abs/path/to/workspace"
  },
  "error": null
}
```

**Note:** Clone reuses the existing `pull` schema. The `additionalProperties: true` allows `target_dir` field. Only schema change: command enum addition.

### Code Reuse Strategy
- Leverage `pull.py:snapshot_download()` logic
- Reuse `push.py:_analyze_workspace()` for post-clone health validation
- Maintain consistent error handling patterns with existing operations

## JSON API Schema Updates

### Required Schema Changes (docs/json-api-schema.json)

**1. Command Enum Update (Line 9):**
```json
"command": {"type": "string", "enum": ["list", "show", "health", "pull", "rm", "version", "push", "run", "clone"]}
```

**2. No new schema definition needed:**
- Clone reuses existing `pull` schema (lines 180-202)
- `"additionalProperties": true` allows `target_dir` field
- `"required": ["download_status", "message"]` covers clone requirements
- Schema validation works automatically for clone commands

### Specification Documentation Update

**Version:** 0.1.4 (minimal bump for command enum change)
**New Section:** Clone Command documentation with examples

## Testing Strategy

**Test Categories (following TESTING.md patterns):**
- **Offline Tests:** ~10 tests (CLI validation, error handling, schema compliance)
- **Online Tests:** ~3 opt-in tests with live HF repos (MLXK2_LIVE_CLONE=1)
- **Spec Tests:** ~3 JSON schema validation tests (JSON API 0.1.4)

**Environment Variables:**
- `MLXK2_ENABLE_EXPERIMENTAL_CLONE=1` - Enable clone tests in CI
- `MLXK2_LIVE_CLONE=1` - Enable live network tests (opt-in)

**Schema Validation Testing:**
- All clone responses validate against updated JSON schema 0.1.4
- Test both success and error response structures
- Verify backward compatibility with existing commands

## Benefits

1. **Completes Issue #29:** Provides full workspace-based model management workflow
2. **Swift/iOS Friendly:** Clean JSON API suitable for cross-platform porting
3. **Low Risk:** Reuses battle-tested components (snapshot_download, workspace analysis)
4. **Fast Implementation:** Can be completed in 1-2 Claude sessions
5. **Test Coverage:** Follows established patterns from push implementation
6. **JSON API Compliance:** Full schema validation and version management

## Security Classification and Risk Analysis

### Clone vs Push: Fundamental Safety Difference

**Clone Operation: LOW RISK**
- **Read-only operation:** Downloads existing HF content to local workspace
- **No publication risk:** Cannot create/modify remote repositories
- **Local-only impact:** Only affects specified target directory
- **Cache isolation:** Bypasses HF cache entirely - direct download to target
- **Validation safeguards:** Refuses to overwrite non-empty directories
- **Risk profile:** Similar to `pull` operation - safe for general use

**Push Operation: HIGH RISK**
- **Write operation:** Publishes content to HuggingFace Hub
- **Publication risk:** Can accidentally expose private/sensitive data
- **Global impact:** Creates permanent public records
- **Requires authentication:** Uses HF_TOKEN with write permissions
- **Experimental status:** Hidden behind `MLXK2_ENABLE_EXPERIMENTAL_PUSH=1`

### Implementation Implications

**Clone does NOT require experimental gating:**
- No `MLXK2_ENABLE_EXPERIMENTAL_CLONE=1` flag needed
- Can be enabled by default in 2.0.0-beta.3
- Standard test integration (not opt-in only)
- Live tests follow normal marker patterns (like `list`, `pull`)

**Clone workspace isolation guarantees:**
1. **No cache pollution:** Downloads directly to target_dir with `local_dir_use_symlinks=False`, never touches HF_HOME
2. **No overwrite risk:** Validation ensures target directory is empty or non-existent
3. **Explicit targeting:** User must specify exact target path
4. **Atomic operation:** Either succeeds completely or fails cleanly
5. **Real file copies:** Target contains actual files, not symlinks to cache (validated in tests)

## Risks and Mitigations

**Risk:** Directory conflicts and overwrite behavior
**Mitigation:** Require explicit target directory, validate empty/non-existent before download

**Risk:** Large model download interruption
**Mitigation:** Leverage huggingface_hub's built-in resume_download=True

**Risk:** Disk space exhaustion
**Mitigation:** Pre-flight disk space check, clear error messages

**Risk:** JSON API version compatibility
**Mitigation:**
- Follow established versioning patterns from existing commands
- Complete schema validation test coverage
- Document breaking changes clearly

**Risk:** Test suite complexity
**Mitigation:** Standard test integration (not experimental opt-in), proven patterns from pull tests

## Timeline

**Target:** 2.0.0-beta.3 release within 24 hours
- Session 1: Core implementation + minimal schema update + basic tests (1.5-2 hours)
- Session 2: Complete test suite + documentation (1-2 hours)
- Issue #29 feedback request: Immediate after implementation

## Success Criteria

1. ✅ Complete workflow: `clone` → `health` → `push`
2. ✅ Both use cases supported (fork-modify-push + author-generated)
3. ✅ JSON API 0.1.4 compliance with full schema validation
4. ✅ Test coverage matches push patterns (~15 total tests)
5. ✅ Schema backwards compatibility maintained
6. ✅ feynon feedback positive for Swift porting use case

## References

- GitHub Issue #29: https://github.com/ml-explore/mlx-knife/issues/29
- TESTING.md: Push test patterns (21 tests, offline/online/spec structure)
- ADR-001: JSON-first architecture principles
- mlxk2/operations/push.py: Workspace analysis and health check patterns
- docs/json-api-schema.json: Current schema definition (0.1.3)
- docs/json-api-specification.md: Current specification (0.1.3)