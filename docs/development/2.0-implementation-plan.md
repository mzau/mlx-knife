# MLX-Knife 2.0 Implementation Plan

## Executive Summary

Clean-room implementation of MLX-Knife with JSON-first architecture for broke-cluster automation.
Start: Immediately | Target: 2.0.0-alpha0 in 4 hours, stable in 6-8 weeks

## Session-by-Session Breakdown

### Session 1: Bootstrap (4 hours)
**Goal**: Working 2.0.0-alpha0 for broke-cluster

**Setup (30 min):**
```bash
# Create new repo structure
cd /Volumes/mz-SSD/gitprojekte/
mkdir mlx-knife-2
cd mlx-knife-2
git init
git remote add origin <repo-url>

# Initial structure
mkdir -p mlxk2/{core,operations,output}
touch mlxk2/__init__.py
touch mlxk2/cli.py
touch pyproject.toml
touch README.md
```

**Core Implementation (3 hours):**
```python
# mlxk2/core/cache.py (50 lines)
- HF_CACHE_ROOT constant
- hf_to_cache_dir()
- cache_dir_to_hf()
- get_model_path()

# mlxk2/core/discovery.py (100 lines)
- find_all_models()
- expand_model_name()
- resolve_model()

# mlxk2/operations/list.py (50 lines)
- list_models() -> dict

# mlxk2/operations/show.py (50 lines)
- show_model(name) -> dict

# mlxk2/output/json.py (30 lines)
- format_output(data, error=None)
- format_error(type, message)

# mlxk2/cli.py (100 lines)
- main() entry point
- Command routing
- JSON output only
```

**Testing (30 min):**
```bash
# Manual test
python -m mlxk2.cli list
python -m mlxk2.cli show Phi-3

# Verify JSON output
python -m mlxk2.cli list | jq .
```

**Deliverable**: Working `mlxk2 list` and `mlxk2 show` with JSON output

---

### Session 2: Robust Operations (4 hours)
**Goal**: Add health, pull, rm commands with edge case handling

**Health Checking (1.5 hours):**
```python
# mlxk2/core/health.py (150 lines)
- is_lfs_pointer()  # Critical!
- check_config_exists()
- check_tokenizer_exists()
- check_weights_valid()
- get_model_health(path) -> HealthStatus

# mlxk2/operations/health.py (50 lines)
- health_check(name=None) -> dict
```

**Pull Operation (1 hour):**
```python
# mlxk2/operations/pull.py (100 lines)
- validate_model_name()  # 96 char limit!
- pull_model(name) -> dict
- Use huggingface_hub.snapshot_download directly
```

**Remove Operation (1 hour):**
```python
# mlxk2/operations/remove.py (80 lines)
- remove_model(name, force=False) -> dict
- MUST handle force flag correctly (Issue #23)
- MUST clean .lock files
```

**Integration (30 min):**
- Wire up commands in cli.py
- Test each operation
- Verify edge cases from ADR-002

**Deliverable**: Complete CLI with all basic commands

---

### Session 3: Test Suite Foundation (3 hours)
**Goal**: Automated test coverage for alpha0 functionality

**Test Structure:**
```bash
tests/
├── conftest.py          # CRITICAL: isolated_cache fixture
├── test_core.py         # Pure functions
├── test_operations.py   # Command tests
└── test_edge_cases.py   # From ADR-002
```

**CRITICAL - conftest.py must include:**
```python
@pytest.fixture
def isolated_cache(monkeypatch):
    """Prevents ANY test from touching user's cache."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_cache = Path(tmpdir) / "huggingface/hub"
        test_cache.mkdir(parents=True)
        monkeypatch.setenv("HF_HOME", str(tmpdir / "huggingface"))
        # Patch all cache references
        yield test_cache
```

**Core Tests (1 hour):**
```python
# test_core.py
- test_hf_cache_round_trip()
- test_model_name_expansion()
- test_invalid_names()
- test_96_char_limit()
```

**Operation Tests (1 hour):**
```python
# test_operations.py
- test_list_empty_cache()
- test_list_with_models()
- test_show_existing()
- test_rm_force_flag()
```

**Edge Case Tests (1 hour):**
```python
# test_edge_cases.py
- test_lfs_pointer_detection()
- test_lock_file_cleanup()
- test_partial_model_handling()
```

**Deliverable**: 30+ passing tests, CI ready

---

### Session 4: Production Hardening (4 hours)
**Goal**: Make alpha1 production-ready for broke-cluster

**Error Handling (1 hour):**
- Consistent error JSON format
- Graceful degradation
- Timeout handling
- Network retry logic

**Performance (1 hour):**
- Optimize model discovery
- Parallel health checks
- Caching where appropriate

**Documentation (1 hour):**
```markdown
# README.md
- Installation
- JSON schema documentation
- Migration from 1.x
- broke-cluster examples
```

**Packaging (1 hour):**
```toml
# pyproject.toml
[project]
name = "mlx-knife2"
version = "2.0.0-alpha1"
```

**Deliverable**: PyPI-ready package

---

### Session 5: Server Mode Port (6 hours)
**Goal**: Add server functionality (beta1)

**Server Foundation (3 hours):**
```python
# mlxk2/server.py
- FastAPI app
- /v1/models endpoint
- /v1/chat/completions endpoint
- Token limit handling from ADR-002
```

**Model Loading (2 hours):**
```python
# mlxk2/runner.py
- Port minimal MLXRunner
- Memory management
- Context length extraction
```

**Testing (1 hour):**
- Server startup/shutdown
- Endpoint testing
- Token limit validation

**Deliverable**: Working server mode

---

### Session 6: Migration & Polish (4 hours)
**Goal**: Ready for release

**Compatibility Tests (2 hours):**
- Compare output with 1.x
- Verify cache compatibility
- Test migration scenarios

**Documentation (1 hour):**
- Complete API documentation
- Migration guide
- Changelog

**Release Prep (1 hour):**
- Version bump to 2.0.0-rc1
- GitHub release notes
- PyPI upload

**Deliverable**: Release candidate

---

## File Structure Summary

```
mlx-knife-2/
├── mlxk2/
│   ├── __init__.py
│   ├── cli.py               # Entry point (100 lines)
│   ├── core/
│   │   ├── __init__.py
│   │   ├── cache.py         # Cache paths (50 lines)
│   │   ├── discovery.py     # Model finding (100 lines)
│   │   └── health.py        # Health checks (150 lines)
│   ├── operations/
│   │   ├── __init__.py
│   │   ├── list.py          # List command (50 lines)
│   │   ├── show.py          # Show command (50 lines)
│   │   ├── pull.py          # Pull command (100 lines)
│   │   ├── remove.py        # Remove command (80 lines)
│   │   └── health.py        # Health command (50 lines)
│   ├── output/
│   │   ├── __init__.py
│   │   └── json.py          # JSON formatting (30 lines)
│   ├── server.py            # Server mode (300 lines)
│   └── runner.py            # Model runner (200 lines)
├── tests/
│   ├── conftest.py
│   ├── test_core.py
│   ├── test_operations.py
│   └── test_edge_cases.py
├── pyproject.toml
├── README.md
├── CHANGELOG.md
└── LICENSE
```

**Total Lines**: ~1200 (vs 3000+ in 1.x)

## Risk Mitigation Checklist

### Before Each Session:
- [ ] Review relevant ADR sections
- [ ] Check Issue tracker for new findings
- [ ] Backup current progress

### After Each Session:
- [ ] Run all tests
- [ ] Compare output with 1.x
- [ ] Update documentation
- [ ] Commit with clear message

### Critical Validations:
- [ ] Force flag works correctly (Issue #23)
- [ ] Lock files are cleaned up
- [ ] LFS pointers detected
- [ ] 96 char name limit enforced
- [ ] JSON always valid (even on error)
- [ ] Token limits respected

## Success Metrics

### Alpha0 (Session 1):
- [ ] List command works
- [ ] JSON output valid
- [ ] broke-cluster can parse output

### Alpha1 (Session 4):
- [ ] All basic commands work
- [ ] 30+ tests passing
- [ ] Edge cases handled

### Beta1 (Session 5):
- [ ] Server mode works
- [ ] Feature parity (except formatting)
- [ ] Performance acceptable

### RC1 (Session 6):
- [ ] Migration guide complete
- [ ] No known bugs
- [ ] Community tested

## Go/No-Go Criteria

### Proceed to Next Phase If:
✅ Current phase tests pass
✅ No blocking bugs
✅ Performance acceptable
✅ JSON schema stable

### Stop and Reassess If:
❌ Core assumption wrong
❌ Complexity exceeding estimates
❌ Breaking changes needed
❌ Performance regression

## Timeline Summary

**Week 1:**
- Day 1: Sessions 1-2 (alpha0)
- Day 2: Session 3 (tests)
- Day 3: Session 4 (alpha1)
- Day 4-5: broke-cluster testing

**Week 2:**
- Session 5 (server mode)
- Community feedback
- Bug fixes

**Week 3-4:**
- Session 6 (polish)
- Beta testing
- Documentation

**Week 5-6:**
- Release candidates
- Production validation
- 2.0.0 release

## Notes for Implementation

1. **Start Simple**: Get list/show working first
2. **JSON First**: No dual format complexity
3. **Test Early**: Write tests as you go
4. **Document Everything**: Capture decisions
5. **Compare Constantly**: Validate against 1.x

## Command Quick Reference

```bash
# Development
python -m mlxk2.cli list
python -m mlxk2.cli show Phi-3
python -m mlxk2.cli health
python -m mlxk2.cli pull mlx-community/model
python -m mlxk2.cli rm model -f

# Testing
pytest tests/ -xvs
pytest tests/test_edge_cases.py

# Comparison
mlxk list | head -20
python -m mlxk2.cli list | jq .

# broke-cluster usage
mlxk2 list | jq -r '.models[].name'
mlxk2 health | jq '.summary'
```

---

This plan provides a clear, session-by-session roadmap to implement MLX-Knife 2.0 with JSON-first architecture while maintaining the robustness of 1.x.