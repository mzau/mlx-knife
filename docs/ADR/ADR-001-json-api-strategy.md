# ADR-001: MLX-Knife 2.0 Migration Path to JSON-First Architecture

## Status
**Accepted & Implemented** - 2025-08-28

**Implementation Status:**
- ✅ Clean-room 2.0 implementation complete (Sessions 1-3)
- ✅ JSON-first architecture validated
- ✅ Parallel deployment strategy documented
- ✅ Broke-cluster integration ready

## Context

MLX-Knife 1.1.0 has achieved stability with 150/150 tests passing, but faces architectural challenges:
- `cache_utils.py` contains 1000+ lines causing ~4000 tokens per Claude interaction
- Dual output format (human + JSON) would add complexity 
- Refactoring existing code risks breaking stable functionality
- broke-cluster project needs scriptable JSON API for automated model management

## Decision

We will create MLX-Knife 2.0 as a **clean-room implementation** with JSON-first architecture, maintaining the robust maintenance functions while simplifying the codebase.

## Migration Path

### Phase 1: Alpha Foundation 
**Version: 2.0.0-alpha**
- Feature-complete JSON-only implementation
- All 5 commands: list, show, pull, rm, health
- 100% test coverage (45/45 passing)
- Clean modular architecture
- No server/run functionality (JSON-only scope)

### Phase 2: Beta Validation (6-8 weeks)
**Version: 2.0.0-beta**
- All alpha features with production-grade testing
- Performance benchmarks with large caches
- Robust broke-cluster integration validation
- Still JSON-only (no server/run)

### Phase 3: Feature Parity (Release Candidate)
**Version: 2.0.0-rc**  
- Add server functionality from 1.x
- Add run/chat functionality
- Full feature parity with MLX-Knife 1.x
- Human-readable output via CLI layer 
- All features JSON-first design
- No dual output logic

### Phase 4: Test Suite Migration (Week 5)
**Version: 2.0.0-beta2**
- New test suite for JSON output
- Compatibility tests against 1.1.0
- Edge case coverage (from ADR-002)
- Target: 50-70 focused tests vs 150 in 1.x

### Phase 5: Production Ready (Month 2)
**Version: 2.0.0-rc1 → 2.0.0**
- Documentation complete
- Migration guide from 1.x
- broke-cluster validated in production
- Community feedback incorporated

## Architecture Principles

### 1. Module Structure
```
mlx-knife-2/
├── mlxk2/
│   ├── core/
│   │   ├── cache.py       # Cache path management
│   │   └── model_resolution.py  # Model discovery & resolution
│   ├── operations/
│   │   ├── list.py        # List operation
│   │   ├── health.py      # Health validation
│   │   ├── show.py        # Show details (50 lines)
│   │   ├── pull.py        # Download models (100 lines)
│   │   └── remove.py      # Delete models (50 lines)
│   ├── output/
│   │   └── json.py        # JSON serialization (50 lines)
│   └── cli.py             # CLI entry point (100 lines)
```

### 2. Dependency Rules
- No circular dependencies
- Core modules are dependency-free
- Operations depend on core only
- CLI depends on operations and output
- Maximum dependency depth: 3 levels

### 3. Code Limits
- No file exceeds 200 lines
- No function exceeds 50 lines
- No class exceeds 100 lines
- Clear separation of concerns

## Implementation Guidelines

### JSON Output Schema
All commands return consistent JSON structure:
```json
{
  "status": "success|error",
  "command": "list|show|pull|rm|health",
  "data": { /* command specific */ },
  "error": null | { "type": "...", "message": "..." }
}
```

### Error Handling
- All errors return valid JSON
- Exit codes remain compatible with 1.x
- Detailed error messages for debugging

### Backward Compatibility
- Same cache directory structure
- Same model naming conventions
- Can run parallel to 1.1.0
- No shared state between versions

## Testing Strategy

### Alpha Testing (alpha0-alpha1)
- Manual testing against known models
- Comparison with 1.1.0 output
- broke-cluster integration testing

### Beta Testing (beta1-beta2)
- Automated test suite
- Edge case coverage from ADR-002
- Performance benchmarks

### Release Testing (rc1)
- Full compatibility validation
- Community beta testing
- Production deployment in broke-cluster

## Success Metrics

1. **Code Reduction**: <1000 lines total (vs 3000+ in 1.x)
2. **Token Efficiency**: <500 tokens per file for Claude
3. **Test Coverage**: >90% for critical paths
4. **Performance**: Same or better than 1.1.0
5. **broke-cluster**: Successful production deployment

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Missing edge cases | High | Extract from 1.x tests (ADR-002) |
| User migration resistance | Medium | Maintain 1.x support, clear benefits |
| Feature gaps | Low | Incremental feature addition |
| Performance regression | Medium | Benchmark against 1.1.0 |

## Consequences

### Positive
- Clean, maintainable codebase
- 80% reduction in Claude token usage
- Perfect for automation/scripting
- Faster development cycles
- Clear architecture

### Negative
- Breaking change for users
- Temporary feature gaps
- Parallel maintenance (short-term)
- Learning curve for JSON output

## Decision Outcome

Proceed with clean-room 2.0.0 implementation following the phased approach, starting with alpha0 for immediate broke-cluster value.

## References
- Issue #8: Model caching
- Issue #26: Embeddings API  
- JSON Feature Request document
- mlx-knife-refactoring-plan.md (rejected approach)