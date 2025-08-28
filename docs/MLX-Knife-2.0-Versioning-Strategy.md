# MLX-Knife 2.0 Versioning Strategy

**Document Status:** Approved Session 3 (2025-08-28)  
**Purpose:** Clear versioning scheme and deployment strategy for MLX-Knife 2.0

## Versioning Schema

### **2.0.0-alpha** (Feature-Complete for JSON-Only)
**Scope:** Core JSON operations without server/run functionality

**Features:**
- ✅ All 5 Operations: `list`, `health`, `show`, `pull`, `rm`
- ✅ JSON API fully implemented per specification
- ✅ Core functionality working (broke-cluster compatible)
- ❌ **Not robustly tested** - Mock fixtures have issues
- ❌ No `server` or `run` commands

**Quality Gate:**
- Core operations functional in isolation
- JSON schema stable and documented
- Basic edge case handling

**Target Users:**
- Broke-cluster integration (POC environment)
- Early adopters for JSON automation
- Parallel deployment alongside 1.x

### **2.0.0-beta** (Robustly Tested, JSON-Only)
**Scope:** All alpha features with production-grade testing

**Quality Improvements:**
- ✅ **100% test coverage** - All mock fixtures working correctly
- ✅ All edge cases from ADR-002 validated
- ✅ Integration tests with realistic scenarios
- ✅ Performance benchmarks established
- ✅ Error handling comprehensive

**Quality Gate:**
- Zero test failures on core operations
- All ADR-002 edge cases handled
- Performance acceptable for large caches
- Documentation complete

**Target Users:**
- Production JSON automation
- CI/CD pipeline integration
- Broke-cluster production deployment

### **2.0.0-rc** (Feature-Complete vs 1.x)
**Scope:** Full feature parity with MLX-Knife 1.x

**New Features:**
- ✅ `server` command - OpenAI-compatible API server
- ✅ `run` command - Interactive model execution
- ✅ `embed` command - Embedding generation (if merged from 1.x)
- ✅ Human-readable output via CLI layer formatting

**Quality Gate:**
- All 1.x functionality replicated
- Migration path documented
- Performance parity or better
- Server functionality validated

**Target Users:**
- Full 1.x replacement candidates
- Users requiring both JSON and human output
- Server-mode applications

### **2.0.0-stable**
**Scope:** Production-ready replacement for MLX-Knife 1.x

**Requirements:**
- ✅ All RC features stable and documented
- ✅ Migration guide with examples
- ✅ Community feedback incorporated
- ✅ Long-term support commitment
- ✅ Package management (pip/brew) ready

**Target Users:**
- All MLX-Knife users
- General availability deployment

## Deployment Strategy

### Broke-Cluster POC Environment

**Parallel Deployment Architecture:**
```bash
# System-wide: MLX-Knife 1.1.0 (stable server functionality)
pip install mlx-knife==1.1.0

# Local development: MLX-Knife 2.0.0-alpha (JSON management)
pip install -e /path/to/mlx-knife-2.0  # Local install
```

**Usage Pattern:**
```bash
# Server operations: Use 1.x (stable, proven)
mlxk server --model "Phi-3-mini" --port 8000

# Management operations: Use 2.0.0-alpha (JSON automation)
mlxk-json list --json | jq '.data.models[].name'
mlxk-json health --json | jq '.data.summary'
mlxk-json pull "new-model" --json
```

**Benefits:**
- ✅ **Risk mitigation**: Server stability maintained with 1.x
- ✅ **Feature validation**: JSON API tested in production environment  
- ✅ **Gradual migration**: Teams can adopt 2.0 features incrementally
- ✅ **Rollback safety**: Can disable 2.0 without affecting server operations

### Package Naming Strategy

**Development Phase:**
- `mlx-knife` (1.1.0) - Stable production version
- `mlxk2` / `mlxk-json` - Development 2.0.0-alpha local install

**Production Phase:**
- `mlx-knife` (2.0.0+) - New major version
- `mlx-knife-v1` (1.1.0) - Legacy support if needed

## Quality Gates Summary

| Version | Test Coverage | Features | Server Mode | Production Ready |
|---------|---------------|----------|-------------|------------------|
| **alpha** | ~70% (mock issues) | JSON-only (5 ops) | ❌ | Limited |
| **beta** | 100% | JSON-only (5 ops) | ❌ | Yes (JSON) |
| **rc** | 100% | Full parity | ✅ | Yes (All) |
| **stable** | 100% + community | Full parity | ✅ | Yes (LTS) |

## Success Metrics

### Alpha Success Criteria
- [ ] Broke-cluster integration working
- [ ] Core JSON operations stable
- [ ] No user cache corruption in testing
- [ ] JSON schema documentation complete

### Beta Success Criteria  
- [ ] 100% test pass rate
- [ ] Performance benchmarks established
- [ ] All ADR-002 edge cases handled
- [ ] Production deployment successful

### RC Success Criteria
- [ ] Feature parity with 1.x achieved
- [ ] Migration guide validated
- [ ] Server mode performance acceptable
- [ ] Community feedback positive

### Stable Success Criteria
- [ ] 6+ months beta stability
- [ ] Multiple production deployments
- [ ] Documentation comprehensive
- [ ] Long-term support plan

## Timeline Estimates

**Current Status (2025-08-28):** Session 3 Complete
- Feature-complete alpha with test issues

**Projected Milestones:**
- **2.0.0-alpha**: 1-2 weeks (fix test fixtures)
- **2.0.0-beta**: 4-6 weeks (robust testing)
- **2.0.0-rc**: 8-12 weeks (server/run implementation)  
- **2.0.0-stable**: 16-20 weeks (community validation)

## Risk Mitigation

### HuggingFace Cache Compatibility (CRITICAL)

**Apple MLX Team & HuggingFace Hub Integration:**
- **~20+ MLX ecosystem users** depend on cache stability
- **HuggingFace Hub attention** - changes monitored by upstream
- **Cache structure**: MLX-Knife follows HuggingFace standards

**Cache Safety Guidelines:**
```markdown
### Shared Cache Environment Best Practices
- **Read operations** (`list`, `health`, `show`): Always safe with concurrent processes
- **Write operations** (`pull`, `rm`): Coordinate with team during maintenance windows
- **Lock cleanup**: Automatic in MLX-Knife, avoid during active HuggingFace downloads
- **User responsibility**: Coordinate cache access, no special flags needed
```

### Parallel Deployment Risks
- **Configuration conflicts**: Different cache paths, environment variables
- **User confusion**: Clear naming and documentation required
- **Maintenance burden**: Supporting two codebases temporarily

### Mitigation Strategies
- **Clear separation**: Different package names, installation paths
- **Comprehensive docs**: Usage examples, best practices, cache guidelines
- **Automated testing**: Both versions in CI/CD pipeline
- **Community support**: Active communication about roadmap

## Decision Authority

**Architecture Decisions:** Development team consensus required
**Version Releases:** Lead maintainer approval + community review
**Breaking Changes:** Major version bump + migration period
**Support Policy:** LTS for stable versions, best-effort for pre-release

---

This versioning strategy provides a clear path from current alpha-quality code to production-ready 2.0.0 while maintaining stability through parallel deployment with 1.x versions.