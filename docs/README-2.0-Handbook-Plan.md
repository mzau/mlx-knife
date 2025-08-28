# MLX-Knife 2.0 README.md Handbook - Planning Document

**Purpose:** Plan for comprehensive README.md that documents current capabilities and limitations of feature/2.0.0-json-only branch

**Target Audience:** 
- Broke-cluster integration developers
- Early 2.0.0-alpha adopters
- Apple MLX team members
- Community contributors

## Handbook Structure Plan

### 1. **Quick Start Section**
```markdown
# MLX-Knife 2.0.0-alpha - JSON-First Model Management

## Quick Start
```bash
# Installation (local development)
git clone <repo> -b feature/2.0.0-json-only
cd mlx-knife
pip install -e .

# Basic usage
mlxk-json list --json | jq '.data.models[].name'
mlxk-json health --json | jq '.data.summary'
```

**What's New:** JSON-first architecture for automation and scripting
**What's Missing:** Server mode, run command (use MLX-Knife 1.x for those)
```

### 2. **Current Capabilities**
- Complete feature matrix: What works, what doesn't
- JSON API documentation with examples
- Performance characteristics
- Tested platforms and Python versions

### 3. **Limitations & Constraints**
- No server/run functionality (alpha scope)
- Cache safety guidelines for shared environments
- Known test suite issues (10 failing tests)
- HuggingFace cache compatibility notes

### 4. **Migration from 1.x**
- Command comparison table
- Workflow examples
- Parallel deployment strategy
- When to use 1.x vs 2.0

### 5. **Development Status**
- Version roadmap (alpha → beta → rc → stable)
- Test coverage status
- Known issues and workarounds
- Contributing guidelines

## Key Messages to Communicate

### **Alpha Quality Transparency**
```markdown
## ⚠️ Alpha Status Disclaimer

MLX-Knife 2.0.0-alpha is **feature-complete for JSON operations** but has test suite issues:
- **Core functionality works:** All 5 commands (`list`, `health`, `show`, `pull`, `rm`)
- **Test status:** 31/45 passing (mock fixture issues, not core bugs)
- **Production use:** Suitable for broke-cluster integration, not general users yet
- **Parallel use:** Deploy alongside MLX-Knife 1.x for server functionality
```

### **Clear Scope Definition**
```markdown
## What 2.0.0-alpha Includes
✅ `list` - Model discovery with JSON output
✅ `health` - Corruption detection and cache analysis  
✅ `show` - Detailed model information with --files, --config
✅ `pull` - HuggingFace model downloads with corruption detection
✅ `rm` - Model deletion with lock cleanup and fuzzy matching

## What's Coming Later
🔄 `server` - OpenAI-compatible API server (2.0.0-rc)
🔄 `run` - Interactive model execution (2.0.0-rc)
🔄 Human-readable output - CLI formatting layer (2.0.0-rc)
🔄 `embed` - Embedding generation (if merged from 1.x)
```

### **Cache Safety Guidelines**
```markdown
## HuggingFace Cache Safety

MLX-Knife 2.0 respects standard HuggingFace cache structure and practices:

### Best Practices for Shared Environments
- **Read operations** always safe with concurrent processes
- **Write operations** coordinate during maintenance windows  
- **Lock cleanup** automatic but avoid during active downloads
- **Your responsibility:** Coordinate with team, use good timing

### Example Safe Workflow
```bash
# Check what's in cache (always safe)
mlxk-json list --json | jq '.data.count'

# Maintenance window - coordinate with team
mlxk-json rm "corrupted-model" --json --force
mlxk-json pull "replacement-model" --json

# Back to normal operations
mlxk-json health --json | jq '.data.summary'
```

## Content Sections Detail

### Installation Section
- Development installation (pip install -e .)
- Package naming (mlxk-json vs mlxk2 CLI commands)
- Python version requirements (3.9+)
- Dependencies (huggingface-hub, etc.)

### API Documentation
- Complete JSON schema for all 5 commands
- Error response formats
- Exit codes and scripting compatibility
- jq examples for common tasks

### Real-World Examples
- Broke-cluster integration snippets
- CI/CD pipeline usage
- Model management workflows
- Health monitoring automation

### Troubleshooting
- Common error messages and solutions
- Cache corruption recovery workflows
- Test suite issues and workarounds
- Performance tuning for large caches

### Development Info
- Architecture decisions (JSON-first)
- Test suite structure and isolation
- Contributing guidelines
- Roadmap and timeline

## Success Criteria

### Handbook should enable:
- [ ] New user can get started in <5 minutes
- [ ] Clear understanding of alpha limitations
- [ ] Safe usage in shared cache environments
- [ ] Successful broke-cluster integration
- [ ] Confidence in development roadmap

### Community feedback should show:
- [ ] Reduced support questions
- [ ] Successful parallel deployments
- [ ] No cache corruption incidents
- [ ] Increased adoption for automation use cases

## Timeline

**Immediate (Session 3 completion):**
- Create comprehensive README.md
- Document current test status honestly
- Provide clear migration examples

**Before 2.0.0-beta:**
- Update with improved test results
- Add performance benchmarks
- Expand troubleshooting section

**Before 2.0.0-stable:**
- Complete feature documentation
- Add server/run mode examples
- Finalize migration guide

---

This handbook plan ensures users have realistic expectations and can successfully deploy MLX-Knife 2.0.0-alpha in appropriate contexts while maintaining ecosystem stability.