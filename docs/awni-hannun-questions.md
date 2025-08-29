# Questions for Awni Hannun (MLX Core Developer)

## Context
- **MLX-Knife Issue #26**: Adding embedding support (`mlxk embed` + `/v1/embeddings`)
- **Your endorsement**: MLX-Knife announcement got 🚀 from you
- **Your recommendation**: You promoted `mlx_embedding_models` on Twitter
- **Timeline**: Need release-ready beta in max 1 day

## Questions

### 1. **Integration Strategy** 🎯
```
Hey Awni! Working on adding embedding support to MLX-Knife (Issue #26). 
Would love your thoughts on integrating mlx_embedding_models vs. 
direct MLX implementation for text embeddings. Any gotchas I should know about?
```

**Why asking**: Want authoritative guidance on best approach

### 2. **Technical Direction** 🛠️
```
Planning to add `mlxk embed -m "model" -c "text"` + `/v1/embeddings` endpoint.
Should I use mlx_embedding_models or is there a more "official" MLX way coming?
```

**Why asking**: Avoid implementing something that becomes deprecated

### 3. **Model Compatibility** 🔍
```
Testing with mlx-community/multilingual-e5-base-mlx - 
does mlx_embedding_models handle XLMRobertaModel architectures well?
```

**Why asking**: Want to ensure our test model works reliably

### 4. **MLX Ecosystem Integration** 🤝
```
If this works well, would there be interest in MLX-Knife becoming 
a more "official" part of the MLX ecosystem for local model management?
```

**Why asking**: Gauge long-term collaboration potential

## Expected Benefits

### If Positive Response:
- ✅ **Technical validation** from MLX core team
- ✅ **Avoid implementation pitfalls** 
- ✅ **Community visibility** for MLX-Knife
- ✅ **Future collaboration** opportunities

### If No Response:
- ✅ **Proceed with mlx_embedding_models** (already endorsed)
- ✅ **MIT license compatibility** confirmed
- ✅ **Fallback to direct MLX** if issues arise

## Implementation Plan Post-Response

### Scenario A: **Positive + Guidance**
- Follow Awni's technical recommendations
- Use suggested library/approach
- Implement with confidence

### Scenario B: **No Response**
- Proceed with `mlx_embedding_models` 
- Keep implementation simple & robust
- Ship beta within 1 day constraint

### Scenario C: **Concerns Raised**
- Reassess scope and approach
- Consider postponing Issue #26
- Focus on core MLX-Knife features

---

## Contact Details
- **Discord**: Active MLX community member
- **GitHub**: @awnihannun  
- **Twitter**: @awnihannun
- **Relationship**: Already familiar with MLX-Knife project