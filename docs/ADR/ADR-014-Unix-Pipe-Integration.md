# ADR-014 — Unix Pipe Integration

- **Status:** Beta (Phase 1 feature complete)
- **Authors:** mlx-knife maintainers
- **Date:** 2025-11-16
- **Updated:** 2025-12-03
- **Target Version:** 2.0.4-beta.1 (Phase 1)
- **Prerequisite:** 2.0.3 (stdout/stderr separation) ✅
- **Related:** ADR-012 (Vision Support), Issue #26 (Embeddings API)

## API Stability

**Phase 1 (2.0.4-beta.1):** Feature complete behind `MLXK2_ENABLE_PIPES=1` gate.

| Aspect | API Stability | Notes |
|--------|---------------|-------|
| `stdin -` | **Stable** | Unix standard, will not change |
| `isatty()` auto-batch | **Stable** | Unix standard, will not change |
| Exit codes 0/1 | **Stable** | Unix standard, will not change |
| `\n\n` separator | **Stable default** | May become configurable, default unchanged |
| `mlx-run` wrapper | **Stable** | Part of toolchain family |
| SIGPIPE/BrokenPipeError | **Stable** | Robust Unix pipe handling |

**Phase 2+ (future):** Additive features only, no breaking changes to Phase 1 API.
- `mlx-tee` tool (new tool, not modifying existing)
- `--in`/`--out` flags (additive, `-` semantics unchanged)

**Gate removal:** When `MLXK2_ENABLE_PIPES=1` is removed (stable release), Phase 1 API is frozen.

## Context

Current `mlxk run` requires all input as command-line arguments. For complex workflows (vision→reasoning chains, multi-stage analysis, preprocessing), users need temporary files:

```bash
# Current: Requires temp files
mlxk run vision_model --image x.jpg "describe" > /tmp/desc.txt
mlxk run reasoning_model "$(cat /tmp/desc.txt)" "analyze for hallucinations"
```

Unix philosophy encourages **composable pipelines** where tools read stdin and write stdout. Tools like `tar`, `docker`, `jq`, and `git` support `-` for pipe integration. Can mlx-knife adopt similar semantics for **model chaining**?

**Motivating use case (vision-reasoning chain):**
```bash
mlxk run vision_model --image cockpit.jpg "Describe what you see" \
  | mlxk run reasoning_model - "Check this description for technical accuracy"
```

## Goals

1. Enable stdin/stdout pipes for model chaining without temp files
2. Maintain backward compatibility (existing CLI unchanged)
3. Follow Unix conventions (`-` for stdin, `isatty()` for output formatting)
4. Support all model types (text, vision, reasoning - generic infrastructure)
5. Keep implementation simple (no complex protocol, just text streams)
6. Establish pipe semantics for local workflows that are **topologically equivalent** to distributed execution (test locally, deploy to cluster without syntax changes)

## Non-Goals

- Binary protocols or structured formats (JSON-RPC, msgpack)
- Bidirectional streaming (stdin→stdout simultaneously)
- Interactive prompts in pipe mode
- Multipart/MIME protocols (use server API for that)
- Cluster orchestration (load balancing, node discovery - see broke-cluster)
- Network protocols for distributed execution (HTTP/gRPC between nodes)
- Production-ready multi-model tools (mlxk-tee is experimental/reference implementation)

## Proposed Design

### Input: Explicit `-` (stdin)

**Syntax:**
```bash
mlxk run <model> -                    # Read entire prompt from stdin
mlxk run <model> - "Additional text"  # stdin + CLI arg combined
```

**Semantics:**
- `-` as positional prompt argument → read from stdin until EOF
- If additional text provided → `f"{stdin_content}\n\n{cli_arg}"`
- Compatible with `--image` flag: `mlxk run vision_model --image x.jpg "describe"`

**Implementation:**
```python
def parse_prompt(args):
    if args.prompt == "-":
        stdin_content = sys.stdin.read()
        if args.additional_text:
            return f"{stdin_content}\n\n{args.additional_text}"
        return stdin_content
    return args.prompt
```

**Why explicit `-` (not implicit `isatty()` check)?**
- Predictable: User declares intent ("I want stdin")
- Avoids surprises: Empty prompt vs. waiting for stdin
- Unix precedent: `tar -`, `docker load -`, `jq -`
- Scripting-safe: `$(mlxk run ...)` doesn't accidentally block on stdin

### Output: Implicit `isatty()` (stdout)

**Behavior:**
```python
if sys.stdout.isatty():
    # TTY → Pretty output with colors, stats, token/sec
    print_colored(response, tokens_per_sec=7.9, ...)
else:
    # Pipe → Clean text only (no ANSI codes, no stats)
    print(response["text"])
```

**Why implicit for stdout?**
- User **expects** clean text in pipes (like `git log`, `ls`)
- No debugging confusion: `| cat` shows what's piped
- Override available: `--json` forces JSON even in TTY

**Critical for Unix tool integration:**
Plain text output is **required** for standard Unix tools:
- `tee` - Log intermediate results
- `grep` - Filter/search responses
- `sed/awk` - Text transformations
- `head/tail` - Truncate output
- `wc` - Count words/lines
- `sort/uniq` - Deduplication

Example (only works with plain text):
```bash
echo "test" | mlxk run model1 - | tee stage1.txt | grep "keyword" | mlxk run model2 -
```

**Edge cases:**
- Reasoning models: Only output `content` (not `reasoning_content`) in pipe mode
- Streaming: Works in both modes (SSE in TTY, plain text in pipe)
- Errors: Always stderr (implemented in 2.0.3), exit code 1
- Exit codes: 0 (success), 1 (error) - standard Unix behavior

## Example Workflows

### Single-Model Chains

#### 1. Vision → Reasoning Chain
```bash
mlxk run mlx-community/Llama-3.2-11B-Vision-Instruct-4bit \
  --image cockpit.jpg "Describe the cockpit in detail" \
  | mlxk run mlx-community/DeepSeek-R1-Distill-Qwen-14B - \
  "Analyze this description for hallucinations"
```

#### 2. Multi-stage Analysis
```bash
cat codebase.py \
  | mlxk run model1 - "Extract function signatures" \
  | mlxk run model2 - "Generate unit tests for these"
```

#### 3. Batch Processing
```bash
for img in *.jpg; do
  mlxk run vision_model --image "$img" "describe"
done | mlxk run summarizer_model - "Summarize these descriptions"
```

#### 4. Interactive (TTY) - No change
```bash
mlxk run model "test"  # Pretty output with stats (as today)
```

### Multi-Model Workflows (mlxk-tee)

**Motivation:** Test distributed topologies locally before scaling to cluster. Different models locally = different nodes in cluster (topologically equivalent).

**⚠️ RESOURCE WARNING:** Parallel LLM inference on unified memory (Apple Silicon) can cause severe RAM pressure. Focus on **embedding use cases** where models are small and can run on CPU (`--cpu-only`), keeping GPU free for primary LLM workload. Advanced users may run parallel LLM inference but must understand RAM constraints.

#### 1. Parallel Embedding (RECOMMENDED - Safe & Practical)
```bash
# Embed with multiple models for ensemble retrieval
echo "machine learning tutorial" \
  | mlxk-tee \
      "mlxk embed nomic-embed --cpu-only -" \
      "mlxk embed bge-small --cpu-only -" \
  | combine-embeddings > ensemble.jsonl
```

**Use Case:** Multi-model embeddings for robust RAG retrieval
**Safety:** Small models (~500MB each), CPU-only keeps GPU free
**Cluster equivalent:** Two embedding nodes

#### 2. Model A/B Testing (⚠️ ADVANCED - RAM intensive)
```bash
# Compare two models on same prompt
# WARNING: Both models loaded simultaneously!
echo "Explain quantum computing" \
  | mlxk-tee \
      "mlxk run phi-3-mini -" \
      "mlxk run deepseek-r1 -" \
  | diff -y --width=160
```

**Use Case:** Quality comparison, regression testing
**⚠️ Risk:** High RAM usage (both models in memory)
**Safer alternative:** Sequential testing or cluster deployment
**Cluster equivalent:** Two nodes with different models

#### 3. Multi-Source Embedding (RECOMMENDED - Safe)
```bash
# Index documents with multiple embedding models
cat docs/*.md | while read doc; do
  echo "$doc" | mlxk-tee \
    "mlxk embed semantic-model --cpu-only - >> semantic.jsonl" \
    "mlxk embed keyword-model --cpu-only - >> keyword.jsonl"
done
```

**Use Case:** Hybrid search (semantic + keyword embeddings)
**Safety:** CPU-only embeddings, no GPU contention
**Cluster equivalent:** Parallel embedding nodes

#### 4. Ensemble / Majority Voting (⚠️ ADVANCED)
```bash
# Three models vote on classification
# WARNING: 3 models loaded simultaneously!
echo "Sentiment: 'meh, it's okay'" \
  | mlxk-tee \
      "mlxk run model1 - 'Answer: positive or negative'" \
      "mlxk run model2 - 'Answer: positive or negative'" \
      "mlxk run model3 - 'Answer: positive or negative'" \
  | python3 majority-vote.py
```

**Use Case:** Robust classification, consensus-based decisions
**⚠️ Risk:** Very high RAM usage (3 models)
**Safer alternative:** Cluster deployment recommended
**Cluster equivalent:** Three nodes for redundancy

#### 5. Parallel Vision + Embedding (RECOMMENDED - Safe pattern)
```bash
# Process image for analysis AND indexing simultaneously
mlxk run vision-model --image photo.jpg "describe" \
  | mlxk-tee \
      "cat > description.txt" \
      "mlxk embed vision-embed --cpu-only - >> index.jsonl"
```

**Use Case:** RAG indexing while preserving description
**Safety:** Only vision LLM on GPU, embedding on CPU
**Cluster equivalent:** Vision on Node1, Embedding on Node2

#### 6. Speed vs Quality Trade-off (⚠️ ADVANCED)
```bash
# Fast preview + thorough analysis in parallel
# WARNING: 2 models loaded simultaneously!
echo "Analyze: $(cat script.py)" \
  | mlxk-tee \
      "mlxk run fast-small - | notify-user" \
      "mlxk run slow-large - | save-to-db"
```

**Use Case:** Instant feedback + high-quality archival
**⚠️ Risk:** RAM contention between models
**Safer alternative:** Sequential execution or cluster
**Cluster equivalent:** Fast node (M1) + slow node (Cloud GPU)

#### 7. Multi-Format Output (Safe - no parallel LLMs)
```bash
# Save output to multiple formats
mlxk run model "Implement binary search in Python" \
  | mlxk-tee \
      "tee code.py" \
      "python3 -m py_compile -" \
      "wc -l > metrics.txt"
```

**Use Case:** Multiple outputs from single inference
**Safety:** No parallel LLMs, just stream processing
**Cluster equivalent:** Single node, multiple consumers

#### 8. Pipeline Debugging (Safe)
```bash
# Log every stage of complex pipeline
cat input.txt \
  | mlxk run stage1 - \
  | mlxk-tee "tee stage1.log" "mlxk run stage2 -" \
  | mlxk-tee "tee stage2.log" "mlxk run stage3 -" \
  | tee final.log
```

**Use Case:** Debugging multi-stage workflows
**Cluster equivalent:** Distributed pipeline with centralized logging

#### Summary: Safe vs Advanced Patterns

**✅ SAFE (Recommended for local development):**
1. **Parallel embeddings** with `--cpu-only` (GPU stays free for LLM)
2. **Stream processing** (tee, wc, grep - no model loading)
3. **Sequential LLM calls** (one model at a time)
4. **Vision → Embedding** (LLM on GPU, embedding on CPU)

**⚠️ ADVANCED (Cluster recommended):**
1. **Parallel LLM inference** (2+ large models simultaneously)
2. **Ensemble voting** (3+ models = high RAM pressure)
3. **Speed vs Quality** (concurrent large models)

**Rule of thumb:**
- **Local:** Embeddings in parallel (safe), LLMs sequential
- **Cluster:** LLMs in parallel across nodes (distributed RAM)

## RAG Use Cases (Low-Hanging Fruit)

**Motivation:** Pipe integration + embeddings (Issue #26) enables simple RAG workflows without complex infrastructure (FAISS, ChromaDB, LangChain). Unix philosophy: **composable tools** > monolithic frameworks.

### Classic RAG Pipeline (Stateless)
```
Query → Embed → Vector Search → Retrieve Context → LLM → Answer
```

**As Unix Pipes:**
```bash
echo "What is MLX?" \
  | mlxk embed model - \
  | vector-search index.db - --top-k 3 \
  | mlxk run chat-model - "Answer based on this context"
```

### What Works TODAY (No Embeddings)

#### 1. Keyword-Based RAG (grep/ripgrep)
```bash
# "Poor man's RAG" - surprisingly effective!
rg "stop.*token" docs/ --context 5 \
  | mlxk run chat-model - "Explain stop token handling"
```

#### 2. LLM as Retriever (Expensive but Creative)
```bash
# Two-stage: Filter → Analyze
cat all_documentation.txt \
  | mlxk run small-fast-model - "Extract sections about vision support" \
  | mlxk run large-reasoning-model - "Analyze the vision roadmap"
```

#### 3. Multi-Document Synthesis
```bash
# Combine multiple sources
cat design_doc.md spec.md changelog.md \
  | mlxk run analyst-model - "Compare planned vs implemented features"
```

### What Works WITH Embeddings (Issue #26)

**Once `mlxk embed` is implemented:**

#### 1. Codebase Q&A
```bash
# Index once (one-time setup)
find . -name "*.py" -exec cat {} \; \
  | mlxk embed code-model - --batch --output codebase.jsonl

# Query repeatedly (fast)
echo "How does stop token detection work?" \
  | mlxk embed code-model - \
  | cosine-search codebase.jsonl - --top-k 5 \
  | mlxk run chat-model - "Explain based on this code"
```

#### 2. Semantic Code Search
```bash
# Find similar implementations
mlxk embed code-model --file runner.py "function generate_batch" \
  | vector-search codebase.jsonl - --top-k 10 \
  | mlxk run code-model - "Compare these implementations for differences"
```

#### 3. Document Analysis (Vision + RAG)
```bash
# Multi-modal RAG: Image → Text → Semantic Search → Analysis
mlxk run vision_model --image invoice.pdf "Extract all text" \
  | tee >(mlxk embed doc_model -) \
  | mlxk run analyst_model - "Extract key financial data"
```

#### 4. Simple JSON "Vector Store"
```bash
# No FAISS/Chroma needed - JSONL is enough for small datasets!
# embeddings.jsonl format:
{"text": "...", "embedding": [0.1, 0.2, ...], "metadata": {...}}

# Search with jq + Python one-liner
cat embeddings.jsonl \
  | python3 -c "import sys, json, numpy as np
query = json.loads(sys.argv[1])
for line in sys.stdin:
    doc = json.loads(line)
    score = np.dot(query['embedding'], doc['embedding'])
    if score > 0.8:
        print(json.dumps({'text': doc['text'], 'score': score}))" \
  "$(echo 'query text' | mlxk embed model -)"
```

### Why Constraints are Features

| Property | Limitation | **Benefit** |
|----------|------------|-------------|
| **Stateless** | No persistence between runs | Reproducible, git-versionable, testable |
| **Text Streams** | No binary vector formats | Debuggable with `\| less`, `\| head`, `tee` |
| **Separate Tools** | Need external vector search | Composable, Unix-like, no vendor lock-in |
| **No Vector DB** | Slower than FAISS (linear scan) | No infrastructure, works on laptop, CI/CD-friendly |
| **JSONL Format** | Not optimized for GB-scale | Human-readable, jq-queryable, git-diffable |

**Sweet spot:** 1K-100K documents (most codebases, documentation sets, personal knowledge bases)

### External Tools for Vector Search

**Lightweight (no dependencies):**
```bash
# Pure Python (numpy only)
pip install numpy
# cosine-search.py (100 LOC, ships with mlx-knife examples/)
```

**Production-ready:**
```bash
# sqlite-vss (SQLite extension)
pip install sqlite-vss
sqlite3 embeddings.db "CREATE VIRTUAL TABLE vec USING vss0(...)"

# usearch (fast, no server)
pip install usearch
```

**Framework integration:**
```bash
# LangChain (if you must)
echo "query" | mlxk embed model - | python langchain_search.py
```

### Issue #26 Motivation

**What mlx-knife needs:**
- CLI: `mlxk embed <model> <text>` or `mlxk embed <model> -`
- Server: `/v1/embeddings` (OpenAI-compatible)
- Batch mode: `--batch` for processing large documents
- Output: JSON or JSONL (pipe-friendly)

**Low-hanging fruit approach:**
1. ✅ Pipe integration (ADR-014) - enables composition
2. 🔄 Embeddings API (Issue #26) - completes the RAG loop
3. 📦 Ship example `cosine-search.py` (100 LOC, no framework)
4. 📚 Document JSONL-based RAG patterns (README)
5. 🎯 **No need for complex RAG frameworks** - Unix pipes are the framework!

**User benefits:**
- RAG without LangChain/LlamaIndex (lighter dependencies)
- Debuggable workflows (`set -x`, pipe inspection)
- Testable pipelines (fixtures = text files)
- Git-versionable indexes (JSONL in repo)
- CI/CD friendly (no vector DB server needed)

## Implementation Plan

### Phase 1: stdin/stdout Support for mlxk run (2.0.4-beta.1) ✅ COMPLETE

**Critical:** This phase MUST be implemented before ADR-015 (embeddings). Minimal viable pipe semantics to enable embed use cases.

**Scope:** `mlxk run` with `-` stdin support

**Implementation Tasks:**

- [x] Add `-` parsing to `mlxk2/cli.py` argument parser
- [x] Update `mlxk2/operations/run.py` to read stdin when `-` detected
- [x] Handle combined stdin + additional text: `stdin_content + "\n\n" + additional_text`
- [x] Detect `isatty()` in output formatting:
  - TTY: Pretty output (colors, stats, streaming)
  - Pipe: Clean text only (no ANSI, no stats, batch mode)
- [x] Automatic batch mode in pipes: `if not sys.stdout.isatty(): stream = False`
- [x] Implement `mlx-run` wrapper tool (pyproject.toml + 10 LOC wrapper)
- [x] SIGPIPE handler (`signal.signal(SIGPIPE, SIG_DFL)`) for robust pipe termination
- [x] BrokenPipeError handling in streaming + batch output
- [x] Unit tests for stdin edge cases (empty, stdin-only, trailing text)
- [x] Unit tests for SIGPIPE and BrokenPipeError handling
- [x] Regression tests: Ensure TTY behavior unchanged

**Enables:**
- ✅ Model chains: `mlxk run model1 - | mlxk run model2 -`
- ✅ ADR-015 implementation: `mlxk embed model -` uses same stdin semantics
- ✅ RAG pipelines: `mlxk embed - | cosine-search - | mlxk run -`

**Explicitly NOT in Phase 1:**
- ❌ `--stream` flag (YAGNI - automatic batch in pipes is sufficient)
- ❌ Reasoning-specific output handling (defer to ADR-010)
- ❌ Vision integration (defer to ADR-012)
- ❌ `--in` / `--out` flags (defer to Phase 2+)

### Phase 2: Example Scripts (2.0.4+)

**Note:** `mlx-tee` is not a core mlx-knife tool but an example script demonstrating
parallel model execution. For production distributed workflows, see broke-cluster.

- [x] `examples/mlx-tee.py` - Reference implementation (~100 LOC)
  - Broadcast stdin to multiple commands in parallel
  - SSH placeholder for remote execution (`@node:command`)
  - ThreadPoolExecutor for parallel execution
- [ ] `examples/cosine-search.py` - Vector search for RAG (depends on ADR-015)
- [ ] `examples/rag-pipeline.sh` - End-to-end RAG example

### Phase 3: Documentation (2.0.3 or 2.1)
- [ ] README: Unix Pipes section with single-model + multi-model examples
- [ ] TESTING-DETAILS.md: Pipe mode test cases
- [ ] `mlxk run --help`: Document `-` syntax
- [ ] `examples/README.md`: mlxk-tee usage guide
- [ ] Document topological equivalence (local models ↔ cluster nodes)

### Phase 4: Server Integration (Future - 2.4+)
- [ ] `/v1/completions` with `stream: true` → SSE
- [ ] Client-side pipe emulation (broke-nchat)
- [ ] Server stays simple (no pipe protocol)

## Risks & Mitigation

### Risk 1: Parallel LLM Inference (RAM Pressure)
**Problem:** `mlxk-tee` with multiple LLMs loads all models simultaneously → OOM on unified memory systems

**Mitigation:**
- **Primary use case: Embeddings** (small models, `--cpu-only`)
- Document safe patterns (embeddings parallel, LLMs sequential)
- Warning in examples: "⚠️ ADVANCED - RAM intensive"
- Recommend cluster for parallel LLM workflows
- Future: RAM budget check before parallel execution

### Risk 2: Binary Input (images as stdin)
**Problem:** `mlxk run vision-model -` could confuse users (image as binary stdin?)

**Mitigation:**
- Vision models REQUIRE `--image <path>` flag (no binary stdin)
- `-` only for text prompts
- Error message: "Vision models require --image flag, cannot use binary stdin"

### Risk 3: Large Input (RAM)
**Problem:** `cat 10GB.txt | mlxk run model -` could OOM

**Mitigation:**
- Document: stdin is buffered in RAM (warn about large inputs)
- Future: Streaming stdin processing (chunked prompts)
- Current: Practical limit ~100MB (acceptable for most use cases)

### Risk 4: Prompt Injection via Pipes
**Problem:** Malicious content in piped data

**Mitigation:**
- Not our concern (user controls pipeline)
- Document: Sanitize untrusted input before piping
- Same risk as `cat untrusted.txt | bash` (Unix philosophy)

### Risk 5: Backward Compatibility
**Problem:** Existing scripts using `-` as prompt text

**Mitigation:**
- Unlikely: `-` is unusual prompt text
- Feature gate: `MLXK2_ENABLE_PIPES=1` (optional, for alpha)
- Can graduate to default in 2.1 after validation

## Alternatives Considered

### Alternative 1: Explicit `--input -` / `--output -` flags
**Rejected:** Verbose, breaks Unix conventions (most tools use `-` directly)

### Alternative 2: Implicit stdin (no `-` needed)
```bash
echo "test" | mlxk run model  # Implicit stdin
```
**Rejected:** Confusing when stdin is empty, scripting pitfalls, unpredictable

### Alternative 3: Named pipes / temp files
**Rejected:** Not cross-platform, requires cleanup, complexity

### Alternative 4: Server-side pipe protocol
**Rejected:** Scope creep, mlx-knife is CLI-first, server stays simple

### Alternative 5: JSON-by-default in pipes
**Considered:** Make all pipe output JSON for robustness (error propagation, metadata)

**Rejected for Phase 1:**
- Breaks Unix tool integration (`tee`, `grep`, `sed`, `awk`, `wc`, `head/tail`, `sort/uniq`)
- JSON output makes standard text processing impossible
- Example broken workflow: `mlxk run model - | tee log.txt | grep "keyword"`
  - With JSON: `grep` matches JSON syntax, not response content
  - With plain text: Works as expected

**Phase 1 Decision:**
- Plain text default (Unix compatibility)
- `--json` only for terminal output or external tools (jq, broke-cluster)
- Accept limitation: No metadata in pipes (use server API if needed)

**Future Evaluation:**
- Gather usage data: mlxk→mlxk chains vs mlxk→Unix tool usage
- Consider: stderr JSON metadata, separate metadata files, or `mlxk-pipe` wrapper
- Defer decision until real-world usage patterns emerge

## `--json` Flag Behavior

**Terminal (TTY):**
```bash
mlxk run model "test" --json
# Output: {"status": "success", "data": {"response": "..."}, ...}
```

**In Pipes (NOT RECOMMENDED):**
```bash
# Problem: JSON output breaks Unix tools
echo "test" | mlxk run model - --json | grep "keyword"
# grep matches JSON keys/syntax, not response content ❌

# Workaround (if JSON needed for external tools):
echo "test" | mlxk run model - --json | jq -r '.data.response' | mlxk run model2 -
# But this is complex and fragile
```

**Rule for Phase 1:**
- Don't use `--json` when piping to Unix tools or other `mlxk` commands
- Use plain text (auto-detect) for pipe workflows
- Use `--json` only for terminal inspection or external consumers (jq scripts, broke-cluster)

**Future Consideration:**
- Auto-detect JSON input (parse if stdin is mlxk JSON format)
- `--json` could mean: "JSON output + parse JSON input if available"
- Deferred to post-Phase 1 based on user feedback

## Open Questions

1. **Reasoning models:** Pipe mode outputs only `content` or also `reasoning_content`?
   - Proposal: Add `--think-only` flag for reasoning-only output
   - Default pipe: `content` (user-facing answer)

2. **Error handling:** Partial output on error (streaming), or no output?
   - Proposal: stderr for errors, stdout partial results (like `grep`)

3. **Multiline prompts:** How to combine stdin + CLI arg?
   - Proposal: `"{stdin}\n\n{cli_arg}"` (paragraph separator)

4. **Feature gate:** Ship behind `MLXK2_ENABLE_PIPES=1` or direct to main?
   - Proposal: Direct to main (low risk, well-established Unix pattern)

5. **Embeddings output format:** Should `mlxk embed` output raw JSON arrays or JSONL with metadata?
   - Proposal: JSONL with `{"text": "...", "embedding": [...], "metadata": {...}}` (pipe-friendly, extendable)
   - Allows `| jq`, `| grep`, easy inspection

6. **Embedding execution mode:** Should `mlxk embed` default to `--cpu-only`?
   - Proposal: YES - Keep GPU free for primary LLM workload
   - Embeddings are small (100MB-1GB) and fast enough on CPU
   - User can override with `--gpu` if needed
   - Critical for safe parallel embedding workflows (`mlxk-tee`)

## Success Criteria

### Phase 1 (Complete)
- [x] `cat prompt.txt | mlxk run model -` works (basic stdin)
- [x] `mlxk run model - "extra"` combines stdin + arg
- [x] TTY output unchanged (colors, stats preserved)
- [x] Pipe output clean (no ANSI, only text)
- [x] **Exit codes:** 0 (success), 1 (error) - pipes abort on error
- [x] **SIGPIPE handling:** `mlxk run model | head -1` terminates cleanly
- [x] **BrokenPipeError:** Streaming/batch output handles pipe closure gracefully
- [x] No regression in existing CLI behavior
- [x] `--json` flag documented as NOT for pipe workflows (terminal/jq use only)

### Phase 2 (Examples - Complete)
- [x] `examples/mlx-tee.py` - Parallel model execution with SSH placeholder

### Future (Depends on ADR-015 Embeddings)
- [ ] **Unix tools work:** `mlxk run model - | tee log.txt | grep "keyword" | wc -l`
- [ ] Vision chains work: `mlxk run vision_model --image x.jpg "describe" | mlxk run chat_model -`
- [ ] RAG workflow: `mlxk embed model - | cosine-search - | mlxk run chat_model -`
- [ ] `examples/cosine-search.py` - Vector search for RAG
- [ ] `examples/rag-pipeline.sh` - End-to-end RAG example

## Future: Distributed Execution

The pipe semantics defined in this ADR are **location-agnostic by design**. While mlx-knife focuses on local execution, the same stdin/stdout contracts enable distributed execution.

**Topological Equivalence:**
- Local: Different models on one machine → Test workflows locally
- Cluster: Different nodes in network → Deploy same workflows distributed

**Example (identical syntax):**
```bash
# Local development (mlx-knife)
echo "test" | mlxk-tee "mlxk run model1 -" "mlxk run model2 -"

# Production cluster (broke-cluster)
echo "test" | broke-tee "node1:model1" "node2:model2"
```

**Out of scope for mlx-knife:**
- Cluster orchestration (load balancing, node discovery)
- Network protocols (HTTP/gRPC between nodes)
- Distributed state management

These concerns are addressed in the broke-cluster project, which builds on this ADR's pipe semantics.

## Next Steps

1. **Validation:** Prototype in feature branch (2-3 hours implementation)
2. **Testing:** Unit tests + manual workflow validation
3. **mlxk-tee implementation:** Multi-model broadcast tool (examples/)
4. **Documentation:** README examples + mlxk-tee usage guide
5. **Decision:** Ship in 2.0.3 (quick win) or 2.1 (with vision support)?
6. **Community feedback:** Reddit/Discord examples to gauge adoption
7. **Issue #26 synergy:** Once pipes are stable, embeddings API enables RAG workflows (ship example `cosine-search.py`)

---

**Status:** Beta - Phase 1 feature complete (2.0.4-beta.1).

**Completed effort:**
- Phase 1 (stdin/stdout): ✅ Complete (2025-12-03)
- Phase 2 (examples/mlx-tee.py): ✅ Complete (2025-12-03)

**Remaining effort:**
- Phase 3 (docs): 0.5 session
- Future examples (cosine-search.py, rag-pipeline.sh): Depends on ADR-015

**Compatibility:** 100% backward compatible (additive feature, behind gate)

---

## Appendix A: Toolchain Familie (Design Exploration)

This ADR establishes pipe semantics for a **family of specialized Unix-style tools** built on shared MLX infrastructure.

### Proposed Toolchain

**Core Management:**
- `mlxk` - Model management (list, health, pull, serve, rm)

**Execution Tools:**
- `mlx-run` - Direct model execution (Phase 1 - this ADR)
- `mlx-tee` - Multi-model broadcast (Phase 2 - this ADR)
- `mlx-embed` - Embeddings (ADR-015)

**Vision Tools (ADR-012):**
- `mlx-img` - Image → text (vision models)

**Conversion Tools (future):**
- `mlx-convert` - Format conversion (GGUF ↔ MLX, quantization)

**Cluster Tools (broke-cluster project):**
- `broke-run` - Distributed execution (topologically equivalent to `mlx-run`)
- `broke-tee` - Distributed multi-model (topologically equivalent to `mlx-tee`)

### Design Principles

1. **Shared Core:** All tools use `MLXRunner`, `mlxk2.core`, same model loading
2. **Consistent Pipes:** All tools follow `-` for stdin, `isatty()` for output
3. **Specialization:** Each tool optimized for specific use case
4. **Composability:** Tools chain naturally via Unix pipes
5. **Topological Equivalence:** Local tools (`mlx-*`) have distributed equivalents (`broke-*`) with identical syntax

### Rationale: Why NOT "implicit run"?

**Considered:** Making `mlxk` auto-detect pipe mode and execute models implicitly:
```bash
cat prompt.txt | mlxk phi-3    # Implicit run in pipe mode?
```

**Rejected because:**
1. **Parser Complexity:** Must disambiguate commands vs models (expensive fuzzy-matching)
2. **Namespace Pollution:** Can never use model names matching commands
3. **Inconsistency:** Different behavior TTY vs Pipe (violates principle of least surprise)
4. **Future-Brittleness:** Every new command blocks a model name

**Preferred Solution:** Separate tools with clear purpose:
- `mlxk` = Management (never ambiguous)
- `mlx-run` = Execution (compact, pipe-native)
- `mlx-tee` = Multi-model (specialized)

### Implementation (Phase 1)

`mlx-run` is trivial wrapper around `mlxk run`:

```python
# pyproject.toml
[project.scripts]
mlxk = "mlxk2.cli:main"
mlx-run = "mlxk2.tools.run:main"

# mlxk2/tools/run.py
def main():
    """mlx-run: Direct model execution."""
    sys.argv.insert(1, 'run')
    from mlxk2.cli import main as mlxk_main
    mlxk_main()
```

**Effort:** 10 lines of code, zero parser complexity increase.

**User Experience:**
```bash
# Traditional (always works):
mlxk run phi-3 "test"
cat prompt.txt | mlxk run phi-3 -

# Compact (pipe-optimized):
mlx-run phi-3 "test"
cat prompt.txt | mlx-run phi-3 -

# Power users can alias:
alias m='mlx-run'
cat prompt.txt | m phi-3 -
```

---

## Appendix B: Advanced Input/Output Semantics (WIP)

**Status:** Design exploration for future phases. NOT part of Phase 1 implementation.

### Motivation

Phase 1 provides basic stdin/stdout pipes. Advanced workflows may require:
- **Multi-source input:** Combine stdin + multiple files (RAG, context injection)
- **Pipeline debugging:** Capture intermediate outputs without breaking chains
- **Structured prompts:** CLI equivalent of server's message arrays

### Proposed Extensions (Post-Phase 1)

#### Extension 1: Multi-Source Input (`--in`)

**Use Case:** Combine query (stdin) with knowledge bases (files):

```bash
echo "How does MLX quantization work?" | mlx-run model \
  --in docs/quantization.md \
  --in examples/4bit.py \
  -
```

**Open Questions:**
1. **Combination semantics:** How to merge inputs?
   - Simple concatenation: `file1 + "\n\n" + file2 + "\n\n" + stdin`
   - Structured labels: `"Context 1:\n" + file1 + "\n\nQuery:\n" + stdin`
   - Chat-template aware: Map to message array (see below)

2. **Order:** `--in` files first, then stdin? Configurable?

3. **Separator control:** `--separator "\n---\n"` flag?

**Relationship to Server API:**
```json
// Server uses message arrays:
POST /v1/chat/completions
{
  "messages": [
    {"role": "system", "content": "<file1>"},
    {"role": "user", "content": "<stdin>"}
  ]
}
```

**CLI could map to same structure:**
```bash
mlx-run model --system system.txt --in context.txt - "query"
# → Internally builds message array, applies chat template
```

**Related Issues/Features:**
- `--system` flag (cli.py:131, already stubbed for future)
- ADR-010 (Reasoning Content API) - message-based
- Issue #39 (OpenAI Function Calling) - message-based
- Multi-turn conversations (run.py:132 interactive_chat)

**All use Chat Templates + Message Arrays!** This suggests a unified approach.

---

#### Extension 2: Pipeline Debugging (`--out`)

**Use Case:** Capture intermediate stages without breaking pipe:

```bash
cat input.txt \
  | mlx-run stage1 --out log/stage1.txt - \
  | mlx-run stage2 --out log/stage2.txt - \
  | mlx-run stage3 -
```

**Semantics:** Write to file AND stdout (tee-like behavior).

**Alternative:** Use Unix `tee`:
```bash
cat input.txt | mlx-run stage1 - | tee log/stage1.txt | mlx-run stage2
```

**Trade-off:** `--out` is more compact, but `tee` already exists and works. Low priority.

---

### Design Constraints for Phase 1

To preserve future compatibility with these extensions, Phase 1 MUST:

1. **Reserve argument syntax:**
   - `--in <file>` - Reserved for multi-source input
   - `--out <file>` - Reserved for output capture
   - `--system <text|file>` - Already reserved (cli.py:131)

2. **Semantic foundation:**
   - `-` always means stdin (never conflicts with `--in`)
   - Additional text after `-` always allowed: `mlx-run model - "instruction"`
   - This enables future: `mlx-run model --in file - "instruction"`

3. **Chat template compatibility:**
   - Internal prompt construction must support future message array mapping
   - `--no-chat-template` flag preserves raw mode for testing

4. **Output modes remain simple:**
   - stdout = main output channel
   - stderr = errors only
   - `--out` (future) adds file, doesn't replace stdout

### Phase 1 Minimal Implementation

**What Phase 1 WILL implement:**
```bash
# Basic stdin:
cat prompt.txt | mlx-run model -

# stdin + additional:
cat context.txt | mlx-run model - "Answer this question"

# TTY mode unchanged:
mlx-run model "prompt"

# Output adapts automatically:
mlx-run model "test"          # TTY: pretty
mlx-run model "test" | cat    # Pipe: clean text
```

**What Phase 1 will NOT implement:**
- `--in` (multi-source)
- `--out` (tee-like)
- `--system` (stubbed but not functional)
- Chat-template aware input structuring

**Phase 1 is intentionally minimal** to validate core pipe semantics before adding complexity.

---

### Connection to Server Architecture

The server already implements message-array semantics via Chat Templates:

```python
# mlxk2/core/server_base.py (existing)
def _format_conversation(messages: List[dict]) -> str:
    """Apply chat template to message array."""
    return tokenizer.apply_chat_template(messages, ...)
```

**Future CLI-Server symmetry:**

| Feature | Server API | CLI (Future) | Shared Implementation |
|---------|-----------|--------------|----------------------|
| System prompts | `messages[0].role = "system"` | `--system file.txt` | `_format_conversation()` |
| Multi-turn | `messages[]` array | `--in file1 --in file2` | Chat template engine |
| Context injection | `messages[].content` | `--in context.txt` | Token management |
| Reasoning content | `reasoning_content` field | `--show-reasoning` | ADR-010 parser |

**Benefit:** CLI becomes a thin wrapper around server logic, ensuring consistency.

---

### Recommendation

**Phase 1 (2.0.3-beta.x):**
- Implement: Basic stdin (`-`), stdout adaptation (`isatty()`)
- Reserve: `--in`, `--out`, `--system` syntax
- Document: Minimal viable pipe semantics

**Phase 2+ (Post-2.0.3):**
- Evaluate: Real-world usage patterns from Phase 1
- Design: Multi-source (`--in`) semantics based on feedback
- Implement: Chat-template aware input mapping (CLI-Server symmetry)
- Consider: `--out` if `tee` proves insufficient

**This appendix serves as design documentation, not implementation commitment.**
