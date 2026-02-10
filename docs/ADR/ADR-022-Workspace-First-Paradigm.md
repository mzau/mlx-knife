# ADR-022: Workspace-First Paradigm

**Status:** Draft (Discussion)
**Created:** 2026-02-06
**Related:** ADR-018 (Convert Operation), SECURITY.md
**Target:** 2.0.5

---

## Context

### The HuggingFace Cache Problem

The HF cache (`$HF_HOME/hub/`) is a **shared mutable namespace** used by multiple uncoordinated actors:

```
$HF_HOME/hub/
├── models--mlx-community--whisper-large-v3-mlx/  ← mlx-knife pull
├── models--Qwen--Qwen2.5-7B/                      ← mlx-audio runtime (!!)
└── ...
```

**Actors writing to the cache:**
- `transformers` (AutoTokenizer, AutoModel)
- `mlx-lm` (model loading)
- `mlx-vlm` (vision model loading)
- `mlx-audio` (audio model loading, **including undeclared dependencies**)
- `huggingface_hub` (downloads)

**This creates classic shared-state problems:**

| Problem | Description | Example |
|---------|-------------|---------|
| Undeclared dependencies | Runtime downloads not visible at pull time | VibeVoice needs Qwen2.5-7B tokenizer |
| Write pollution | Upstream libs modify cache during inference | mlx-audio downloads during `run` |
| No isolation | All libs see and write same namespace | Cross-model interference possible |
| Implicit state | "Works after first run" syndrome | Cache state determines behavior |

### The Broken Promise

SECURITY.md currently states:
> "Network activity is limited to explicit interactions with Hugging Face: downloading models (pull)"

This promise is **broken** when upstream libraries download during `run`:

```bash
mlxk pull VibeVoice-ASR-4bit   # ✓ Model downloaded
# Network disabled
mlxk run VibeVoice --audio x.wav  # ✗ Fails - needs Qwen2.5-7B
```

### What mlx-knife Controls

| Layer | Control | Can Guarantee |
|-------|---------|---------------|
| mlx-knife CLI | Full | Own behavior |
| mlx-lm / mlx-vlm / mlx-audio | None | Nothing |
| HuggingFace Hub | None | Nothing |
| Model repositories | None | Nothing |

**Reality:** mlx-knife is an integration layer. It can recommend models but cannot guarantee their behavior remains constant.

---

## Decision

### Workspace as Primary Paradigm

Shift from HF-cache-centric to workspace-centric model management:

**Current (2.0.4):**
```bash
mlxk pull Model        → $HF_HOME (shared, uncontrolled)
mlxk run Model         → reads from shared cache
                       → upstream may write to cache (hidden)
```

**New (2.0.5):**
```bash
mlxk clone Model ./models/Model   → local workspace (controlled)
mlxk run ./models/Model           → reads from workspace
                                  → side effects visible in .hf_cache/
```

### Workspace-Local Cache

Each workspace gets an isolated HF cache for runtime artifacts:

```
./models/
├── whisper-large-v3-mlx/         # cloned model
├── VibeVoice-ASR-4bit/           # cloned model
└── .hf_cache/                    # workspace-local cache
    └── Qwen--Qwen2.5-7B/         # runtime artifact (VISIBLE!)
```

**Implementation:** When running from workspace path (`./`), set:
```bash
HF_HOME=<workspace>/.hf_cache
```

### Isolation Guarantees

| Guarantee | HF Cache | Workspace |
|-----------|----------|-----------|
| Model isolation | No | Yes (per-workspace) |
| Side effects visible | No (hidden in ~/.cache) | Yes (.hf_cache/) |
| Reproducible | No | Yes (tar/zip/archive) |
| Auditable | Difficult | Trivial (`ls -la`) |
| Offline after first run | Unknown | Yes (everything local) |

### What mlx-knife CAN and CANNOT Guarantee

**CAN guarantee (workspace mode):**
- Models are isolated from each other
- Runtime artifacts are visible in `.hf_cache/`
- After successful first run, all dependencies are local
- Workspace can be archived/transferred

**CANNOT guarantee:**
- Upstream libraries won't attempt network access
- First run won't download additional artifacts
- Model behavior remains constant over time

### Revised Security Promise

Update SECURITY.md to reflect reality:

> **Network Activity**
>
> mlx-knife itself performs network activity only during explicit commands (`pull`, `clone`, `push`).
>
> **Important:** mlx-knife integrates upstream libraries (mlx-lm, mlx-vlm, mlx-audio) whose behavior is outside our control. These libraries may perform their own network requests during model loading or inference.
>
> **For offline/air-gapped environments:**
> 1. Use `mlxk clone` to create isolated workspaces
> 2. Run the model once (online) to capture all runtime dependencies
> 3. Verify `.hf_cache/` contains all artifacts
> 4. Subsequent runs will be fully offline
>
> We recommend tested models from `mlx-community/*` but cannot guarantee third-party code behavior.

---

## UX Changes

### Command Prominence

| Command | 2.0.4 Role | 2.0.5 Role |
|---------|------------|------------|
| `pull` | Primary download | Caching/convenience |
| `clone` | Secondary | **Primary** for managed workflows |
| `run Model` | Default | Legacy/quick testing |
| `run ./path` | Supported | **Recommended** |

### Documentation Shift

**Before:** "Download models with `mlxk pull`"

**After:** "For reproducible workflows, use `mlxk clone` to create managed workspaces"

### New Flags/Behavior

```bash
# Automatic workspace-local cache when path starts with ./
mlxk run ./models/whisper "transcribe"
# Internally: HF_HOME=./models/.hf_cache

# Explicit flag (optional, for cache models)
mlxk run Model --workspace-cache ./cache
```

---

## Relationship to ADR-018

ADR-018 defines workspace operations (clone, convert, push) and the workspace sentinel concept.

**ADR-022 extends this by:**
1. Making workspace the **primary** paradigm, not secondary
2. Adding workspace-local HF cache isolation
3. Defining security/offline guarantees
4. Driving UX changes (clone > pull)

**ADR-018 provides:** Infrastructure (sentinel, convert, workspace paths)
**ADR-022 provides:** Philosophy and user-facing paradigm shift

---

## Implementation Phases

### Phase 1: Workspace-Local Cache (2.0.5-beta.1)

**Goal:** Isolate runtime artifacts per workspace

**Changes:**
- `run ./path` sets `HF_HOME=<workspace>/.hf_cache` before loading
- `.hf_cache/` added to workspace structure
- `.hf_cache/` documented in workspace sentinel

**Files:**
- `mlxk2/core/runner/__init__.py` — HF_HOME redirect
- `mlxk2/core/vision_runner.py` — HF_HOME redirect
- `mlxk2/core/audio_runner.py` — HF_HOME redirect
- `mlxk2/operations/workspace.py` — .hf_cache handling

**Tests:** ~10-15 new tests

### Phase 2: Testsuite Migration (2.0.5-beta.2)

**Goal:** Tests support both paradigms

**Changes:**
- Fixtures for `cached_model` and `workspace_model`
- E2E tests for workspace isolation
- Tests for .hf_cache artifact capture

**Effort:** High (many fixtures affected)

### Phase 3: Documentation & UX (2.0.5-beta.3)

**Goal:** Shift user guidance to workspace-first

**Changes:**
- README: clone as primary workflow
- SECURITY.md: revised guarantees
- Tutorials: workspace-based examples
- `mlxk pull` help text: "For caching; use clone for managed workflows"

### Phase 4: SECURITY.md Update (2.0.5 stable)

**Goal:** Honest, defensible security claims

**Changes:**
- Clear separation: mlx-knife behavior vs upstream behavior
- Workspace-based offline workflow documented
- Disclaimer for third-party library behavior

---

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Breaking change for pull-centric users | pull still works, just de-emphasized |
| Testsuite complexity | Phased migration, both modes supported |
| Disk space (workspace + cache duplication) | Document, user choice |
| User confusion (two paradigms) | Clear docs, gradual deprecation of pull-first |

---

## Open Questions

1. **Should `pull` warn about workspace-first?** → No, just document
2. **Auto-create .hf_cache/?** → Yes, automatic
3. **Workspace health include .hf_cache scan?** → Yes, with `--verbose`
4. **Archive format?** → Deferred to 2.0.6+

---

## MLXK_WORKSPACE_HOME

Single workspace path (like `HF_HOME`):

```bash
export MLXK_WORKSPACE_HOME=~/mlx-models

mlxk clone whisper-large-v3
# → ~/mlx-models/whisper-large-v3/

mlxk list
# Shows: HF cache + MLXK_WORKSPACE_HOME

mlxk run whisper-large-v3
# Search order: 1. MLXK_WORKSPACE_HOME  2. HF cache
```

**Implementation:**
- `mlxk2/core/cache.py` — new `get_workspace_home()` function
- `mlxk2/operations/clone.py` — default target if no path given
- `mlxk2/operations/list.py` — include MLXK_WORKSPACE_HOME in scan
- `mlxk2/core/model_resolution.py` — search MLXK_WORKSPACE_HOME first

**Future:** `MLXK_MODEL_PATH` for multi-path search (2.0.6+)

---

## UX Details

### list: Source Column

```
Name              | Source | Size   | Type
whisper-large-v3  | ws     | 400MB  | audio
phi-3-mini        | cache  | 2.1GB  | chat
```

### list --full-paths

```
Name                              | Source | Size
/Users/.../models/whisper-large-v3| ws     | 400MB
```

### list --origin

```
Name              | Source | Origin                         | Size
whisper-large-v3  | ws     | mlx-community/whisper-large-v3 | 400MB
```

### show: Workspace Metadata

```
Model: whisper-large-v3
Framework: MLX
...
Workspace:
  Source: mlx-community/whisper-large-v3-mlx
  Operation: clone
  Created: 2026-02-08
  Content Hash: sha256:a1b2c3...
  Modified: no
```

---

## JSON API Schema 0.2.0

New fields in `modelObject`:

```json
{
  "name": "whisper-large-v3",
  "source": "workspace",
  "origin": "mlx-community/whisper-large-v3-mlx",
  "content_hash": "sha256:a1b2c3...",
  "hash_modified": false,
  "cached": false
}
```

| Field | Type | Description |
|-------|------|-------------|
| `source` | `"cache" \| "workspace"` | Where model lives |
| `origin` | `string \| null` | HF origin (from sentinel) |
| `content_hash` | `string \| null` | SHA256 of workspace content |
| `hash_modified` | `boolean` | True if hash changed since clone/convert |

**Breaking Changes:** None (additive)

---

## Content Hash

### Exclude List

```python
HASH_EXCLUDE = [
    ".mlxk_workspace.json",  # contains the hash itself
    ".hf_cache/",            # runtime artifacts
    ".DS_Store",
    ".git/",
    "__pycache__/",
    "*.log",
    "*.tmp",
]
```

### Algorithm

```python
def compute_workspace_hash(workspace_path: Path) -> str:
    hasher = hashlib.sha256()
    for file in sorted(workspace_path.rglob("*")):
        if should_exclude(file):
            continue
        if file.is_file():
            # Hash: relative path + content
            hasher.update(file.relative_to(workspace_path).encode())
            hasher.update(file.read_bytes())
    return f"sha256:{hasher.hexdigest()}"
```

### When Computed

- After `clone` (before declaring success)
- After `convert` (before declaring success)
- Stored in `.mlxk_workspace.json`

---

## Sentinel Schema (Extended)

```json
{
  "mlxk_version": "2.0.5",
  "created_at": "2026-02-08T10:30:00Z",
  "source_repo": "mlx-community/whisper-large-v3-mlx",
  "source_revision": "abc123def456",
  "managed": true,
  "operation": "clone",
  "content_hash": "sha256:a1b2c3d4e5f6...",
  "hash_computed_at": "2026-02-08T10:30:05Z",
  "hash_excludes": [".mlxk_workspace.json", ".hf_cache/"]
}
```

---

## Code-Findings (Session 2026-02-08)

### Bug 1: PyTorch Warning bei Workspace-Pfaden

**Symptom:** `mlxk list ./path` zeigt "PyTorch was not found" Warnung

**Root Cause:** `vision_runtime_compatibility()` (common.py:456) importiert `transformers` als erstes bei healthy Vision-Modellen. Bei HF-Cache wird `mlx_lm` vorher importiert (unterdrückt Warnung).

**Betroffene Befehle:** `list`, `show` (nicht `run`, `health`)

**Fix:**
```python
# ALT (common.py:456)
import transformers
tf_version = getattr(transformers, "__version__", "0.0.0")

# NEU
from importlib.metadata import version
tf_version = version("transformers")
```

### Bug 2: Clone ohne HF_HOME

**Symptom:** `clone` schlägt fehl wenn `HF_HOME=""` (unset)

**Root Cause:** `_validate_same_volume()` (clone.py:100) prüft `volume(workspace) == volume(HF_HOME)`. Aber temp_cache wird sowieso auf Workspace-Volume erstellt (Zeile 439).

**Fix:** Check entfernen — ist überflüssig.

### Bug 3: Empty HF_HOME String

**Symptom:** `get_current_cache_root()` gibt `Path("")` → `PosixPath(".")` zurück

**Root Cause:** `os.environ.get("HF_HOME", DEFAULT)` gibt `""` zurück wenn Key existiert aber leer ist.

**Fix:**
```python
def get_current_cache_root() -> Path:
    hf_home = os.environ.get("HF_HOME")
    if not hf_home:  # None or ""
        return DEFAULT_CACHE_ROOT
    return Path(hf_home)
```

---

## References

- ADR-018: Convert Operation (workspace infrastructure)
- SECURITY.md (current promises)
- VibeVoice tokenizer issue (docs/ISSUES/vibevoice-missing-tokenizer.md)
- HuggingFace Hub caching behavior
