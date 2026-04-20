# ADR-025: content_hash v2 Algorithm + Sentinel Migration

**Status:** Accepted
**Created:** 2026-04-19
**Drafted:** 2026-04-20
**Accepted:** 2026-04-20
**Related:** ADR-018 (Convert Operation), ADR-022 (Workspace-First Paradigm), Issue [#52](https://github.com/mzau/mlx-knife/issues/52) (label `bug`, `2.0.x-branch`)
**Target:** 2.0.6

---

## Context

### The coverage gap (Issue #52)

`compute_workspace_hash()` in `mlxk2/operations/workspace.py:315-378`
(v1 algorithm) covers a narrow set of files:

- `config.json` — full content
- `*.safetensors` (root only) — filename + size + first 4 KB sample
  (the 4 KB sample differentiates quantization variants since
  2.0.5-beta.2)
- Tokenizer-related patterns (`tokenizer*.json`, `vocab*.json`,
  `merges.txt`) — filename + size only, **no content**

v1 silently ignores `model.safetensors.index.json`,
`processor_config.json`, `preprocessor_config.json`,
`chat_template.json` / `.jinja`, `generation_config.json`, tokenizer
*content*, `.gitattributes`, `README.md`, `modeling_*.py`, other
custom Python files, and every subdirectory.

**Primary consequence.** `mlxk convert --repair-index` rewrites
`model.safetensors.index.json` but produces a target with an
identical `content_hash` to its source — `Clean: ✓` is not a
trustworthy integrity indicator. Manual repairs (adding
`spatial_merge_size: 2` to `processor_config.json` for the documented
`mistral3` workflow; the 2.0.6-planned `--repair-config`) share the
same blindness: they modify semantically critical files that v1
never reads.

### Listing performance (fleet-scale)

mlx-knife is a fleet-management tool. Users accumulate dozens of
models. A naïve v2 that simply broadens v1's read surface — hashing
full content of every config file, every tokenizer file, every
markdown file on every `mlxk ls` — turns a common command into an
I/O stall. v2 must therefore include a stat-based cache that skips
file content entirely in the common case.

### Transport robustness

Users move workspaces: Finder copy to an external drive, `cp -R`
onto a backup volume, cross-OS moves via an SMB share. Each of these
rewrites mtimes without touching file content. A v2 that naïvely
stat-compared would turn every transport into a `Clean: ✗` flag. The
algorithm must be transport-invariant at the `content_hash` level,
and must self-heal stat differences that correspond to no content
change.

### Why now

Workspace adoption is still narrow in 2.0.5. The one-time transition
where old workspaces show `clean: None` until `--recalc-hash` is
therefore low-cost right now. Later waves of workspace adoption
would make that transition more visible. 2.0.6 is the release
window.

---

## Decision

### 1. File classification — what gets hashed

Include-by-default. Every file inside the workspace is hashed
unless it matches the exclude list.

The table below is **examples per class, not a closed enumeration**.
The authoritative mechanism is the catch-all row: any file not
matching an exclude pattern is hashed according to its class strategy
(safetensors vs. everything-else).

| File class | Examples (non-exhaustive) | Per-file hash strategy |
|---|---|---|
| Safetensors | `*.safetensors` | parse 8-byte little-endian length prefix; hash the JSON metadata header bytes only (no tensor data) |
| JSON configs | `config.json`, `*_config.json`, `generation_config.json`, `preprocessor_config.json`, `processor_config.json`, `chat_template.json` | full content |
| Tokenizer | `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt` | full content |
| Jinja / text templates | `chat_template.jinja`, `*.jinja`, `*.txt` | full content |
| Markdown / docs | `README.md`, `MODEL_CARD.md`, `LICENSE`, `NOTICE` | full content |
| Custom code | `modeling_*.py`, other `.py` | full content |
| Upstream git metadata | `.gitattributes` | full content |
| Catch-all (authoritative) | every other file not in the exclude list | full content, subject to the size cap below |

**Known semantically relevant files that enter via the catch-all**
(no special-case code needed, listed only for visibility):
`added_tokens.json`, `special_tokens_map.json`,
`model.safetensors.index.json`,
`config_sentence_transformers.json`.

**Rationale.** Workspaces are active-work territory; hash changes
are expected when the user edits anything. The include-by-default
principle also makes the algorithm robust against future upstream
additions — a new config file introduced by mlx-vlm or mlx-audio
automatically enters the hash without an algorithm revision.

**Catch-all size cap.** Non-safetensors files larger than
`CATCHALL_FULL_READ_CAP` (initial value: 1 GiB) fall back to a
**stat-only strategy**: `sha256(filename || 0x00 || size)`, no file
read. The cap prevents a single large non-model file (video sample,
dataset dump, accidentally-placed backup) from turning `mlxk ls`
into a multi-second I/O stall. The cap value is part of the v2
algorithm — changing it requires a `hash_algorithm` bump per §9.

**Safetensors header edge cases.**

- Zero-length header prefix (`length == 0`) is treated as a corrupt
  file: the per-file hash is set to a sentinel value and the
  workspace reports `clean: None` with hint *"safetensors file
  `<path>` has empty header — run `--recalc-hash` after verifying
  the download"*. This prevents two distinct corrupt files from
  colliding on an empty-header hash.
- Header length > 100 MiB is treated as a malformed prefix:
  identical handling (`clean: None` + hint).

**Symlink handling.** Symlinks exist in practice inside a workspace:
the workspace-internal `.hf_cache/` uses HuggingFace's native
`snapshots/ → blobs/` symlink layout, and model loaders may populate
runtime-downloaded fragments into that cache. A blanket "reject all
symlinks" policy would break legitimate use. Policy:

- The tree walk runs with `followlinks=False` (explicit; Python
  default). We never iterate *through* a symlink.
- When the walk encounters a symlinked file, resolve its target and
  compare against the workspace root:
  - **Target inside workspace root:** hash the link as a
    path-fingerprint — `sha256("SYMLINK" || 0x00 ||
    posix_relpath_from_workspace)`. Do not dereference. If the
    target is itself included in the walk (not excluded), it is
    hashed separately on its own entry; if the target is excluded
    (e.g., under `.hf_cache/`), only the link fingerprint is
    captured.
  - **Target outside workspace root, or broken:** refuse.
    `clean: None` with hint *"workspace contains symlink pointing
    outside workspace root: `<path>` → `<target>`; refuse to
    follow"*.

The outside-workspace reject is security-critical: a malicious HF
repo could ship a symlink pointing at `/etc/passwd`, `/dev/urandom`,
or a named pipe; naive dereferencing would leak host content into
the hash, hang the hasher, or read sensitive files. The workspace-
scope bound closes this without breaking the legitimate cache case.

**Safetensors header-only.** safetensors files have a known
parse-able layout: 8-byte little-endian length prefix → JSON block
with tensor names, shapes, and dtypes → tensor data. Hashing the
header captures every structural change (new tensor, dtype change,
shape change, layout change) without reading tensor bytes. A 50 GB
file hashes in milliseconds. The upstream snapshot pin
(`source_revision`, already stored in the sentinel) is the real
integrity guarantee against byte-level corruption of tensor data;
v2 `content_hash` is a post-download modification detector, not a
cryptographic tamper seal.

**Hash primitive.** sha256 throughout. No new dependency. The
per-class sampling keeps end-to-end throughput well under a second
even for multi-GB workspaces.

**Path normalization.** Paths are stored relative to the workspace
root, using POSIX separators, byte-sorted, with UTF-8 NFC
normalization of filenames.

**Recursion.** On. Subdirectories are walked. Exclude patterns
apply to directory names and file patterns uniformly.

### 2. Exclude list

The default exclude list is narrow and grouped by purpose:

- *mlxk runtime:* `.hf_cache/`, `.mlxk_workspace.json`
- *Dev / VCS tooling:* `.git/`, `.github/`, `__pycache__/`
- *User IDE / dev-tool cruft* (produced when users work inside the
  workspace — editing configs, debugging, running tests):
  `.vscode/`, `.idea/`, `.ipynb_checkpoints/`, `.mypy_cache/`,
  `.pytest_cache/`, `.ruff_cache/`, `.tox/`, `node_modules/`,
  `.venv/`, `venv/`
- *Generic temp / logs:* `*.lock`, `*.tmp`, `*.log`, `*.bak`
- *macOS file-manager cruft:* `.DS_Store`, `._*` (AppleDouble),
  `.Spotlight-V100/`, `.Trashes/`, `.fseventsd/`,
  `.TemporaryItems/`, `.apdisk`, `Icon\r`
- *Windows file-manager cruft:* `Thumbs.db`, `desktop.ini`,
  `$RECYCLE.BIN/`, `System Volume Information/`
- *Editor backups:* `*~`, `.*.swp`, `.*.swo`

The OS / file-manager categories exist specifically for transport
robustness: Finder, Windows Explorer on an SMB share, and
cross-filesystem moves inject these involuntarily. The IDE / dev-tool
category exists for the same reason applied to active workspace use
— users who debug a model inside its workspace must not flip
`Clean: ✗` the moment they open VS Code or run `pytest`. Hashing any
of these categories would produce spurious `Clean: ✗` when no model
content has changed.

`.gitattributes` is **included** (hashed). HuggingFace repos are
git-based; `.gitattributes` ships with the snapshot download as
git-LFS configuration. Including it follows the include-by-default
principle, and the cost is trivial (the file is tiny and rarely
changes).

**Placement: the exclude list is part of the recipe and lives in the
sentinel** (see §3 `exclude_patterns` field, §4 architectural
principle). The code holds a default list that seeds the first
compute; at write time the effective list is **frozen into the
sentinel** for that workspace. Subsequent reads — clean-check,
self-heal, future `hash_algorithm` readers — use the sentinel's
stored list, not the current code's default. Consequence: when a
later mlx-knife release ships an expanded default exclude list,
existing workspaces keep their frozen list until the user runs
`--recalc-hash`. There is no drift-induced false positive on
ordinary mlx-knife upgrades.

### 3. Sentinel schema (v2 additions)

```json
{
  "mlxk_version": "2.0.6",
  "hash_algorithm": "v2",
  "content_hash": "sha256:aabb...",
  "hash_modified": "2026-04-20T10:30:00Z",
  "exclude_patterns": [
    ".hf_cache/", ".mlxk_workspace.json",
    ".git/", ".github/", "__pycache__/",
    ".vscode/", ".idea/", ".ipynb_checkpoints/",
    ".mypy_cache/", ".pytest_cache/", ".ruff_cache/", ".tox/",
    "node_modules/", ".venv/", "venv/",
    "*.lock", "*.tmp", "*.log", "*.bak",
    ".DS_Store", "._*", ".Spotlight-V100/", ".Trashes/",
    ".fseventsd/", ".TemporaryItems/", ".apdisk", "Icon\r",
    "Thumbs.db", "desktop.ini",
    "$RECYCLE.BIN/", "System Volume Information/",
    "*~", ".*.swp", ".*.swo"
  ],
  "file_index": [
    {
      "path": "config.json",
      "size": 1234,
      "mtime_ns": 1714000000500000000,
      "sha": "sha256:def..."
    },
    {
      "path": "model-00001-of-06.safetensors",
      "size": 4000000000,
      "mtime_ns": 1714000000500000000,
      "sha": "sha256:ghi..."
    }
  ],
  // unchanged existing fields:
  "created_at": "...", "source_repo": "...", "source_revision": "...",
  "managed": true, "operation": "clone"
}
```

Aggregate formula:

```
content_hash = sha256( concat( sorted( path || 0x00 || sha )
                               over file_index entries ) )
```

The NUL byte (`0x00`) separates `path` from `sha` to eliminate
ambiguity at concatenation boundaries. Path is part of the aggregate
so renames flip `content_hash` — a renamed weight file is
semantically a different workspace even when byte content is
unchanged.

**Transport invariance.** `content_hash` and per-file `sha` depend
only on file content (safetensors header bytes, other files' full
content) and relative path. They do not depend on `mtime_ns`, the
workspace's absolute path on the host, or host-filesystem
peculiarities. `cp`, Finder, and cross-OS moves never change the
aggregate digest.

**Path constraints (security).** Every `path` in `file_index` MUST be:

- relative (no leading `/`),
- POSIX-separated (`/`, never `\`),
- free of `..` components,
- free of absolute drive prefixes (`C:\`, `/Users/...`),
- free of NUL bytes.

Readers validate each path **before** any filesystem operation — in
particular before the self-heal read (`(workspace_root / path).open`,
§6 step 5). A violation is treated as a corrupt/tampered sentinel:
return `clean: None` with hint *"workspace sentinel has invalid file
path entry — run `--recalc-hash` to rebuild"*. The reader never opens
or stats the offending path. Rationale: without validation, a
tampered sentinel with `path = "../../etc/passwd"` would cause the
self-heal step to read outside the workspace on every `mlxk ls`,
turning a passive status command into an information-disclosure
vector.

### 4. Portable recipe (architectural principle)

The recipe for reproducing a clean-check is the pair
(`file_index`, `exclude_patterns`). Together they describe
everything a reader needs to know about *which files were covered
and how*, without relying on the code that produced them.

- `file_index` is the self-describing record of **what was included**
  and with which per-file sha.
- `exclude_patterns` is the self-describing record of **what was
  legitimately ignored**. Without it, the "is this walk entry
  dirty-new or acceptable cruft?" decision would require consulting
  the current code's exclude list — and that list drifts between
  releases.

Without this pair in the sentinel, the knowledge of *which files v2
covered, which it ignored, and how each was hashed* would live only
in the v2 algorithm code. Every future `hash_algorithm` bump would
force one of two unacceptable options: either preserve historical
algorithm implementations in the codebase forever (dual-, triple-,
N-algo read logic), or force a full `--recalc-hash` migration on
every user at every bump.

With the full recipe inside the sentinel, a future v3 reader can
verify a v2 workspace by replaying the stored exclude patterns
during the walk and comparing per-file shas against current file
content — no v2 walk/classification code needed. Historical
algorithm implementations can be removed cleanly once no active
workspaces reference the old version.

If a future version introduces a new per-file *strategy* (for
example, `.safetensors` switching from header-only to full-content),
the strategy tag migrates into each entry additively:
`{path, size, mtime_ns, sha, strategy}`. v2 leaves strategy
implicit because all v2 entries share the v2 default set. v3
readers extend each entry with `strategy` only for entries that use
the new behavior.

### 5. Cache placement (architectural principle)

`file_index` lives inside the sentinel. mlx-knife holds **no state
outside the models it manages** — no `~/.mlxk/`, no user-level
database, no XDG cache directory. The workspace sentinel is the
**sole** exception to this rule.

Consequences:

- State travels with the workspace: transport stays coherent across
  machines and volumes.
- Deleting a workspace deletes its cache automatically — no
  orphans.
- Cache-backed (non-workspace) models carry nothing on top of the
  HuggingFace cache; their clean state, where applicable, is
  derived on-the-fly.

### 6. Clean-check flow (hot path: every `mlxk ls` / `mlxk show`)

1. Read sentinel.
2. If `hash_algorithm` is missing or `!= "v2"` → return
   `clean: None` with the actionable hint *"run
   `mlxk show <name> --recalc-hash` to upgrade"*. STOP — no tree
   walk.
3. Validate every `path` entry in `file_index` per §3 path
   constraints. Any violation → `clean: None` with corrupt-sentinel
   hint. STOP — no tree walk, no file open.
4. Tree walk, filtered by the sentinel's stored `exclude_patterns`
   (not the current code defaults); `stat()` each file, no reads.
   Symlink entries are handled per §1 symlink policy (path
   fingerprint or reject).
5. Path-set comparison:
   - File in walk but not in `file_index` → dirty (new file).
   - File in `file_index` but missing from walk → dirty
     (removed file).
6. For each matched file, inspect the stat tuple:
   - `size` differs → dirty (shortcut; no hashing needed).
   - `size` matches and `mtime_ns` matches → OK (fast path).
   - `size` matches and `mtime_ns` differs → **self-heal**: revalidate
     the path against §3 constraints, then hash the file, compare to
     the stored `sha`. If the sha matches → update `mtime_ns` in the
     in-memory `file_index` and mark OK. If it differs → dirty.
7. After the walk: any file marked dirty → `clean: False`. All
   files OK → `clean: True`. If any self-heals occurred, silently
   persist the updated `file_index` back to the sentinel
   (best-effort; swallow `OSError` on read-only filesystems —
   in-memory `clean: True` is still returned).

Performance targets:

- Common case (no changes): 50 workspaces × ~10 files ≈ 500
  `stat()` calls ≈ 50 ms total.
- Post-transport (all mtimes shifted, content unchanged): one slow
  listing (full re-hash, ~1 s total), persisted; subsequent
  listings return to the fast path.
- Real modification: dirty, as intended.

### 7. First-compute flow (clone, convert, `--recalc-hash`)

1. Walk the filtered tree.
2. Hash each file according to its class strategy.
3. Build sorted `file_index` entries `{path, size, mtime_ns, sha}`.
4. Compute the aggregate `content_hash` per the formula in §3.
5. Write the sentinel.

Budget: a 50 GB workspace hashes in ≈ 1 s (safetensors header-only,
other files small). No tensor-body reads anywhere in the algorithm.

### 8. Migration — strict cut-over

v1 sentinels do not carry a `hash_algorithm` field. v2 readers
detect the absence and return `clean: None` plus a one-line hint
pointing at `--recalc-hash`.

- Workspaces remain fully usable during the transition. Only the
  `Clean` indicator is affected.
- `mlxk show <name> --recalc-hash` performs the upgrade in-place:
  a full first-compute, a new sentinel with `hash_algorithm: "v2"`
  + `file_index`, and `Clean: ✓` on the next listing.
- No dual-algo read logic. A sentinel is either v2 (trustable) or
  not-v2 (show `clean: None` and move on).

The 2.0.6 release window minimizes migration surface. Workspace
adoption is still narrow, so the `clean: None` transition lands on
few users.

### 9. Forward compatibility (future v3+)

The `hash_algorithm` field is the versioning contract for all
future changes — to the algorithm, the file set, the per-entry
structure, or the per-class hash strategies.

- A future v3 bumps `hash_algorithm: "v3"`. v2 readers see the
  unknown algorithm and return `clean: None` with the same
  `--recalc-hash` hint (exact same pattern as v1 → v2). No
  misreading, no silent drift, no cross-version miscompare.
- v3 readers know both v2 and v3, and handle each.
- Sentinel JSON itself is forward-compatible via standard hygiene:
  unknown top-level fields must be ignored by readers.
- `file_index` entries can gain optional fields (such as a future
  per-entry `strategy` or `mode`) without breaking v2 readers,
  provided `hash_algorithm` has not bumped.

**Exclude-list evolution does not require a bump.** Because
`exclude_patterns` is frozen into the sentinel per workspace (§2,
§4), shipping an expanded default exclude list in a later mlx-knife
release affects only *new* workspaces. Existing workspaces keep
their stored list until the user opts in via `--recalc-hash`. No
`hash_algorithm` change is needed for routine exclude-list tuning.

**Per-class hash strategies do require a bump.** Changing how a file
class is hashed — for example, shifting safetensors from
header-only to full-content, or altering `CATCHALL_FULL_READ_CAP` —
changes every per-file `sha` produced by the algorithm and must
therefore follow the strict cut-over in §8.

The migration pattern *(strict cut-over + `--recalc-hash`)* scales
to any future version bump. No dual-algo complexity, no ambiguity,
no "is this v2 or v3?" guessing.

---

## Threat model

content_hash v2 is a **post-download tamper detector** for the files
mlx-knife manages on behalf of the user. Stating the threat model
explicitly keeps the "everything full-content" design readable — it
is a security property, not just an integrity nicety.

### In scope — attacks v2 detects

Files inside a workspace that mlx-* / transformers subsequently
**loads and, in common configurations, executes**. If a local process
or remote sync quietly swaps one of these post-clone, v2 flips
`Clean: ✗` on the next `mlxk ls`:

- `config.json` with `auto_map` / `architectures` entries +
  `trust_remote_code` → Python exec of `modeling_*.py`.
- `chat_template.jinja` / `chat_template.json` → Jinja rendering;
  prompt-injection / sandbox-escape vectors live here.
- `processor_config.json`, `preprocessor_config.json` → custom
  processor class references, image/audio preprocessing hooks.
- `tokenizer.json` (content) → pattern rules; pathological regex can
  cause ReDoS during tokenization.
- `modeling_*.py`, any `.py` under the workspace → direct exec
  vector when `trust_remote_code=True`.

The v1 algorithm missed the Jinja templates, processor configs, and
tokenizer *content* entirely (it only hashed tokenizer filename +
size). These are the precise files most interesting to an attacker
who wants code execution on a user who runs `mlxk run` or loads the
model via transformers. Closing this gap is the primary driver for
"everything full-content" in §1 — not aesthetic thoroughness.

### Out of scope — what v2 does not protect against

- **Pre-download tampering** (malicious HuggingFace repo). v2 hashes
  what was downloaded; if the upstream ships poisoned files, the
  hash faithfully captures the poison as "clean". Defense lives in
  `source_revision` pinning + HF-upstream trust (signed commits,
  LFS hash verification), which are upstream concerns.
- **Local attacker with filesystem write + ability to run
  `--recalc-hash`.** Such an attacker refreezes the hash after
  tampering; `Clean: ✓` will lie. No content hash can defend against
  this — content hashes detect drift from a baseline, they do not
  authenticate the baseline. Defense lives at the OS permissions
  layer.
- **sha256 collisions.** Not computationally feasible at current
  attack economics; not considered.

### Mlx-knife's own exposure during hashing

The workspace walker reads file *bytes* and feeds them to sha256.
There is no `json.load`, no Jinja render, no `eval`, no dynamic
import of anything inside the workspace. Hashing is
side-effect-free. The dangerous loaders — transformers,
mlx-lm / mlx-vlm / mlx-audio — run only when the user explicitly
asks for generation (`mlxk run`, server request); they are outside
the content_hash code path.

### Safetensors tensor bytes — a deliberate non-coverage

Safetensors files are hashed header-only (§1). Tensor data is not
read. An attacker who modifies tensor bytes — for example, a subtle
weight perturbation to induce targeted misclassification — is **not
detected** by v2 content_hash.

This is a deliberate trade, not an oversight:

- Tensor-byte integrity is what HuggingFace's SHA-of-blob plus the
  snapshot pin (`source_revision`) already guarantee at download
  time. Recomputing it locally on every `mlxk ls` would add
  minute-scale I/O to a status command for no additional trust.
- Post-download tensor tampering requires local filesystem write,
  which already falls into the out-of-scope "local attacker"
  category above.
- The header hash still catches *structural* tampering — added or
  removed tensors, reshaping, dtype changes. A post-download swap
  that replaces weights with structurally different weights is
  caught.

If a future threat model ever demands tensor-byte coverage, it
becomes a new per-file `strategy: "full-content"` for safetensors
entries and bumps `hash_algorithm` per §9.

## Considered alternatives

- **Stay with v1.** Rejected. `Clean: ✓` becomes increasingly
  misleading as `--repair-index`, `--repair-config` (2.0.6), and
  manual repair workflows produce more hash-invisible changes.
- **Dual-algo read logic (v1 fallback).** Rejected. Trusting v1
  hashes in legacy sentinels means `Clean: ✓` has two different
  meanings — "clean per narrow v1" vs. "clean per comprehensive
  v2". Not acceptable for an integrity indicator.
- **Aggressive full-content hashing of everything, no cache.**
  Rejected. Works until the user has 30+ workspaces, at which
  point `mlxk ls` becomes an I/O stall.
- **Aggregate `stat_digest` instead of per-file `file_index`.**
  Considered. Simpler implementation and ~2 KB smaller sentinel,
  but *loses the portable-recipe property*: v3 could not verify v2
  workspaces without preserving v2 code indefinitely. Rejected.
- **Central cache at `~/.mlxk/`.** Rejected. Violates the "no
  state outside managed models" architectural principle. Transport
  breaks. Orphans accumulate. Keying workspaces back to the cache
  requires a path-or-hash database — which is the kind of state
  the principle exists to prevent.
- **xxh64 / BLAKE3 as hash primitive.** Considered. Faster, but
  adds a new runtime dependency for no practical benefit:
  safetensors are header-only (microseconds) and other files are
  small. sha256 is never the bottleneck.
- **Fast-path "dirty on any stat mismatch" (no self-heal).**
  Considered. Simpler, but fails transport robustness: every
  `cp -R` / Finder copy / cross-OS move would flip `Clean: ✗` and
  require the user to `--recalc-hash` explicitly. Rejected in
  favor of git-style self-heal.

---

## Consequences

### Positive

- `Clean: ✓` becomes trustworthy across the full spectrum of
  repairs and edits that touch workspace files.
- `mlxk ls` stays fast at fleet scale (~50 ms for 50 unchanged
  workspaces) because the hot path is stat-only.
- Transport across volumes and operating systems preserves
  `Clean: ✓` without user intervention, via self-heal.
- Future algorithm evolution is unblocked: historical v2 code can
  be removed when no active v2 workspace remains, because
  `file_index` carries the recipe.
- Uninstalling mlx-knife still leaves no cruft outside the model
  directories — the "no state outside models" invariant is
  preserved.

### Trade-offs

- Existing 2.0.5 workspaces show `clean: None` in 2.0.6 until the
  user runs `mlxk show <name> --recalc-hash`. The transition is
  functionally non-breaking but visibly noisy to fleet owners for
  one release. Mitigation: the 2.0.6 CHANGELOG and release notes
  describe the one-liner upgrade.
- `file_index` adds ~2–4 KB per sentinel (≈ 200 B per entry ×
  ~10–20 files for a typical mlx model). For a 100-workspace
  fleet this is ~400 KB total — well below any practical budget.
- The clean-check flow has one more code path (self-heal) than the
  obvious "stat-diff → dirty" minimum. Implementation complexity
  justified by transport robustness.

---

## Implementation notes

Phase 1 of implementation is this ADR document itself (Draft → user
review → Accepted).

Phase 2 is the code. Anchor points (2.0.5 baseline):

| File | Role |
|---|---|
| `mlxk2/operations/workspace.py:315-378` | v1 `compute_workspace_hash()` — deleted, replaced by v2 |
| `mlxk2/operations/workspace.py:381-436` | `update_workspace_hash()` — extended to write v2 format |
| `mlxk2/operations/workspace.py:439-478` | `is_workspace_clean()` — interior rewrite, signature preserved |
| `mlxk2/operations/common.py:678-679` | unchanged (consumes `clean` field) |
| `mlxk2/output/human.py:200-209` | unchanged (renders `Clean: ✓ / ✗ / —`) |
| `mlxk2/cli.py:299` + `mlxk2/operations/show.py:117,167` | unchanged (`--recalc-hash` wiring already in place) |

New helpers in `mlxk2/operations/workspace.py`:

- `_parse_safetensors_header(path) -> bytes | None` — read 8-byte
  little-endian length prefix, read that many header bytes, return;
  return `None` on zero-length prefix or prefix > 100 MiB (caller
  treats as corrupt, reports `clean: None` per §1 edge cases).
- `_validate_sentinel_path(path: str) -> str` — raise
  `CorruptSentinelError` on `..` components, absolute paths, drive
  prefixes, NUL bytes, or non-POSIX separators. Called before every
  `(workspace_root / path).open()` and before the self-heal read in
  §6 step 6.
- `_classify_symlink(entry, workspace_root) -> SymlinkOutcome` —
  returns `(kind="inside", link_fingerprint=...)` for in-workspace
  targets (§1 symlink policy), or `(kind="reject", reason=...)` for
  out-of-workspace / broken targets.
- `_stat_check_with_self_heal(workspace, file_index) -> (clean,
  healed, updated_file_index)` — stat-walk, compare, self-heal
  where applicable. Applies `_validate_sentinel_path` to every entry
  before any filesystem access.
- `compute_workspace_hash_v2(path, exclude_patterns) -> (aggregate_hash,
  file_index)` — the first-compute path. Takes the exclude list as
  a parameter so first-compute can be driven from either the code
  defaults (clone/convert) or the sentinel-stored list
  (`--recalc-hash` on an existing v2 workspace, preserving that
  workspace's frozen list).

Module-level constants:

- `CATCHALL_FULL_READ_CAP = 1024 * 1024 * 1024` (1 GiB). Non-safetensors
  files above this size fall back to stat-only strategy per §1. The
  value is part of the v2 algorithm; changes require a
  `hash_algorithm` bump (§9).
- `SAFETENSORS_HEADER_MAX = 100 * 1024 * 1024` (100 MiB). Hard cap on
  the length prefix; above this the file is treated as corrupt.
- `DEFAULT_EXCLUDE_PATTERNS` — the code-default list that seeds
  first-compute (§2). Reader logic does not consult this directly;
  it reads `exclude_patterns` from the sentinel.

Sentinel writers (`mlxk2/operations/clone.py`,
`mlxk2/operations/convert.py`, `update_workspace_hash()`) extended
to emit `hash_algorithm: "v2"` and `file_index`.

### Verification

- **Regression for Issue #52.** Integration test: clone a source,
  run `convert --repair-index`, assert source and target
  `content_hash` differ.
- **Unit coverage.** Per file class: safetensors header parse
  (including a multi-GB fixture — assert only header bytes are
  read), JSON configs, tokenizer content, chat-template variants,
  `model.safetensors.index.json`, `.gitattributes`, `README.md`.
- **Transport robustness.** `cp -R` a workspace to another
  directory; assert `Clean: ✓` after the first `mlxk ls`
  (self-heal); assert sentinel `file_index` mtimes match
  post-transport stat values (silent write-back).
- **Listing performance.** `mlxk ls` over 50 unchanged workspaces
  completes in < 500 ms end-to-end.
- **First-hash cost.** Clone or convert of a ≥ 20 GB workspace
  adds ≤ 2 s over current v1 wall time.
- **Migration.** Pre-v2 sentinel → `clean: None` with hint; after
  `--recalc-hash` → v2 sentinel with `file_index` + `Clean: ✓`.
- **Hash stability.** Same workspace hashed twice yields identical
  aggregate digest and identical per-file shas. Transport does not
  change the aggregate digest.
- **Symlink — legitimate (inside workspace).** Workspace containing
  `foo.json → .hf_cache/<blob>` (target in excluded cache subtree).
  Expect `Clean: ✓`, symlink hashed as path fingerprint only, no
  read on the target.
- **Symlink — attack (outside workspace).** Workspace with
  `config.json → /etc/passwd` or `weights.safetensors → /dev/urandom`.
  Expect `clean: None` with refuse-to-follow hint. Assert via test
  instrumentation that zero bytes were read from the target path.
- **Path traversal in sentinel.** Hand-crafted sentinel with
  `file_index[].path = "../../etc/passwd"`. Expect `clean: None`
  with corrupt-sentinel hint. Assert no `open()` was called on the
  traversed path.
- **DoS — oversized catch-all.** Workspace containing a 100 GiB
  sparse random-data file (not `.safetensors`). Expect `mlxk ls`
  under 500 ms end-to-end (stat-only strategy engages via
  `CATCHALL_FULL_READ_CAP`); assert bytes-read counter is 0 for
  that file.
- **Dev-cruft transparency.** Workspace with `.vscode/settings.json`,
  `.pytest_cache/v/cache/`, `.ruff_cache/`. After `--recalc-hash`:
  `Clean: ✓`. Mutate a file under any of those paths: `Clean: ✓`
  remains. Mutate `config.json`: `Clean: ✗`.
- **Exclude-list drift resilience.** Write a v2 sentinel with a
  narrower `exclude_patterns` list than current code defaults; add a
  file that is excluded under current code but included under the
  sentinel's stored list. Expect the stored list to govern (file is
  walked and hashed).
