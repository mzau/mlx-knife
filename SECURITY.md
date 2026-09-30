# Security Policy

## Overview

MLX Knife is designed to run locally on your Apple Silicon Mac. It prioritizes user privacy and security by keeping all model execution local.

**Important distinction:** MLX Knife integrates upstream libraries (mlx-lm, mlx-vlm, mlx-audio, transformers) whose behavior is outside our direct control. This document describes what **mlx-knife itself** does; upstream libraries may behave differently.

## Security Model

### What MLX Knife Itself Does
- ✅ Runs models locally on your device
- ✅ Downloads models only from HuggingFace (via `pull`, `clone`)
- ✅ Uploads only when you explicitly run `push` (opt-in, requires credentials)
- ✅ API server binds to localhost by default
- ✅ No telemetry or usage tracking
- ✅ No automatic updates or phone-home features

### What MLX Knife Itself Doesn't Do
- ❌ No model outputs are logged or transmitted
- ❌ No user tracking or analytics
- ❌ mlx-knife code does not initiate network requests during `run`, `server`, or `show` —
  the one exception is `serve --embed-backend URL` (2.0.7, experimental), which forwards
  `/v1/embeddings` to the address you configure

### Third-Party Libraries

MLX Knife uses external libraries to load and run models. These libraries may download additional files when a model is first used - this is outside mlx-knife's control.

**What this means:**
- Downloading a model with `pull` does not guarantee fully offline use
- Some models may need additional downloads when first run

**For offline environments:**
Test each model while online before relying on offline use. Use `mlxk clone` to create a local workspace for better isolation.

## Reporting Security Vulnerabilities

If you discover a security vulnerability in MLX Knife, please help us address it responsibly:

### Do NOT:
- ❌ Open a public GitHub issue
- ❌ Post about it on social media
- ❌ Exploit it maliciously

### Please DO:
1. **Email**: Send details to broke@gmx.eu
2. **Or**: Create a private security advisory on GitHub
3. **Include**:
   - Affected version(s)
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

We will acknowledge receipt within 48 hours and work on a fix.

## Security Considerations

### Model Downloads (`mlxk pull`)
- **Source**: Models are downloaded from HuggingFace only
- **Verification**: HuggingFace provides checksums for file integrity
- **Content**: `pull` downloads a repository as published; mlx-knife makes no judgement about
  what it contains (see *Code in Model Directories*)

### Code in Model Directories

A model directory is not necessarily data. Besides weights and JSON configs it can contain
Python, in two ways: a config declares it — `model_file` in `config.json`, or `auto_map` in
`config.json` or another JSON config such as `tokenizer_config.json` or `processor_config.json`
— or `*.py` files are simply shipped without any config naming them.

- **Refused**: a model whose `config.json` declares `model_file`. With the mlx-lm version
  mlx-knife pins, loading such a model executes that file (CVE-2026-5843). mlx-knife refuses it
  before any backend is called, in `run`, `serve`, `embed`, `embed-serve` and
  `convert --quantize`; `list` and `show` report it as not runnable, with the reason. No flag,
  option or environment variable enables it.
- **Not prevented**: code declared through `auto_map`. For many model types, the pinned mlx-vlm
  and mlx-audio switch on transformers' remote-code loading themselves, and the pinned
  transformers offers no setting that overrides this from outside. On the vision and audio
  paths, mlx-knife cannot stop that code from running when such a model is loaded or converted.
- **What you can see beforehand**: the keys that declare code and any shipped `*.py` files are
  in the model directory before anything runs. `mlxk show <model> --files` lists the files,
  `--config` prints `config.json`; the other configs are plain JSON files next to it. An
  `auto_map` entry can also point to code in another repository — that code is not in the
  directory.
- **Where mlx-knife's part ends**: because mlx-knife cannot control what a backend library
  executes on those paths, it makes no statement about whether a particular model is safe to
  run, and it does not certify models. Whether to run a model is decided by whoever chooses it.

### API Server (`mlxk server`)
```bash
# Safe (localhost only):
mlxk server --port 8000

# CAUTION (network accessible):
mlxk server --host 0.0.0.0 --port 8000
```

**WARNING**: When using `--host 0.0.0.0`:
- The API becomes accessible from your network
- No built-in authentication or rate limiting
- Anyone on your network can use your models
- A client can make the server load any directory on the host that contains a `config.json`
  by naming it in the `model` field — by design ([ADR-022](docs/ADR/ADR-022-Workspace-First-Paradigm.md)).
  Loading a vision or audio model can run Python the checkpoint declares through `auto_map`
  ([#71](https://github.com/mzau/mlx-knife/issues/71), see *Code in Model Directories*); on a
  network, a remote client decides which of those directories is loaded
- Could potentially be exposed to the internet (check firewall!)

**Recommendations for network access:**
- Use a reverse proxy with authentication (nginx, Caddy)
- Implement firewall rules
- Never expose directly to the internet
- Consider VPN-only access

### Embeddings Backend (`mlxk embed-serve`, experimental)

2.0.7 adds a second listening surface, gated by `MLXK2_ENABLE_ALPHA_FEATURES=1`:

- `mlxk embed-serve` binds `127.0.0.1` by default. The `--host` warning above applies
  unchanged — it has no authentication or rate limiting either.
- `mlxk serve --embed-backend URL` turns the main server into an HTTP client of that
  URL: the bodies of `/v1/embeddings` requests are forwarded there verbatim. Point it
  only at a backend you control; neither hop is authenticated.

### Model Execution
- **Memory**: Large models can consume significant RAM/GPU memory
- **CPU/GPU**: Model execution can be resource-intensive
- **Disk**: Models are cached locally (can be multiple GB each)

### File System Access
- **Cache Location**: `~/.cache/huggingface/hub` or `$HF_HOME`
- **Permissions**: Standard user permissions apply
- **Cleanup**: Use `mlxk rm <model>` to safely remove models; avoid manual deletion in the user cache

### Hugging Face Cache Integrity
- Separate contexts: use an isolated test cache for automated tests; keep the user cache for manual/production work
- HF_HOME: set explicitly for user work if needed; tests should not override user HF_HOME by default
- Safe operations: reads (`list`, `health`, `show`) are always safe; coordinate writes (`pull`, `rm`) in maintenance windows
- Test safeguards: the test suite places a sentinel in the test cache and enforces deletion guards to prevent accidental user-cache modification

### Alpha Push (`mlxk2 push`)

The 2.0 alpha introduces an alpha upload capability. Treat it as opt‑in, with explicit user control.

#### Scope and defaults
- Upload‑only: pushes a specified local folder to a Hugging Face model repo via `huggingface_hub.upload_folder`.
- Requires `HF_TOKEN`; in alpha, `--private` is required to reduce accidental exposure.
- Default branch is `main` (overridable with `--branch`). No manifests or content validation yet.
- Honors default ignore patterns and merges project `.hfignore` when present (e.g., excludes `.git/`, `.venv/`, `__pycache__/`, `.DS_Store`).

#### Privacy and boundaries
- Only files under the path you provide are considered; push does not scan your global caches or home directory.
- No prompts, logs, or runtime telemetry are uploaded.
- No background activity: nothing is sent unless you invoke `mlxk2 push`.

#### Safety controls
- Preflight without network: `--check-only` analyzes the local folder for obvious issues (e.g., missing shards, LFS pointers).
- Plan without committing: `--dry-run` lists prospective adds/deletes vs remote (no upload performed).
- Use restricted tokens and test repos when validating; prefer `--private` and organization/user repos you control.

#### Risks and mitigations
- Risk: Accidental upload of sensitive files included in the folder.
  - Mitigate with a minimal, dedicated workspace, `.hfignore`, and `--check-only`/`--dry-run` before pushing.
- Risk: Pushing incomplete or corrupted weights.
  - Mitigate by reviewing `workspace_health` from `--check-only` and model card requirements before uploading.

#### User responsibility
**You are responsible for complying with Hugging Face Hub policies and applicable laws (e.g., copyright/licensing) for any uploaded content.** Review all content before uploading and ensure you have appropriate rights to distribute the models and associated files.

#### Network and logging
- Network egress targets only Hugging Face over HTTPS; no third‑party endpoints.
- In `--json` mode, hub logs may be captured in output for diagnostics; they are not transmitted elsewhere by MLX Knife.

## Security Best Practices

### For Users:
1. **Download models only from sources you trust**, and look at what a model directory
   contains before running it (see *Code in Model Directories*)
2. **Keep the API server local** unless you need network access
3. **Monitor disk usage** - models can be large
4. **Review model cards** on HuggingFace before downloading
5. **Keep Python dependencies updated**: `pip install --upgrade mlx-knife`

### For Contributors:
1. **Never commit secrets** (API keys, tokens)
2. **Validate all inputs** in new features
3. **Use secure defaults** (localhost binding, etc.)
4. **Document security implications** of new features
5. **Test for resource exhaustion** (memory, disk)

## Supported Versions

We provide security updates for the versions below. 2.0.8 fixes two vulnerabilities that
affect every earlier 2.0 release:

- mlx-knife up to and including 2.0.8b1, installed with mlx-lm 0.31.0 or later, executes the
  file a model's `config.json` names in `model_file` when it loads that model through mlx-lm
  (CVE-2026-5843, GHSA-9q3w-6wx3-7vh9). 2.0.8 refuses such a model; see *Code in Model
  Directories*.
- mlx-knife 2.0.0 up to and including 2.0.8b2 runs commands with the directory they are started
  from on the module search path of the processes they start — the `serve` and `embed-serve`
  server process, and the interpreters Python starts underneath any command — so Python modules
  found there are executed (GHSA-3pcp-w323-9wmv).

2.0.8 requires Python 3.11 or later. On Python 3.10, 2.0.7 is the last installable release and
receives no fix; the workarounds in both advisories apply to it.

| Version | Security Support |
| ------- | ---------------- |
| 2.0.8   | :white_check_mark: Recommended — current stable (Python ≥ 3.11) |
| 2.0.7   | :warning: Not patched — last release for Python 3.10; apply the workarounds in GHSA-9q3w-6wx3-7vh9 and GHSA-3pcp-w323-9wmv |
| < 2.0.7 | :x: Upgrade recommended |

## Additional Resources

- [HuggingFace Security](https://huggingface.co/docs/hub/security)
- [Apple Platform Security](https://support.apple.com/guide/security/welcome/web)
- [Python Security](https://python.readthedocs.io/en/latest/library/security_warnings.html)

---

**Remember**: Security is everyone's responsibility. If something doesn't feel right, please report it! 🦫
