# Security Policy

## Overview

MLX Knife is designed to run locally on your Apple Silicon Mac. It prioritizes user privacy and security by keeping all model execution local. Network activity is limited to explicit interactions with Hugging Face: downloading models (pull) and, in 2.0 alpha, an opt‑in alpha upload (push) when you run it explicitly. No background network traffic.

## Security Model

### What MLX Knife Does
- ✅ Runs models locally on your device
- ✅ Downloads models only from HuggingFace (trusted repository)
- ✅ API server binds to localhost by default
- ✅ No telemetry or usage tracking
- ✅ No external API calls (except explicit Hugging Face interactions: downloads via pull; optional upload via experimental push)
- ✅ Can upload a local workspace to Hugging Face only when you explicitly run `mlxk2 push` (alpha feature, opt‑in)

### What MLX Knife Doesn't Do
- ❌ No data is sent to external servers automatically or in the background
- ❌ No model outputs are logged or transmitted
- ❌ No user tracking or analytics
- ❌ No automatic updates or phone-home features
  
  Note: The alpha `push` command will upload files from a user‑selected local folder to Hugging Face only when you run it explicitly and provide credentials. It never runs implicitly.

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
- **Risk**: Malicious models could theoretically exist on HuggingFace
- **Mitigation**: Only download models from trusted organizations (e.g., `mlx-community`)

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
- Could potentially be exposed to the internet (check firewall!)

**Recommendations for network access:**
- Use a reverse proxy with authentication (nginx, Caddy)
- Implement firewall rules
- Never expose directly to the internet
- Consider VPN-only access

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
1. **Download models only from trusted sources** (prefer `mlx-community/*`)
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

We provide security updates for these versions:

| Version | Security Support   |
| ------- | ------------------ |
| 2.0.1   | :white_check_mark: Current stable |
| 2.0.0   | :white_check_mark: Supported |
| < 2.0.0 | :x: Upgrade recommended |

## Additional Resources

- [HuggingFace Security](https://huggingface.co/docs/hub/security)
- [Apple Platform Security](https://support.apple.com/guide/security/welcome/web)
- [Python Security](https://python.readthedocs.io/en/latest/library/security_warnings.html)

---

**Remember**: Security is everyone's responsibility. If something doesn't feel right, please report it! 🦫
