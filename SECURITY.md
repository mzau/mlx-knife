# Security Policy

## Overview

MLX Knife is designed to run locally on your Apple Silicon Mac. It prioritizes user privacy and security by keeping all model execution local. The only network activity is downloading models from HuggingFace (a trusted source).

## Security Model

### What MLX Knife Does
- ✅ Runs models locally on your device
- ✅ Downloads models only from HuggingFace (trusted repository)
- ✅ API server binds to localhost by default
- ✅ No telemetry or usage tracking
- ✅ No external API calls (except HuggingFace for downloads)

### What MLX Knife Doesn't Do
- ❌ No data is sent to external servers
- ❌ No model outputs are logged or transmitted
- ❌ No user tracking or analytics
- ❌ No automatic updates or phone-home features

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
- **Cleanup**: Use `mlxk rm <model>` to safely remove models

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

| Version | Supported          |
| ------- | ------------------ |
| 1.0-rc1 | :white_check_mark: |
| < 1.0   | :x:                |

## Additional Resources

- [HuggingFace Security](https://huggingface.co/docs/hub/security)
- [Apple Platform Security](https://support.apple.com/guide/security/welcome/web)
- [Python Security](https://python.readthedocs.io/en/latest/library/security_warnings.html)

---

**Remember**: Security is everyone's responsibility. If something doesn't feel right, please report it! 🦫