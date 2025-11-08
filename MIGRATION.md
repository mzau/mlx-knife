# Migration Guide: MLX Knife 1.x → 2.0

This guide helps you transition from MLX Knife 1.x (MIT License) to 2.0 (Apache License 2.0).

## License Change: MIT → Apache 2.0

**Important:** MLX Knife 2.0 changes the license from MIT to Apache License 2.0.

### What This Means for Users

**Practical Impact:**
- ✅ Still **free and open source**
- ✅ Still **commercial use allowed**
- ✅ Still **modification allowed**
- ⚠️ **Attribution required** (include NOTICE file in distributions)
- ⚠️ **Patent grant** (explicit patent protection for users)

**Key Differences:**
| Aspect | MIT (1.x) | Apache 2.0 (2.0+) |
|--------|-----------|-------------------|
| Use | ✅ Free | ✅ Free |
| Modify | ✅ Allowed | ✅ Allowed |
| Commercial | ✅ Allowed | ✅ Allowed |
| Attribution | Optional | **Required** (NOTICE file) |
| Patent Grant | Implicit | **Explicit** (better protection) |

### Why Apache 2.0?

1. **Better Patent Protection:** Explicit patent grant protects users from patent litigation
2. **Industry Standard:** Used by major projects (Kubernetes, TensorFlow, Apache projects)
3. **Clear Contribution Terms:** Explicit contributor licensing for future contributions

### For Users: What You Need to Do

**If you just use MLX Knife CLI:**
- ✅ Nothing! Just upgrade: `pip install --upgrade mlx-knife`
- The license change doesn't affect CLI usage

**If you distribute MLX Knife in your product:**
- ⚠️ Include the `LICENSE` file from the mlx-knife package
- ⚠️ Include the `mlxk2/NOTICE` file in your distribution
- Example: If bundling mlx-knife in a Mac app, include both files in your "Licenses" folder

**If you're a 1.x user and prefer MIT:**
- 🔒 Version 1.1.1 remains available under MIT License
- Install specific version: `pip install mlx-knife==1.1.1`
- Security updates will focus on 2.x (Apache 2.0)

## Behavior Changes

### `rm` Command: Lock File Handling

The `rm` command now handles HuggingFace lock files more safely:

**1.x Behavior:**
```bash
$ mlxk rm Phi-3-mini
Delete entire model mlx-community/Phi-3-mini-4k-instruct-4bit? [y/N] y
Model mlx-community/Phi-3-mini-4k-instruct-4bit completely deleted.
Clean up cache files? [Y/n] y
Cache files cleaned up (3 files).
```

**2.0 Behavior:**
```bash
# Models with active locks require --force
$ mlxk rm Phi-3-mini
Error: Model has active locks. Use --force to override.

# With --force, deletion + lock cleanup happen automatically
$ mlxk rm Phi-3-mini --force
rm: mlx-community/Phi-3-mini-4k-instruct-4bit — deleted: Deleted entire model mlx-community/Phi-3-mini-4k-instruct-4bit
```

**Why the change?**
- **Safety:** Prevents accidental deletion of models that may be in use
- **Simplicity:** One confirmation instead of two separate prompts
- **Clarity:** Explicit `--force` makes automation intent clear
- **Automatic cleanup:** Lock files are cleaned silently (no separate prompt)

**Note:** Lock file cleanup count is available in `--json` output (`lock_files_cleaned` field).

**Migration for scripts:**
- Interactive usage: No change needed (you'll get a clear error message if locks exist)
- Automation: Add `--force` if your scripts delete models programmatically

## New Features in 2.0

### 1. JSON API for Automation

All commands now support `--json` for machine-readable output:

```bash
mlxk list --json
mlxk show Phi-3-mini --json
mlxk rm Phi-3-mini --force --json
```

**Example:**
```bash
# Extract model names for scripting
mlxk list --json | jq -r '.data.models[] | .name'

# Check lock cleanup count
mlxk rm test-model --force --json | jq '.data.lock_files_cleaned'
```

### 2. Enhanced Error Handling & Logging

- **Structured errors** with request IDs for debugging
- **Log levels:** `--log-level debug|info|warning|error`
- **JSON logs:** `--log-json` or `MLXK2_LOG_JSON=1`
- **Auto-redaction:** HF tokens and user paths automatically hidden

```bash
# Debug mode
mlxk run Phi-3-mini "test" --log-level debug

# JSON logs for production
mlxk server --log-json --log-level info
```

### 3. Runtime Compatibility Checks

Pre-flight validation catches issues before model loading:

```bash
$ mlxk show Phi-3-mini
Model: Phi-3-mini-4k-instruct-4bit
Health: healthy
Runtime: compatible

$ mlxk show legacy-model
Model: legacy-model
Health: healthy (files OK)
Runtime: incompatible
Reason: Legacy format not supported by mlx-lm
```

### 4. Better Stop Token Detection

Fixed issues with multi-EOS models:
- No more visible stop tokens (`<|end|>`)
- No more "self-conversation" (model continuing after response)
- Works with MXFP4, Qwen, Llama models

### 5. Improved Human Output Formatting

- Shorter model names (strip `mlx-community/` prefix by default)
- Relative timestamps ("2 days ago")
- Better alignment and readability
- Use `--verbose` for full names and details

## Command Compatibility

| Command | 1.x | 2.0 | Notes |
|---------|-----|-----|-------|
| `mlxk list` | ✅ | ✅ | Improved formatting, add `--verbose` for full names |
| `mlxk show <model>` | ✅ | ✅ | Added `runtime_compatible` field |
| `mlxk pull <model>` | ✅ | ✅ | Better error messages |
| `mlxk rm <model>` | ✅ | ⚠️ | Lock files require `--force` (safer) |
| `mlxk run <model>` | ✅ | ✅ | Better stop token handling |
| `mlxk server` | ✅ | ✅ | Added `--log-level`, `--log-json` |
| `mlxk health` | ✅ | ✅ | Added runtime compatibility checks |

## Package & Command Names

- **PyPI Package:** `mlx-knife` (unchanged)
- **Primary Command:** `mlxk` (unchanged)
- **Aliases:** `mlxk-json`, `mlxk2` (for backwards compatibility)

```bash
# All three commands are identical
mlxk --version       # → 2.0.0
mlxk-json --version  # → 2.0.0
mlxk2 --version      # → 2.0.0
```

## Installation & Upgrade

### Upgrade from 1.x

```bash
# Simple upgrade
pip install --upgrade mlx-knife

# Verify version
mlxk --version  # Should show: mlxk 2.0.0
```

### Upgrade from 2.0.0-beta.x

If you've been using beta versions, use a clean reinstall to avoid conflicts:

```bash
# Clean upgrade from beta
pip uninstall mlx-knife -y
pip install mlx-knife

# Verify version
mlxk --version   # Should show: mlxk 2.0.0
mlxk2 --version  # Should show: mlxk2 2.0.0 (alias)
```

**Why clean reinstall for beta users?**
Beta versions used `mlxk2` as the primary command. A clean reinstall ensures all command aliases are properly installed.

### Fresh Installation

```bash
# Install from PyPI
pip install mlx-knife

# Or from GitHub release
pip install https://github.com/mzau/mlx-knife/releases/download/v2.0.0/mlx_knife-2.0.0-py3-none-any.whl
```

### Staying on 1.x (MIT License)

```bash
# Pin to 1.x version
pip install mlx-knife==1.1.1

# Or in requirements.txt
mlx-knife==1.1.1
```

## Data & Cache Compatibility

✅ **Your model cache is 100% compatible**

- Same HuggingFace cache: `~/.cache/huggingface/hub`
- All 1.x models work in 2.0 immediately
- No re-download required
- No migration needed

## Testing Before Upgrade

```bash
# Test 2.0 in a virtual environment
python3 -m venv test-mlxk2
source test-mlxk2/bin/activate
pip install mlx-knife

# Verify your workflow
mlxk list
mlxk run YourFavoriteModel "test prompt"
mlxk rm test-model --force  # Note: --force for locks

# If satisfied, upgrade
deactivate
pip install --upgrade mlx-knife
```

## Upgrade Checklist for Automation Scripts

If you have scripts using mlxk:

- [ ] **Add `--force` to `mlxk rm` commands** (if deleting programmatically)
- [ ] **Replace output parsing with `--json`** (don't parse human output)
- [ ] **Test in virtual environment first**
- [ ] **Update error handling** for structured error responses
- [ ] **Consider `--log-json`** for production logging

## FAQ

**Q: Will my 1.x scripts break?**
A: Only `mlxk rm` scripts that delete models with active locks without `--force`. This is a safety improvement.

**Q: Why does `rm` need `--force` for locks?**
A: To prevent accidental deletion of models that may be in use. Locks indicate active downloads or usage.

**Q: Can I still use interactive deletion?**
A: Yes! Without `--force`, you'll get clear error messages about locks and can decide whether to use `--force`.

**Q: Do I need to re-download models?**
A: No! All cached models from 1.x work immediately.

**Q: What about the web chat interface?**
A: `simple_chat.html` works with both 1.x and 2.0 (OpenAI API unchanged).

**Q: Can I contribute to 1.x?**
A: 1.x is in maintenance mode. New contributions go to 2.x (Apache 2.0).

**Q: Where's the 1.x source code?**
A: The `1.x-legacy` branch contains the final MIT version (1.1.1).

## Need Help?

- **Issues:** https://github.com/mzau/mlx-knife/issues
- **Discussions:** https://github.com/mzau/mlx-knife/discussions
- **Documentation:** https://github.com/mzau/mlx-knife

## Timeline

- **1.1.1 (MIT):** Final 1.x release - September 2025
- **2.0.0 (Apache 2.0):** Stable release - November 2025
- **1.x Support:** Security fixes only

---

**The BROKE Team** 🦫
