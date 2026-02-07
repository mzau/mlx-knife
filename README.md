# <img src="https://github.com/mzau/mlx-knife/raw/main/broke-logo.png" alt="BROKE Logo" width="60" align="middle"> MLX-Knife 2.0

<p align="center">
  <img src="https://github.com/mzau/mlx-knife/raw/main/mlxk-demo.gif" alt="MLX Knife Demo" width="900">
</p>

**Current Version: 2.0.4-beta.10** (Stable: 2.0.3)

[![GitHub Release](https://img.shields.io/badge/version-2.0.4--beta.10-blue.svg)](https://github.com/mzau/mlx-knife/releases)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Python 3.10-3.12](https://img.shields.io/badge/python-3.10--3.12-blue.svg)](https://www.python.org/downloads/)
[![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-green.svg)](https://support.apple.com/en-us/HT211814)
[![MLX](https://img.shields.io/badge/MLX-Latest-orange.svg)](https://github.com/ml-explore/mlx)

**Release Notes:** See [CHANGELOG.md](CHANGELOG.md) for detailed changes, fixes, and migration guides.


## Features

> **⚠️ Beta.9 Audio Bug:** If you installed `mlx-knife[audio]==2.0.4b9` from PyPI, audio transcription fails with "Processor not found". Upgrade to beta.10: `pip install mlx-knife[all]==2.0.4b10`

### What's New in 2.0.4 (Coming Soon - Currently Beta)
- **Audio Transcription (STT)** - Whisper speech-to-text (`--audio` flag, `pip install mlx-knife[audio]`)
- **Vision Models with EXIF Metadata** - Image analysis + automatic GPS/date/camera extraction visible to the model
- **Unix Pipe Integration** - Chain models without temp files (`vision → text` workflows)
- **Local Development Workflow** - Clone → Repair → Test models without HuggingFace round-trips
- **Community Model Repair Tool** - Fix broken mlx-vlm models with `--repair-index`
- **Resumable Downloads** - Interrupted clone/pull operations continue automatically
- **Safe Vision Batch Processing** - Automatic chunking prevents Metal OOM crashes
- **Workspace Path Support** - Run/show/server/list commands work with local directories
- **Updated [SECURITY.md](SECURITY.md)** - Clarifies mlx-knife vs. upstream library behavior; important for offline/air-gapped use

### Core Functionality
- **List & Manage Models**: Browse your HuggingFace cache with MLX-specific filtering
- **Model Information**: Detailed model metadata including quantization info
- **Download Models**: Pull models from HuggingFace with progress tracking
- **Run Models**: Native MLX execution with streaming and chat modes
- **Health Checks**: Verify model integrity and MLX runtime compatibility
- **Cache Management**: Clean up and organize your model storage
- **Privacy & Network**: No background network or telemetry; only explicit Hugging Face interactions when you run pull or the experimental push.

### Unix Pipe Integration (Beta, 2.0.4)
Chain models with standard Unix pipes - no temp files needed:
```bash
export MLXK2_ENABLE_PIPES=1

# Model chaining
cat article.txt | mlx-run translator_model - | mlx-run summarizer_model - "3 bullets"

# Works with Unix tools
mlx-run chat_model "explain quicksort" | tee explanation.txt | head -20
```
Robust handling of SIGPIPE and early pipe termination (`| head`, `| grep -m1`).

### Requirements
- macOS with Apple Silicon
- Python 3.10-3.12 (see Python Compatibility below)
- 8GB+ RAM recommended + RAM to run LLM

## ⚖️ Model Usage and Licenses

`mlx-knife` is a **tooling layer** for running ML models (e.g. from Hugging Face) locally.
The project does **not** distribute any model weights and does **not** decide which models you use or how you use them.

Please note:

- Each model (weights, tokenizer, configuration, etc.) is governed by its **own license**.
- When `mlx-knife` downloads a model from a third-party service (e.g. Hugging Face), it does so **on your behalf**.
- **You** are responsible for:
  - reading and understanding the license of each model you use,
  - complying with any restrictions (e.g. *Non-Commercial*, *Research Only*, RAIL, etc.),
  - ensuring that your use of a given model (private, research, commercial, on-prem services, etc.) is legally permitted.

The `mlx-knife` source code itself is provided under the open-source license specified in this repository.
This license applies **only** to the `mlx-knife` code and **does not extend** to any external models.

> This is not legal advice. Always refer to the original model license text and, if necessary, seek professional legal counsel.

### Python Compatibility

✅ **Python 3.10 - 3.12** - Full support (Text + Vision + Audio)
❌ **Python 3.9** - Not supported (MLX 0.30+ requires 3.10+)
❌ **Python 3.13+** - Not supported (miniaudio lacks pre-built wheels)

**Note:** Vision/Audio features require Python 3.10+. Recommended: **Python 3.10 or 3.11** for best compatibility.



## Installation

### 1. PyPI Stable (2.0.3 - Text models only)

```bash
pip install mlx-knife
mlxk --version  # → mlxk 2.0.3
```

**Requirements:** macOS Apple Silicon, Python 3.9-3.12

### 2. PyPI Beta (2.0.4-beta.10 - Text + Vision + Audio)

```bash
pip install mlx-knife[all]==2.0.4b10
mlxk --version  # → mlxk 2.0.4b10
```

**Requirements:** macOS Apple Silicon, Python 3.10-3.12
**Features:** Audio STT (Whisper), Vision with EXIF metadata, full tiktoken workaround

### 3. Developer Installation

```bash
git clone https://github.com/mzau/mlx-knife.git
cd mlx-knife
pip install -e ".[all,dev,test]"

mlxk --version  # → mlxk 2.0.4b10
pytest -v
```

**Requirements:** macOS Apple Silicon, Python 3.10-3.12

### Migrating from 1.x

If you're upgrading from MLX Knife 1.x, see [MIGRATION.md](MIGRATION.md) for important information about the license change (MIT → Apache 2.0) and behavior changes.


## Quick Start

```bash
# List models (human-readable)
mlxk list
mlxk list --health
mlxk list --verbose --health

# Check cache health
mlxk health

# Show model details
mlxk show "mlx-community/Phi-3-mini-4k-instruct-4bit"

# Pull a model
mlxk pull "mlx-community/Llama-3.2-3B-Instruct-4bit"

# Resume interrupted download (skip prompt)
mlxk pull "model-name" --force-resume

# Run interactive chat
mlxk run "Phi-3-mini" -c

# Start OpenAI-compatible server
mlxk serve --port 8080
```

## Commands

| Command | Description |
|---------|-------------|
| `list` | Model discovery with JSON output; supports cache and workspace paths |
| `show` | Detailed model information with --files, --config |
| `health` | Corruption detection and cache analysis |
| `pull` | HuggingFace model downloads with corruption detection |
| `rm` | Model deletion with lock cleanup and fuzzy matching |
| `run` | Interactive and single-shot model execution with streaming/batch modes |
| `server`/`serve` | OpenAI-compatible API server; SIGINT-robust (Supervisor); SSE streaming |
| `clone` | Model workspace cloning - create local editable copy from cache |
| `push` | Upload to HuggingFace Hub (requires `--private` flag for safety) |
| 🔒 `convert` | **Experimental** - Workspace transformations; requires `MLXK2_ENABLE_ALPHA_FEATURES=1` |
| 🔒 `pipe mode` | **Beta feature** - Unix pipes with `mlxk run <model> - ...`; requires `MLXK2_ENABLE_PIPES=1` |

## Model References

MLX-Knife supports multiple ways to reference models:

### HuggingFace Models (Cache)

| Format | Example | Description |
|--------|---------|-------------|
| Full name | `mlx-community/Phi-4-4bit` | Exact HuggingFace repo ID |
| Short name | `Phi-4` | Fuzzy match against cache |
| With hash | `Phi-4@e96f3b2` | Specific commit/version |

```bash
mlxk run "mlx-community/Phi-4-4bit" "Hello"
mlxk run "Phi-4" "Hello"                    # Fuzzy match
mlxk show "Qwen3@e96" --json                # Specific version
```

### Local Paths (2.0.4-beta.6+)

| Format | Example |
|--------|---------|
| Relative | `./my-workspace` |
| Absolute | `/Volumes/External/model` |
| Prefix match | `./gemma-` (all workspaces starting with "gemma-") |
| Directory | `.` (all workspaces in current directory) |

```bash
# List workspaces
mlxk list .                        # All workspaces in current directory
mlxk list ./gemma-                 # Prefix match: gemma-3n-4bit, gemma-3n-FIXED-4bit, ...
mlxk list $PWD/models              # Absolute path → absolute output

# Clone → Run
mlxk clone org/model ./workspace
mlxk run ./workspace "Hello"

# Convert → Run
mlxk convert ./broken ./fixed --repair-index
mlxk run ./fixed "Test"
```

**Output format:** List output mirrors input format - relative patterns produce relative names (like `ls`), absolute paths produce absolute names.

**Disambiguating paths vs cache names:** When a local directory exists with the same name as a cached model, use `./` prefix to force workspace resolution. Otherwise, cache lookup is attempted first.

---

## Workspace Development Workflow (2.0.4-beta.6+)

**Complete local development cycle** for model experimentation, repair, and testing without HuggingFace round-trips:

```bash
# Clone → Repair → Test → Publish (optional)
mlxk clone "model" ./workspace
MLXK2_ENABLE_ALPHA_FEATURES=1 mlxk convert ./workspace ./fixed --repair-index
mlxk list .                                        # See all local workspaces
mlxk run ./fixed "test prompt"                     # Local inference
mlxk server --model ./fixed                        # Dev server
mlxk push ./fixed "your-org/model"                 # Optional publish
```

**Key capabilities:**
- **Model repair:** Fix index/shard mismatches from mlx-vlm #624
- **Local testing:** Run/server/show without pushing to HuggingFace
- **Side-by-side comparison:** Multiple workspaces with parallel servers
- **Rapid iteration:** Clone → Modify → Test loop

### Workspace Commands Reference

| Command | Workspace Support | Example |
|---------|-------------------|---------|
| `run` | ✅ Yes | `mlxk run ./workspace "prompt"` |
| `show` | ✅ Yes | `mlxk show ./workspace --files` |
| `health` | ✅ Yes | `mlxk health ./workspace` |
| `server` | ✅ Yes | `mlxk server --model ./workspace` |
| `clone` | ✅ Creates | `mlxk clone model ./workspace` |
| `convert` | ✅ Yes | `mlxk convert ./in ./out --repair-index` |
| `push` | ✅ Yes | `mlxk push ./workspace "org/name"` |
| `list` | ✅ Yes | `mlxk list .` or `mlxk list ./gemma-` |
| `pull` | ❌ Cache only | Downloads to HuggingFace cache |
| `rm` | ❌ Cache only | Use `rm -rf ./workspace` for local directories |

---

## Web Interface

For a web-based chat UI, use **[nChat](https://github.com/mzau/broke-nchat)** - a lightweight web interface for the BROKE ecosystem:

```bash
# Clone once (local setup):
git clone https://github.com/mzau/broke-nchat.git
cd broke-nchat

# Start mlx-knife server:
mlxk serve

# Open web UI:
open webui/index.html
```

**On-Prem:** Pure HTML/CSS/JS - runs entirely locally, zero dependencies.

**Note:** nChat is a separate project designed for the entire BROKE ecosystem (MLX Knife + BROKE Cluster). See [nChat README](https://github.com/mzau/broke-nchat/blob/main/README.md) for CORS configuration.


## Multi-Modal Support

MLX Knife supports multiple input modalities beyond text. All multi-modal features share a **common output pattern**: model responses are followed by collapsible metadata tables for transparency and traceability.

### Vision (Beta)

Image analysis via the `--image` flag (CLI and server). Requires Python 3.10+.

#### Requirements

- **Python 3.10+** (mlx-vlm dependency)
- **Installation:** `pip install mlx-knife[vision]`
- **Backend:** mlx-vlm 0.3.10 (auto-installed from PyPI)

#### Usage

```bash
# Image analysis with custom prompt
mlxk run "mlx-community/Llama-3.2-11B-Vision-Instruct-4bit" \
  --image photo.jpg "Describe what you see in detail"

# Multiple images (space-separated or glob)
mlxk run vision-model --image img1.jpg img2.jpg img3.jpg "Compare these images"
mlxk run vision-model --image photos/*.jpg "Which images show outdoor scenes?"

# Auto-prompt (default: "Describe the image.")
mlxk run vision-model --image cat.jpg

# Text-only on vision model (no --image flag)
mlxk run "mlx-community/Llama-3.2-11B-Vision-Instruct-4bit" "What is 2+2?"
```

#### Batch Processing

**Terminology Note:** mlx-knife uses "batch" in the traditional computing sense (sequential job processing in groups), not ML inference batching (parallel batch_size > 1 in a single forward pass). Images are processed sequentially in groups for memory safety, not performance parallelization.

**Breaking Change (2.0.4-beta.6):** Vision processing now defaults to processing **one image at a time** for maximum stability on all systems. Use `--chunk N` to process multiple images per batch when your system can handle it.

```bash
# Default: one image at a time (most robust, automatic chunking)
mlxk run pixtral "Describe image" --image photos/*.jpg

# Faster: 5 images per batch (requires more RAM, may trigger model-specific issues)
mlxk run pixtral "Describe images" --chunk 5 --image photos/*.jpg

# Alternative: Use --prompt flag (useful when experimenting with different prompts)
mlxk run pixtral --chunk 5 --image photos/*.jpg --prompt "Describe images"

# Set default chunk size via environment variable
export MLXK2_VISION_CHUNK_SIZE=3
mlxk run pixtral "Describe images" --image photos/*.jpg
```

**Why chunking?**
- **Safety:** Prevents Metal OOM crashes by limiting images per processing group (`--chunk N`)
- **Isolation:** Fresh inference session per chunk (KV cache cleared, conversation context reset)
- **Trade-off:** ~2-3s model load overhead per chunk vs guaranteed isolation

**Reliability (2.0.4-beta.7):** Vision models can sometimes describe details they didn't actually see. MLX Knife prevents this automatically:
- **Default (chunk=1):** Most reliable - each image processed independently
- **Larger chunks:** Still safe, but models may occasionally confuse details between images in the same batch

For maximum accuracy, use the default chunk=1 (no configuration needed).

**Server API:**
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "pixtral", "chunk": 3, "messages": [...]}'
```

**Note:** `chunk` is an mlx-knife extension parameter. See [SERVER-HANDBOOK.md](docs/SERVER-HANDBOOK.md) for details.

#### Metadata Output Format

When processing images, MLX Knife automatically prepends metadata in a **collapsible table** (collapsed by default) **before** the model output:

```
<details>
<summary>📸 Chunk 1/3: Images 1-4</summary>

| Image | Filename | Original | Location | Date | Camera |
|-------|----------|----------|----------|------|--------|
| 1 | image_abc123.jpeg | beach.jpg | 📍 32.7900°N, 16.9200°W | 📅 2023-12-06 12:19 | 📷 Apple iPhone SE |
| 2 | image_def456.jpeg | mountain.jpg | 📍 32.8700°N, 17.1700°W | 📅 2023-12-10 15:42 | 📷 Apple iPhone SE |
| 3 | image_xyz789.jpeg | sunset.jpg | 📍 32.8200°N, 17.0500°W | 📅 2023-12-08 18:30 | 📷 Apple iPhone SE |
| 4 | image_uvw456.jpeg | forest.jpg | 📍 32.8800°N, 17.1200°W | 📅 2023-12-09 10:15 | 📷 Apple iPhone SE |

</details>

A beach with palm trees and clear blue water. A mountain landscape with snow-capped peaks...
```

**Chunk information in summary:**
- Shows current chunk and total chunks (e.g., "Chunk 1/3")
- Shows image range in current chunk (e.g., "Images 1-4")
- Helps track progress in WebUI and prevents confusion about which images are being described

**Why metadata comes first:**
- The model sees GPS, date, and camera info when analyzing images (enables location/time-aware descriptions)
- The markdown table shows you exactly what the model knows about each image
- Helps verify which description belongs to which file

**Metadata includes:**
- **Image ID** → **Filename mapping** (identify which description belongs to which file)
- **GPS coordinates** (latitude/longitude, if available in EXIF)
  - Precision: 4 decimal places (~11m accuracy) for street-level context
- **Capture date/time** (ISO 8601 format)
- **Camera model** (device info)

**Privacy control:**

EXIF extraction is **enabled by default**. To disable (e.g., for privacy-sensitive images):

```bash
export MLXK2_EXIF_METADATA=0
mlxk run vision-model --image photo.jpg "describe"
```

**Output is the same for CLI and server** - metadata tables work in terminals, web UIs (nChat), and can be parsed programmatically.

#### Limitations

- **Image limits:** Model-dependent due to Metal / unified-memory constraints and peak activation usage
  - **pixtral-12b-8bit:** Up to 5 images tested on M2 Max 64GB (multi-image capable)
  - **Llama-3.2-11B / Other models:** Single-image only
  - **Larger models (24B+):** Limited to 1-2 images on 64GB RAM
  - **Default server guardrails:** 20 MB per image, 50 MB total (configurable). Base64 encoding adds ~33% overhead.

#### Server API

Vision models work with OpenAI-compatible `/v1/chat/completions` endpoint using base64-encoded images:

```bash
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "llama-vision",
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "What is in this image?"},
      {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
    ]
  }]
}'
```

#### Model Compatibility

**⚠️ Important:** Vision support relies on mlx-vlm (upstream), which has known stability issues. While `mlxk health` verifies file integrity, **runtime failures may occur** with certain model architectures due to upstream bugs.

**✅ Tested & Working Models** (mlx-knife v2.0.4-beta.6):

| Model | Size | Notes |
|-------|------|-------|
| `mlx-community/pixtral-12b-8bit` | ~13.5GB | **⭐ Recommended:** Excellent text recognition, multi-image support (5+ images on M2 Max 64GB); beta.4+ for text-only |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-4bit` | ~6.5GB | Reliable, good quality; single-image only |
| `Devstral-Small-2-24B-Instruct-2512-6bit` | ~14GB | Single-image only on 64GB RAM; requires `--repair-index` (mlx-vlm #624); better for Mac Studio/Ultra |

**❌ Known Issues** (as of 2026-01-03):

| Model | Issue | Workaround |
|-------|-------|------------|
| `Mistral-Small-3.1-24B-Instruct-2503-4bit` | Vision feature mismatch (5476 positions ≠ 1369 features). **Note:** This is a different bug than mlx-vlm #624 - `--repair-index` cannot fix it | Use alternative models (pixtral-12b-8bit, Llama-3.2-11B) |
| `MiMo-VL-7B-RL-bf16` | NoneType iteration error | mlx-vlm processor bug, no workaround |
| `DeepSeek-OCR-8bit` | Runs but hallucinates details | Quality issue, not recommended |

**Models affected by mlx-vlm #624** (index/shard mismatch) can be repaired:

```bash
mlxk convert <model> <output> --repair-index
```

**Recommendation:** Test models with real images before production use. The vision ecosystem is evolving rapidly - this list will be updated as mlx-vlm matures.

**Reporting Issues:** If you encounter vision model failures, please report with model name and error message to help improve compatibility tracking.

### Audio Transcription (Speech-to-Text)

> **🎙️ New in beta.9/10:** Professional STT via dedicated Whisper models (mlx-audio backend). Beta.10 fixes PyPI install (no Git workaround needed). Backward compatible with Gemma-3n multimodal audio (mlx-vlm).

**Requirements:**
- **Python 3.10+** (mlx-audio dependency)
- **Installation:** `pip install mlx-knife[audio]` (tiktoken workaround bundled)
- **No system dependencies:** MP3/WAV decoding via embedded libsndfile (no ffmpeg or Homebrew required)

**✅ Recommended Models** (mlx-knife v2.0.4-beta.10):

| Model | Backend | Size | Duration | Notes |
|-------|---------|------|----------|-------|
| `whisper-large-v3-turbo-4bit` | mlx-audio | ~464MB | >10 min | **Recommended** - Best accuracy/speed balance |
| `whisper-tiny` | mlx-audio | ~74MB | >10 min | Fast, lower accuracy |
| `gemma-3n-E2B-it-4bit` | mlx-vlm | ~2.1GB | ~30s | Multimodal (vision+audio), requires workspace repair |

**🔧 Backend Architecture:**

mlx-knife automatically routes audio models to the optimal backend:
- **Whisper/Voxtral** → mlx-audio (dedicated STT, >10min duration, best accuracy)
- **Gemma-3n** → mlx-vlm (multimodal audio, ~30s limit, backward compatible)

**⚙️ Audio Defaults:**

| Setting | Audio | Text/Vision | Reason |
|---------|-------|-------------|--------|
| Temperature | 0.0 | 0.7 | Greedy decoding (STT best practice) |
| Default Prompt | "Transcribe this audio." | - | Minimal prompt for pure transcription |

**💡 Quick Start:**

```bash
# Pull a Whisper model (one-time setup)
mlxk pull mlx-community/whisper-large-v3-turbo-4bit

# Transcribe audio (WAV, MP3, M4A - native on macOS)
mlxk run whisper-large --audio speech.mp3
# → Automatic greedy decoding (temp=0.0)

# With language hint for better accuracy
mlxk run whisper-large --audio speech.mp3 --language en

# Longer audio (>10 minutes supported)
mlxk run whisper-large --audio podcast.wav
```

**⚠️ Known Limitations:**

| Limitation | Whisper Models | Gemma-3n (Multimodal) | Workaround |
|------------|----------------|------------------------|------------|
| **Duration** | >10 minutes ✅ | ~30 seconds (token limit) | Use Whisper for long audio |
| **File size** | 50MB max | 50MB max | Split larger files |
| **Formats** | WAV, MP3, M4A (macOS native, Linux needs ffmpeg) | WAV | M4A uses Core Audio on macOS |
| **Legacy models** | Some use old `weights.npz` format | - | Use models with `.safetensors` |

**🎯 Advanced Usage:**

```bash
# Explicit temperature control (0.0 = greedy, deterministic)
mlxk run whisper-large --audio speech.wav --temperature 0.0

# Force specific language (improves accuracy)
mlxk run whisper-large --audio german.mp3 --language de

# Segment metadata (MLXK2_AUDIO_SEGMENTS=1 for timestamps)
MLXK2_AUDIO_SEGMENTS=1 mlxk run whisper-large --audio meeting.wav
```

**🔄 Gemma-3n Multimodal (Backward Compatibility):**

> **Note:** Gemma-3n requires workspace repair due to mlx-vlm #624. Use Whisper for production STT.

```bash
# One-time setup (if using Gemma-3n)
MLXK2_ENABLE_ALPHA_FEATURES=1
mlxk clone mlx-community/gemma-3n-E2B-it-4bit ./gemma-3n-audio
mlxk convert ./gemma-3n-audio ./gemma-3n-audio-FIXED --repair-index

# Run (30s limit, multimodal audio)
mlxk run ./gemma-3n-audio-FIXED --audio short-clip.wav
```


## JSON API

> **📋 Complete API Specification**: See [JSON API Specification](docs/json-api-specification.md) for comprehensive schema, error codes, and examples.

All commands support both human-readable and JSON output (`--json` flag) for automation and scripting, enabling seamless integration with CI/CD pipelines and cluster management systems.

### Command Structure

All commands support JSON output via `--json` flag:

```bash
mlxk list --json | jq '.data.models[].name'
mlxk health --json | jq '.data.summary'
mlxk show "Phi-3-mini" --json | jq '.data.model'
```

**Response Format:**
```json
{
    "status": "success|error",
    "command": "list|health|show|pull|rm|clone|version|push|run|server",
    "data": { /* command-specific data */ },
    "error": null | { "type": "...", "message": "..." }
}
```

### Examples

#### List Models
```bash
mlxk list --json
# Output:
{
  "status": "success",
  "command": "list",
  "data": {
    "models": [
      {
        "name": "mlx-community/Phi-3-mini-4k-instruct-4bit",
        "hash": "a5339a41b2e3abcdef1234567890ab12345678ef",
        "size_bytes": 4613734656,
        "last_modified": "2024-10-15T08:23:41Z",
        "framework": "MLX",
        "model_type": "chat",
        "capabilities": ["text-generation", "chat"],
        "health": "healthy",
        "runtime_compatible": true,
        "reason": null,
        "cached": true
      }
    ],
    "count": 1
  },
  "error": null
}
```

#### Health Check
```bash
mlxk health --json
# Output:
{
  "status": "success",
  "command": "health",
  "data": {
    "healthy": [
      {
        "name": "mlx-community/Phi-3-mini-4k-instruct-4bit",
        "status": "healthy",
        "reason": "Model is healthy"
      }
    ],
    "unhealthy": [],
    "summary": { "total": 1, "healthy_count": 1, "unhealthy_count": 0 }
  },
  "error": null
}
```

#### Show Model Details
```bash
mlxk show "Phi-3-mini" --json --files
# Output (simplified):
{
  "status": "success",
  "command": "show",
  "data": {
    "model": {
      "name": "mlx-community/Phi-3-mini-4k-instruct-4bit",
      "hash": "a5339a41b2e3abcdefgh1234567890ab12345678",
      "size_bytes": 4613734656,
      "framework": "MLX",
      "model_type": "chat",
      "capabilities": ["text-generation", "chat"],
      "last_modified": "2024-10-15T08:23:41Z",
      "health": "healthy",
      "runtime_compatible": true,
      "reason": null,
      "cached": true
    },
    "files": [
      {"name": "config.json", "size": "1.2KB", "type": "config"},
      {"name": "model.safetensors", "size": "2.3GB", "type": "weights"}
    ],
    "metadata": null
  },
  "error": null
}
```

### Integration Examples

#### Broke-Cluster Integration
```bash
# Get available model names for scheduling
MODELS=$(mlxk list --json | jq -r '.data.models[].name')

# Check cache health before deployment
HEALTH=$(mlxk health --json | jq '.data.summary.healthy_count')
if [ "$HEALTH" -eq 0 ]; then
    echo "No healthy models available"
    exit 1
fi

# Download required models
mlxk pull "mlx-community/Phi-3-mini-4k-instruct-4bit" --json
```

#### CI/CD Pipeline Usage
```bash
# Verify model integrity in CI
mlxk health --json | jq -e '.data.summary.unhealthy_count == 0'

# Clean up CI artifacts
mlxk rm "test-model-*" --json --force

# Pre-warm cache for deployment
mlxk pull "production-model" --json
```

#### Model Management Automation
```bash
# Find models by pattern
LARGE_MODELS=$(mlxk list --json | jq -r '.data.models[] | select(.name | contains("30B")) | .name')

# Show detailed info for analysis
for model in $LARGE_MODELS; do
    mlxk show "$model" --json --config | jq '.data.model_config'
done
```


## Human Output

MLX Knife provides rich human-readable output by default (without `--json` flag).

**Error Handling (2.0.3+):** Errors print to stderr for clean pipe workflows:
```bash
mlxk show badmodel | grep ...      # Errors don't contaminate stdout
mlxk pull badmodel > log 2> err    # Capture errors separately
```

### Basic Usage

```bash
mlxk list
mlxk list --health
mlxk health
mlxk show "mlx-community/Phi-3-mini-4k-instruct-4bit"
mlxk pull "mlx-community/Llama-3.2-3B-Instruct-4bit"
```

### Pull Command

Download models from HuggingFace:

```bash
mlxk pull "mlx-community/Phi-3-mini-4k-instruct-4bit"
```

**Interrupted downloads (2.0.4-beta.5+):** If a download fails (network issue, Ctrl-C), `mlxk pull` will detect this and prompt to resume:

```bash
$ mlxk pull "model-name"
Model 'model-name' has partial download:
  No model weights found. Use --force-resume to attempt resume or 'mlxk rm' to delete.
Resume download? [Y/n]: y
```

**Automation/scripting:** Use `--force-resume` to skip the prompt:

```bash
mlxk pull "model-name" --force-resume
```

### List Filters

- `list`: Shows MLX chat models only (compact names, safe default)
- `list --verbose`: Shows all MLX models (chat + base) with full org/names and Framework column
- `list --all`: Shows all frameworks (MLX, GGUF, PyTorch)
- Flags are combinable: `--all --verbose`, `--all --health`, `--verbose --health`

### Health Status Display (--health flag)

The `--health` flag adds health status information to the output:

**Compact mode** (default, `--all`):
- Shows single "Health" column with values:
  - `healthy` - File integrity OK and MLX runtime compatible
  - `healthy*` - File integrity OK but not MLX runtime compatible (use `--verbose` for details)
  - `unhealthy` - File integrity failed or unknown format

**Verbose mode** (`--verbose --health`):
- Splits into "Integrity" and "Runtime" columns:
  - **Integrity:** `healthy` / `unhealthy`
  - **Runtime:** `yes` / `no` / `-` (dash = gate blocked by failed integrity)
  - **Reason:** Explanation when problems detected (wrapped at 26 chars for readability)

**Examples:**

```bash
# Compact health view
mlxk list --health
# Output:
# Name                    | Hash    | Size   | Modified | Type | Health
# Llama-3.2-3B-Instruct   | a1b2c3d | 2.1GB  | 2d ago   | chat | healthy
# Qwen2-7B-Instruct       | 1a2b3c4 | 4.8GB  | 3d ago   | chat | healthy*

# Verbose health view with details
mlxk list --verbose --health
# Output:
# Name                    | Hash    | Size   | Modified | Framework | Type | Integrity | Runtime | Reason
# Llama-3.2-3B-Instruct   | a1b2c3d | 2.1GB  | 2d ago   | MLX       | chat | healthy   | yes     | -
# Qwen2-7B-Instruct       | 1a2b3c4 | 4.8GB  | 3d ago   | PyTorch   | chat | healthy   | no      | Incompatible: PyTorch

# All frameworks with health status
mlxk list --all --health
# Output:
# Name                    | Hash    | Size   | Modified | Framework | Type    | Health
# Llama-3.2-3B-Instruct   | a1b2c3d | 2.1GB  | 2d ago   | MLX       | chat    | healthy
# llama-3.2-gguf-q4       | b2c3d4e | 1.8GB  | 3d ago   | GGUF      | unknown | healthy*
# broken-download         | -       | 500MB  | 1h ago   | Unknown   | unknown | unhealthy
```

**Design Philosophy:**
- `unhealthy` is a catch-all for anything not understood/supported (broken downloads, unknown formats, creative HuggingFace structures)
- `healthy` guarantees the model will work with `mlxk2 run`
- `healthy*` means files are intact but MLX runtime can't execute them (e.g., GGUF/PyTorch models, incompatible model_type, or mlx-lm version too old)

Note: JSON output is unaffected by these human-only filters and always includes full health/runtime data.


## Logging & Debugging

MLX Knife 2.0 provides structured logging with configurable output formats and levels.

### Log Levels

Control verbosity with `--log-level` (server mode):

```bash
# Default: Show startup, model loading, and errors
mlxk serve --log-level info

# Quiet: Only warnings and errors
mlxk serve --log-level warning

# Silent: Only errors
mlxk serve --log-level error

# Verbose: All logs including HTTP requests
mlxk serve --log-level debug
```

**Log Level Behavior:**
- `debug`: All logs + Uvicorn HTTP access logs (`GET /v1/models`, etc.)
- `info`: Application logs (startup, model switching, errors) + HTTP access logs
- `warning`: Only warnings and errors (no startup messages, no HTTP access logs)
- `error`: Only error messages

### JSON Logs (Machine-Readable)

Enable structured JSON output for log aggregation tools:

```bash
# JSON logs (recommended - CLI flag)
mlxk serve --log-json

# JSON logs (alternative - environment variable)
MLXK2_LOG_JSON=1 mlxk serve
```

**Note:** `--log-json` also formats Uvicorn access logs as JSON for consistent output.

**JSON Format:**
```json
{"ts": 1760830072.96, "level": "INFO", "msg": "MLX Knife Server 2.0 starting up..."}
{"ts": 1760830073.14, "level": "INFO", "msg": "Switching to model: mlx-community/...", "model": "..."}
{"ts": 1760830074.52, "level": "ERROR", "msg": "Model type bert not supported.", "logger": "root"}
```

**Fields:**
- `ts`: Unix timestamp
- `level`: Log level (INFO, WARN, ERROR, DEBUG)
- `msg`: Log message (HF tokens and user paths automatically redacted)
- `logger`: Source logger (`mlxk2` = application, `root` = external libraries like mlx-lm)
- Additional fields: `model`, `request_id`, `detail`, `duration_ms` (context-dependent)

### Security: Automatic Redaction

**Sensitive data is automatically removed from logs:**
- HuggingFace tokens (`hf_...`) → `[REDACTED_TOKEN]`
- User home paths (`/Users/john/...`) → `~/...`

**Example:**
```bash
# Original (unsafe):
Using token hf_AbCdEfGhIjKlMnOpQrStUvWxYz123456 from /Users/john/models

# Logged (safe):
Using token [REDACTED_TOKEN] from ~/models
```


## Configuration Reference

MLX Knife supports comprehensive runtime configuration via environment variables. All settings can be controlled without code changes.

### Feature Gates

Enable experimental and alpha features:

| Variable | Description | Default | Since |
|----------|-------------|---------|-------|
| `MLXK2_ENABLE_ALPHA_FEATURES` | Enable alpha commands (`convert`) | `0` (disabled) | 2.0.0 |
| `MLXK2_ENABLE_PIPES` | Enable Unix pipe integration (`mlxk run <model> -`) | `0` (disabled) | 2.0.4 |
| `MLXK2_EXIF_METADATA` | Extract EXIF metadata from images (Vision models) | `1` (enabled) | 2.0.4 |

**Examples:**
```bash
# Enable pipe mode for stdin processing
export MLXK2_ENABLE_PIPES=1
echo "Hello" | mlxk run model - "translate to Spanish"

# Disable EXIF extraction for privacy (enabled by default)
export MLXK2_EXIF_METADATA=0
mlxk run vision-model --image photo.jpg "describe this"

# Enable alpha features for convert command
export MLXK2_ENABLE_ALPHA_FEATURES=1
mlxk convert ./broken ./fixed --repair-index
```

### Server Configuration

Control server behavior without command-line flags:

| Variable | Description | Default | Since |
|----------|-------------|---------|-------|
| `MLXK2_HOST` | Server bind address | `127.0.0.1` | 2.0.0 |
| `MLXK2_PORT` | Server port | `8000` | 2.0.0 |
| `MLXK2_PRELOAD_MODEL` | Model to load at startup (set by `--model` flag) | (none) | 2.0.0-beta |
| `MLXK2_MAX_TOKENS` | Override default max_tokens for all requests | (auto) | 2.0.4 |
| `MLXK2_RELOAD` | Enable Uvicorn auto-reload (development only) | `0` (disabled) | 2.0.0 |

### Vision Processing

Control vision model behavior (Python 3.10+, beta):

| Variable | Description | Default | Since |
|----------|-------------|---------|-------|
| `MLXK2_VISION_CHUNK_SIZE` | Default chunk size for vision image processing | `1` | 2.0.4-beta.7 |

**Examples:**
```bash
# Process 3 images per chunk instead of 1 (faster but requires more RAM)
export MLXK2_VISION_CHUNK_SIZE=3
mlxk run pixtral --image photos/*.jpg "Describe images"

# CLI flag overrides environment variable
mlxk run pixtral --chunk 5 --image photos/*.jpg "Describe images"  # Uses 5, not 3
```

### Server Configuration Examples

```bash
# Custom host/port binding
MLXK2_HOST=0.0.0.0 MLXK2_PORT=9000 mlxk serve

# Preload model for faster first request
MLXK2_PRELOAD_MODEL="mlx-community/Qwen2.5-3B-Instruct-4bit" mlxk serve

# Override max_tokens for all requests
MLXK2_MAX_TOKENS=4096 mlxk serve

# Development mode with auto-reload
MLXK2_RELOAD=1 mlxk serve
```

### Logging Configuration

Control log output format and verbosity:

| Variable | Description | Default | Since |
|----------|-------------|---------|-------|
| `MLXK2_LOG_JSON` | Enable JSON log format | `0` (text) | 2.0.0 |
| `MLXK2_LOG_LEVEL` | Log level (`debug`, `info`, `warning`, `error`) | `info` | 2.0.0 |

**Examples:**
```bash
# JSON logs for log aggregation tools
MLXK2_LOG_JSON=1 mlxk serve

# Quiet mode (warnings and errors only)
MLXK2_LOG_LEVEL=warning mlxk serve

# Verbose debug output
MLXK2_LOG_LEVEL=debug mlxk serve
```

**Note:** CLI flags (`--log-json`, `--log-level`) take precedence over environment variables.

### HuggingFace Integration

Control HuggingFace Hub authentication and cache:

| Variable | Description | Default | Since |
|----------|-------------|---------|-------|
| `HF_HOME` | HuggingFace cache directory | `~/.cache/huggingface` | N/A |
| `HF_TOKEN` | HuggingFace API token (for private models, `push`) | (none) | N/A |
| `HUGGINGFACE_HUB_TOKEN` | Alternative token variable (fallback) | (none) | N/A |

**Examples:**
```bash
# Custom cache location
HF_HOME=/data/models mlxk list

# Authentication for private models
HF_TOKEN=hf_... mlxk pull org/private-model

# Upload to HuggingFace Hub (requires MLXK2_ENABLE_ALPHA_FEATURES=1)
HF_TOKEN=hf_... mlxk push ./workspace org/model --private
```

### Configuration Priority

When multiple sources define the same setting, precedence order is:

1. **CLI flags** (highest priority) - e.g., `--log-json`, `--port`
2. **Environment variables** - e.g., `MLXK2_LOG_JSON=1`
3. **Defaults** (lowest priority) - documented above

**Example:**
```bash
# CLI flag wins over environment variable
MLXK2_PORT=9000 mlxk serve --port 8080  # Uses port 8080, not 9000
```


## HuggingFace Cache Safety

MLX-Knife 2.0 respects standard HuggingFace cache structure and practices:

### Best Practices for Shared Environments
- **Read operations** (`list`, `health`, `show`) always safe with concurrent processes
- **Write operations** (`pull`, `rm`) coordinate during maintenance windows
- **Lock cleanup** automatic but avoid during active downloads
- **Your responsibility:** Coordinate with team, use good timing

### Example Safe Workflow
```bash
# Check what's in cache (always safe)
mlxk list --json | jq '.data.count'

# Maintenance window - coordinate with team
mlxk rm "corrupted-model" --json --force
mlxk pull "replacement-model" --json

# Back to normal operations
mlxk health --json | jq '.data.summary'
```


## Workspace Features: `clone`, `push`, `convert`

### Workspace Structure

A **workspace** is a self-contained directory containing model files in a flat structure (not the HuggingFace cache format). Workspaces are portable, editable, and can be health-checked standalone.

**Structure:**
```
workspace/
├── config.json              # Model configuration
├── tokenizer.json           # Tokenizer definition
├── tokenizer_config.json    # Tokenizer settings
├── model.safetensors        # Weights (single file)
├── (or model-*.safetensors) # Weights (multi-shard)
└── README.md                # Optional documentation
```

**Key characteristics:**

| Aspect | **Workspace** | **HuggingFace Cache** |
|--------|--------------|----------------------|
| Structure | Flat, self-contained | Nested (hub/models--org--repo/snapshots/...) |
| Models | **Exactly one** model per workspace | Many models (models--org--repo1, models--org--repo2, ...) |
| Purpose | Portable working directory | Download cache (managed) |
| Health Check | Standalone (no cache needed) | Requires cache structure |
| Portability | **Goal:** USB stick, SMB share, any volume | Fixed location (HF_HOME) |
| Ownership | User owns files | Managed by HuggingFace Hub |
| Operations | `clone` (creates), `push` (uploads from) | `pull` (downloads to) |

**Portability (Phase 1 limitation):**
- **Current:** Same APFS volume as cache (CoW optimization)
- **Community Goal:** Any location (USB stick, SMB share, different volumes)
- **Future:** Cross-volume support planned

**Typical workflow:**
1. `mlxk pull org/model` → Downloads to cache
2. `mlxk clone org/model workspace/` → Creates editable workspace copy
3. Edit files in `workspace/` (modify config, quantize, etc.)
4. `mlxk push workspace/ org/new-model` → Upload modified version
5. (Optional) Copy workspace to USB stick for sharing

### `clone` - Model Workspace Creation

`mlxk clone` creates a local workspace from a cached model for modification and development.

- Creates isolated workspace from cached models
- Supports APFS copy-on-write optimization on same-volume scenarios
- Includes health check integration for workspace validation
- Resumable: Interrupted pulls resume automatically
- Use case: Fork-modify-push workflows

Example:
```bash
# Clone model to workspace
mlxk clone org/model ./workspace
```

### `push` - Upload to Hub

`mlxk push` uploads a local folder to a Hugging Face model repository using `huggingface_hub/upload_folder`.

- Requires `HF_TOKEN` (write-enabled).
- Default branch: `main` (explicitly override with `--branch`).
- Safety: `--private` is required to avoid accidental public uploads.
- No validation or manifests. Basic hard excludes are applied by default: `.git/**`, `.DS_Store`, `__pycache__/`, common virtualenv folders (`.venv/`, `venv/`), and `*.pyc`.
- `.hfignore` (gitignore-like) in the workspace is supported and merged with the defaults.
- Repo creation: use `--create` if the target repo does not exist; harmless on existing repos. Missing branches are created during upload.
- JSON output: includes `commit_sha`, `commit_url`, `no_changes`, `uploaded_files_count` (when available), `local_files_count` (approx), `change_summary` and a short `message`.
- Quiet JSON by default: with `--json` (without `--verbose`) progress bars/console logs are suppressed; hub logs are still captured in `data.hf_logs`.
- Human output: derived from JSON; add `--verbose` to include extras such as the commit URL or a short message variant. JSON schema is unchanged.
- Local workspace check: use `--check-only` to validate a workspace without uploading. Produces `workspace_health` in JSON (no token/network required).
- Dry-run planning: use `--dry-run` to compute a plan vs remote without uploading. Returns `dry_run: true`, `dry_run_summary {added, modified:null, deleted}`, and sample `added_files`/`deleted_files`.
- Testing: see TESTING.md ("Push Testing (2.0)") for offline tests and opt-in live checks with markers/env.
- Carefully review the result on the Hub after pushing.
- Responsibility: **You are responsible for complying with Hugging Face Hub policies and applicable laws (e.g., copyright/licensing) for any uploaded content.**

Example:
```bash
# Upload to private repo
mlxk push --private ./workspace org/model --create --commit "init"
```

### `convert` - Workspace Transformations (Experimental)

`mlxk convert` is **experimental** and requires `MLXK2_ENABLE_ALPHA_FEATURES=1`. Currently implements `--repair-index` for fixing safetensors index mismatches (mlx-vlm #624). Future modes like `--quantize` are planned but not yet implemented.

**Use case:** Repair models affected by mlx-vlm #624 conversion bug (7+ mlx-community Vision models).

**Workflow:**
```bash
# Enable alpha features (required for clone)
export MLXK2_ENABLE_ALPHA_FEATURES=1

# Clone affected model to workspace
mlxk clone mlx-community/Qwen2.5-VL-7B-Instruct-4bit ./ws-qwen

# Repair safetensors index (no weights changed)
mlxk convert ./ws-qwen ./ws-qwen-fixed --repair-index

# Verify health
mlxk health ./ws-qwen-fixed  # Should report healthy
```

**Affected models (mlx-vlm #624):**
- Qwen2.5-VL-7B-Instruct-4bit
- gemma-3-27b-it-4bit
- Mistral-Small-3.1-24B-Instruct-2503-4bit
- DeepSeek-OCR-4bit
- Devstral-Small-2-24B-Instruct-2512-6bit
- (7+ models total)

**Key features:**
- **Cache sanctity:** Hard blocks writes to HF cache (workspaces only)
- **Workspace-to-workspace:** Source can be managed or unmanaged, output always managed
- **Health check integration:** Automatic validation (skip with `--skip-health`)
- **APFS CoW:** Instant, space-efficient cloning via `cp -c`

**Future modes:** `--quantize <bits>` (text models), `--dequantize` (planned).

### `pipe mode` - stdin for `run` (beta, `mlx-run` shorthand)

Pipe mode is beta (feature complete) and requires `MLXK2_ENABLE_PIPES=1`. It lets `mlxk run` (and `mlx-run`) read stdin when you pass `-` as the prompt.

- **Status:** Beta (feature complete), API stable (syntax will not change)
- **Gate:** `MLXK2_ENABLE_PIPES=1` (will become default in a future stable release)
- **Auto-batch:** When stdout is a pipe (non-TTY), streaming is disabled automatically for clean output
- **Robust:** Handles SIGPIPE and BrokenPipeError gracefully (`| head`, `| grep -m1` work correctly)
- **Scope:** Applies to `mlxk run` and `mlx-run`; other commands unchanged
- Usage examples (replace `<model>` with a cached MLX chat model):

```bash
# stdin + trailing text (batch when piped)
MLXK2_ENABLE_PIPES=1 echo "from stdin" | mlxk run "<model>" - "append extra context"

# list → run summarization
MLXK2_ENABLE_PIPES=1 mlxk list --json \
  | MLXK2_ENABLE_PIPES=1 mlxk run "<model>" - "Summarize the model list as a concise table." >my-hf-table.md

# Wrapper shorthand
MLXK2_ENABLE_PIPES=1 mlx-run "<model>" - "translate into german" < README.md

# Vision → Text chain: Photo tour review
MLXK2_ENABLE_PIPES=1 mlxk run pixtral --image photos/*.jpg "Describe each picture" \
  | MLXK2_ENABLE_PIPES=1 mlxk run qwen3 - \
    "Write a tour review. Create a table with picture names, metadata, and descriptions." \
  > tour-review.md
```


## Testing

The 2.0 test suite runs by default (pytest discovery points to `tests_2.0/`):

```bash
# Run 2.0 tests (default)
pytest -v

# Explicitly run legacy 1.x tests (not maintained on this branch)
pytest tests/ -v

# Test categories (2.0 example):
# - ADR-002 edge cases
# - Integration scenarios
# - Model naming logic
# - Robustness testing

# Current status: all current 2.0 tests pass (some optional schema tests may be skipped without extras)
```

**Test Architecture:**
- **Isolated Cache System** - Zero risk to user data
- **Atomic Context Switching** - Production/test cache separation
- **Mock Models** - Realistic test scenarios
- **Edge Case Coverage** - All documented failure modes tested


## Compatibility Notes

- Streaming note: Some UIs buffer SSE; verify real-time with `curl -N`. Server sends clear interrupt markers on abort.


## Contributing

This branch follows the established MLX-Knife development patterns:

```bash
# Run quality checks
python test-multi-python.sh  # Tests across Python 3.9-3.14
./run_linting.sh             # Code quality validation

# Key files:
mlxk2/                       # 2.0.0 implementation
tests_2.0/                   # 2.0 test suite
docs/ADR/                    # Architecture decision records
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.


## Support & Feedback

- **Issues**: [GitHub Issues](https://github.com/mzau/mlx-knife/issues)
- **Discussions**: [GitHub Discussions](https://github.com/mzau/mlx-knife/discussions)
- **API Specification**: [JSON API Specification](docs/json-api-specification.md)
- **Documentation**: See `docs/` directory for technical details
- **Security Policy**: See [SECURITY.md](SECURITY.md)


## License

Apache License 2.0 — see `LICENSE` (root) and `mlxk2/NOTICE`.


## Acknowledgments

- Built for Apple Silicon using the [MLX framework](https://github.com/ml-explore/mlx)
- Models hosted by the [MLX Community](https://huggingface.co/mlx-community) on HuggingFace
- Inspired by [ollama](https://ollama.ai)'s user experience

---

<p align="center">
  <b>Made with ❤️ by The BROKE team <img src="broke-logo.png" alt="BROKE Logo" width="30" align="middle"></b><br>
  <i>Version 2.0.4-beta.10 | February 2026</i><br>
  <a href="https://github.com/mzau/broke-nchat">💬 Web UI: nChat - lightweight chat interface</a> •
  <a href="https://github.com/mzau/broke-cluster">🔮 Multi-node: BROKE Cluster</a>
</p>
