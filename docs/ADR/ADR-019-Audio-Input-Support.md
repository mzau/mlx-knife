# ADR-019 — Audio Input Support (via mlx-vlm)

**Status:** In Progress (Phase 1-3 done, Phase 4 pending)
**Target:** 2.0.4-beta.8
**Depends on:** mlx-vlm ≥0.3.10 (GitHub only, not yet on PyPI)

---

## Context

mlx-vlm documents audio input for Omni models in main branch (post-0.3.9):
- **Gemma-3n** (Google): Vision + Audio + Text — explicitly documented in README

**Not Supported:**
- **Qwen3-Omni** (Alibaba): Architecture `qwen3_omni_moe` not in mlx-lm/mlx-vlm
  - See: github.com/ml-explore/mlx-lm/issues/497
  - Community PR #574 is text-only (Audio-Tower removed)

These models use the same mlx-vlm backend as vision models.
mlx-knife can add audio support with minimal effort.

**Current State (2026-01-19):**
- mlx-vlm 0.3.10 available on GitHub, not yet on PyPI
- pyproject.toml uses Git URL dependency (blocks PyPI release)
- Audio functionality verified with Gemma-3n

---

## Decision

Implement audio input via mlx-vlm (Option A from Session 101 discussion).

### Scope: CLI-first

| Command | Audio Support |
|---------|---------------|
| `list` | Shows `+audio` in type field |
| `show` | Shows audio capability in details |
| `run` | `--audio file.wav` parameter |
| `health` | Checks audio-capable models |
| `serve` | Phase 4 - pending |

### Out of Scope

- STT/Transcription (separate feature via mlx-audio-plus, see Brainstorm ADR)
- TTS/Audio output (Qwen3-Omni can do it, but not exposed)
- stdin audio (`--audio -`) - later
- Audio segments/timestamps (STT feature)

---

## Design

### CLI Interface

```bash
# Audio-only (transcription) - uses default prompt and temperature 0.2
mlxk run gemma-3n-E2B-it-4bit --audio voice.wav

# With explicit prompt
mlxk run gemma-3n-E2B-it-4bit --audio voice.wav --prompt "Transcribe exactly what is said"

# Lower temperature for maximum consistency
mlxk run gemma-3n-E2B-it-4bit --audio voice.wav --temperature 0
```

### Capability Detection

```python
# capabilities.py
AUDIO_MODEL_TYPES = frozenset({
    "gemma3n",        # Google Gemma 3n (verified in mlx-vlm README)
    # "qwen3_omni_moe", # Alibaba Qwen3-Omni — NOT SUPPORTED (mlx-lm #497)
})

# Detection via config.json:
# - model_type in AUDIO_MODEL_TYPES
# - "audio_config" present
```

### list/show Output

```
$ mlxk list
NAME                      TYPE                 HEALTH   SIZE
gemma-3n-E2B-it-4bit     chat+vision+audio    healthy  2.1GB
pixtral-12b-4bit         chat+vision          healthy  6.8GB
llama-3.2-3b-4bit        chat                 healthy  1.8GB
```

### Code Changes (Phase 1-2)

| File | Change |
|------|--------|
| `cli.py` | `--audio` argument, context-aware temperature default |
| `run.py` | Audio passthrough to VisionRunner, capability checks |
| `vision_runner.py` | `_prepare_audio()`, `audio=` parameter, `num_audios=` |
| `capabilities.py` | `AUDIO_MODEL_TYPES`, `Capability.AUDIO` |
| `common.py` | `detect_audio_capability()` |

---

## Implementation Phases

### Phase 1: Capability Detection ✅

- `AUDIO_MODEL_TYPES` in capabilities.py
- `is_audio` detection in ModelCapabilities (via `audio_config` in config.json)
- `list`/`show` shows audio capability
- Consistent with vision capability display

### Phase 2: CLI Run ✅

- `--audio` argument in cli.py
- VisionRunner audio passthrough
- Error handling (audio without audio-capable model → clear error)
- 5MB file size limit (~2-3 min at 16kHz mono; token count is the real constraint)
- Multi-audio blocked with clear error (mlx-vlm limitation)
- Context-aware temperature default (0.2 for audio, 0.7 for text/vision)
- Default prompt: "Transcribe what is spoken in this audio."

### Phase 3: Evaluation ✅

Tested with `gemma-3n-E2B-it-4bit` (mlx-community).

**Findings:**

| Question | Result |
|----------|--------|
| Empty prompt allowed? | Yes, but results poor (model confused) |
| Audio-only without image? | ✅ Works well |
| Text prompt required? | Not required, but recommended for quality |
| Multiple audio files? | ❌ Not supported (mlx-vlm token mismatch bug) |
| Audio + Vision combined? | ❌ Audio silently ignored when images present (cause unclear: mlx-vlm or model) |

**Temperature Impact:**

| Temperature | Behavior |
|-------------|----------|
| 0.7 (text default) | Multilingual drift (outputs Arabic/Hindi for "Amen" misinterpretation) |
| 0.2 (audio default) | Stable, good transcription quality |
| 0.0 | Most consistent, recommended for STT tasks |

**Implemented Defaults:**
- Audio temperature: 0.2 (vs 0.7 for text/vision)
- Audio prompt: "Transcribe what is spoken in this audio."

**Known Limitations:**
- Phonetic errors observed ("A man" → "Amen"); cause unclear (model limitation vs quantization)
- Multi-audio causes token mismatch error in mlx-vlm
- Audio+Vision combined: audio silently ignored (not blocked by mlx-knife to allow future compatibility)
- WAV format confirmed; other formats untested

**Audio Duration Limit (~30 seconds):**

Gemma-3n uses a fixed **188 soft tokens per audio clip**, regardless of audio length:
- Encoding rate: 6.25 tokens/second (from model spec)
- Maximum duration: 188 ÷ 6.25 = **~30.08 seconds**

| Audio Duration | Transcribed | Notes |
|----------------|-------------|-------|
| ≤29 seconds | Full | ✅ Reliable |
| 30-35 seconds | ~29 seconds | Missing last few seconds |
| 43 seconds | ~29 seconds | Only first 2/3 transcribed |

**Cause:** Model architecture constraint in Gemma-3n. Audio is encoded to a fixed-size
representation before being passed to the language model. Content beyond ~30 seconds
is silently dropped during encoding.

**Sources:**
- `Gemma3nProcessor.audio_seq_length = 188` (transformers 5.0+)
- [HuggingFace Gemma3n Docs](https://huggingface.co/docs/transformers/en/model_doc/gemma3n)
- Empirical testing (Session 116)

### Phase 4: Server API (Pending)

> **Note:** This design may evolve based on client ecosystem developments.

**Goal:** OpenAI-compatible audio input for `/v1/chat/completions`

**OpenAI Standard Format (`input_audio`):**
```json
{
  "model": "gemma-3n-E2B-it-4bit",
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "Transcribe this audio"},
      {
        "type": "input_audio",
        "input_audio": {
          "data": "<base64-encoded-wav>",
          "format": "wav"
        }
      }
    ]
  }]
}
```

**Two Approaches for Audio Input:**

| Approach | How it works | Client Support |
|----------|--------------|----------------|
| STT → LLM | Audio → separate Whisper service → Text → LLM | Open WebUI, many clients |
| Native Audio | Audio directly to LLM (GPT-4o-audio, Gemma-3n) | New, limited client support |

mlx-knife uses **native audio** (Gemma-3n understands audio directly).
Most clients (Open WebUI) expect separate STT service, not `input_audio`.

**WebUI Integration (nChat):**

Browser provides native APIs for microphone access:
```javascript
// Microphone access
navigator.mediaDevices.getUserMedia({ audio: true })

// Recording
const mediaRecorder = new MediaRecorder(stream)
mediaRecorder.ondataavailable = (e) => { /* audio blob */ }
```

Workflow: Record → Convert to WAV (JS library) → Base64 → JSON `input_audio`

**Implementation TODO:**
- Parse `input_audio` content blocks in server_base.py
- Base64 decode → temp file → VisionRunner
- 5MB limit enforcement (base64 ~33% overhead)
- Error handling for unsupported formats
- Live tests: `test_audio_server_e2e.py`

---

## Risks

| Risk | Mitigation |
|------|------------|
| mlx-vlm 0.3.10 not on PyPI | Git dependency in pyproject.toml; blocks mlx-knife PyPI release |
| mlx-vlm API changes | Thin wrapper, quickly adaptable |
| Audio format restrictions | Documented: WAV confirmed, others untested |
| Memory requirements unclear | Document in README, profile before server support |
| Gemma-3n index/shard issue | `mlxk convert --repair-index` before use |
| Qwen3-Omni not supported | Only Gemma-3n as target, Qwen3-Omni out-of-scope |
| Phonetic errors | Document limitation, recommend temperature 0 |

---

## References

- mlx-vlm README (main): Multi-Modal Example (Image + Audio)
- mlx-lm Issue #497: Qwen3-Omni Support Request (open since Sep 2025)
- mlx-lm PR #574: qwen3_omni_moe Text-only (Audio-Tower removed)
- Gemma-3n: mlx-community/gemma-3n-E2B-it-4bit
- Session 101: Audio discussion, Option A decision
- Session 102: Research — Qwen3-Omni blocked, Gemma-3n verified
- Session 111: Phase 3 evaluation, temperature findings, limitation documentation
