"""Guard the STT load path's alias rejection (mlx-audio#870).

mlx-audio 0.4.6 moved resampling out of its Kaiser-windowed-sinc filter into the
decoder, whose stopband is far too shallow for an ASR front-end. Upstream's own
regression test stayed green throughout, because it calls `resample_audio`
directly and the change was to *who calls it* — so this test deliberately goes
through the real entry point instead of the function.

It is equally deliberate that the tones are synthetic: every audio asset in this
repository is 16 kHz mono, where no resampling happens at all. Material that
cannot expose the defect is why the regression survived a release cycle here.

No model, no network, no GPU — it lives under live/ only because
`mlx_audio.stt.utils` imports the real `mlx.core` at module level, which
TESTING-DETAILS forbids inside the stub-collecting tree.
"""

import struct
import tempfile
import wave
from pathlib import Path

import pytest

np = pytest.importorskip("numpy", reason="numpy required for the resample guard")
pytest.importorskip("mlx_audio", reason="mlx-audio required for the resample guard")

pytestmark = [pytest.mark.live, pytest.mark.wet]

SRC_RATE = 44100
DST_RATE = 16000
DURATION = 2.0
AMPLITUDE = 0.5
FADE_S = 0.05
TRIM_S = 0.15

PASSBAND = (1000, 3000)
STOPBAND = (8500, 12000)
STOPBAND_MAX_DB = -90.0
PASSBAND_MIN_DB = -1.0


def _write_tone(path: Path, freq: int) -> None:
    """A faded tone: a step at t=0 is broadband and would be mistaken for leakage."""
    n = int(SRC_RATE * DURATION)
    sig = AMPLITUDE * np.sin(2 * np.pi * freq * np.arange(n) / SRC_RATE)

    fade = int(SRC_RATE * FADE_S)
    ramp = 0.5 * (1 - np.cos(np.pi * np.arange(fade) / fade))
    sig[:fade] *= ramp
    sig[-fade:] *= ramp[::-1]

    pcm = (sig * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SRC_RATE)
        handle.writeframes(struct.pack(f"<{len(pcm)}h", *pcm))


def _peak_magnitude(samples: np.ndarray) -> float:
    trim = int(DST_RATE * TRIM_S)
    seg = samples[trim:-trim] if len(samples) > 2 * trim else samples
    return float(np.abs(np.fft.rfft(seg * np.hanning(len(seg)))).max())


@pytest.fixture(scope="module")
def tone_levels() -> dict:
    """Peak level of each tone after the real load path, in dB relative to 1 kHz."""
    from mlx_audio.stt.utils import load_audio

    levels = {}
    with tempfile.TemporaryDirectory() as tmp:
        for freq in PASSBAND + STOPBAND:
            path = Path(tmp) / f"{freq}.wav"
            _write_tone(path, freq)
            loaded = np.asarray(load_audio(str(path), sr=DST_RATE), dtype=np.float64)
            levels[freq] = _peak_magnitude(loaded)

    reference = levels[PASSBAND[0]]
    return {f: 20 * np.log10(max(m, 1e-15) / reference) for f, m in levels.items()}


@pytest.mark.parametrize("freq", STOPBAND)
def test_energy_above_target_nyquist_is_rejected(tone_levels, freq):
    """Below Nyquist the tone must be gone, not merely attenuated."""
    level = tone_levels[freq]
    assert level <= STOPBAND_MAX_DB, (
        f"{freq} Hz survived {SRC_RATE}->{DST_RATE} Hz at {level:.1f} dB "
        f"(limit {STOPBAND_MAX_DB} dB). The load path is not band-limiting; "
        f"aliases fold into the speech band. See mlx-audio#870."
    )


@pytest.mark.parametrize("freq", PASSBAND)
def test_passband_survives(tone_levels, freq):
    """The counterweight: a filter that rejects everything would also pass the above."""
    level = tone_levels[freq]
    assert level >= PASSBAND_MIN_DB, (
        f"{freq} Hz lost {abs(level):.1f} dB through {SRC_RATE}->{DST_RATE} Hz "
        f"resampling — the passband is being attenuated, not just the stopband."
    )
