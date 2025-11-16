import wave
import math
from pathlib import Path

import pytest

from src.vad import get_speech_segments


def _write_tone_wav(path: Path, durations: list, freq: int = 440, sr: int = 16000):
    """Write a mono 16-bit WAV composed of alternating silence/tone blocks.

    `durations` is a list of (is_tone: bool, seconds: float)
    """
    samples = []
    for is_tone, duration in durations:
        n = int(sr * duration)
        if is_tone:
            for i in range(n):
                t = i / sr
                v = int(32767 * 0.5 * math.sin(2 * math.pi * freq * t))
                samples.append(v)
        else:
            samples.extend([0] * n)

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(b"".join(int(s).to_bytes(2, "little", signed=True) for s in samples))


def test_get_speech_segments_detects_tone(tmp_path):
    p = tmp_path / "test.wav"
    # layout: silence 0.3s, tone 0.6s, silence 0.4s, tone 0.5s, silence 0.3s
    layout = [(False, 0.3), (True, 0.6), (False, 0.4), (True, 0.5), (False, 0.3)]
    _write_tone_wav(p, layout)

    segs = get_speech_segments(p, aggressiveness=2)
    # Expect at least one speech segment
    assert len(segs) >= 1

    # Check segments durations and approximate positions
    # First tone should start after ~0.3s
    first = segs[0]
    assert pytest.approx(0.3, rel=0.3) == first.start
    assert first.end - first.start > 0.3

    # If multiple segments detected, ensure later segments start near the expected position
    if len(segs) > 1:
        last = segs[-1]
        assert pytest.approx(1.3, rel=0.3) == last.start
