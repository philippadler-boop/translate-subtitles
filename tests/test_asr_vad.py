import wave
import math
from pathlib import Path
from unittest.mock import patch

from src.asr.asr import transcribe_with_vad


def _write_short_tone(path: Path, sr: int = 16000):
    # write 0.2s silence, 0.5s tone, 0.2s silence
    samples = []
    for is_tone, duration in [(False, 0.2), (True, 0.5), (False, 0.2)]:
        n = int(sr * duration)
        if is_tone:
            for i in range(n):
                t = i / sr
                v = int(32767 * 0.3 * math.sin(2 * math.pi * 440 * t))
                samples.append(v)
        else:
            samples.extend([0] * n)

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(
            b"".join(int(s).to_bytes(2, "little", signed=True) for s in samples)
        )


def test_transcribe_with_vad_calls_chunk_transcriber(tmp_path):
    wav = tmp_path / "short.wav"
    _write_short_tone(wav)

    # Patch the low-level transcribe_with_whisper to return a fake segment for any chunk
    fake_chunk_result = {"segments": [{"start": 0.0, "end": 0.5, "text": "hello"}]}

    with patch("src.asr.transcribe_with_whisper") as mock_trans:
        mock_trans.return_value = fake_chunk_result

        out = transcribe_with_vad(wav, model_name="tiny", device="cpu")

    # Expect combined segments with times shifted by chunk start (first speech starts ~0.2s)
    assert "segments" in out
    segs = out["segments"]
    assert len(segs) >= 1
    # First combined segment should start after ~0.15-0.25s
    assert 0.1 < segs[0]["start"] < 0.4
    assert segs[0]["text"] == "hello"
