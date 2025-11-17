from pathlib import Path
import wave

import pytest

from src.asr.asr import transcribe_with_whisper


def _make_wav(path: Path, duration_s: float = 0.1, framerate: int = 16000):
    n_frames = int(duration_s * framerate)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(framerate)
        wf.writeframes(b"\x00\x00" * n_frames)


def test_transcribe_with_whisper_short_duration_monkeypatched(monkeypatch, tmp_path):
    wav = tmp_path / "short.wav"
    _make_wav(wav, duration_s=0.1, framerate=8000)

    # Monkeypatch transformers availability and pipeline factory
    monkeypatch.setattr("src.asr.asr._TRANSFORMERS_AVAILABLE", True)

    class FakePipe:
        def __call__(self, path, **kwargs):
            return {"text": "transcribed text"}

    def fake_factory(task, model, device):
        return FakePipe()

    monkeypatch.setattr("src.asr.asr._hf_pipeline", fake_factory)

    out = transcribe_with_whisper(wav, model_name="openai/whisper-tiny", device="cpu")
    assert "segments" in out
    assert out["segments"][0]["text"] == "transcribed text"


def test_transcribe_with_whisper_long_duration_returns_segments(monkeypatch, tmp_path):
    wav = tmp_path / "long.wav"
    # create a small-framerate file that appears long (31s) without huge data
    _make_wav(wav, duration_s=31, framerate=1)

    monkeypatch.setattr("src.asr.asr._TRANSFORMERS_AVAILABLE", True)

    class FakePipeSegments:
        def __call__(self, path, **kwargs):
            # return structure with segments
            return {"segments": [{"start": 0.0, "end": 1.0, "text": "seg"}]}

    def fake_factory(task, model, device):
        return FakePipeSegments()

    monkeypatch.setattr("src.asr.asr._hf_pipeline", fake_factory)

    out = transcribe_with_whisper(wav, model_name="openai/whisper-tiny", device="cpu")
    assert "segments" in out
    assert out["segments"][0]["text"] == "seg"
