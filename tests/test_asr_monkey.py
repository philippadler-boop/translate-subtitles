import wave
from pathlib import Path

import pytest

from src.asr.asr import transcribe_with_vad


def _make_silent_wav(path: Path, duration_s: float = 0.1, framerate: int = 16000):
    # create a tiny silent wav file; content won't be used because we monkeypatch
    n_frames = int(duration_s * framerate)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(framerate)
        wf.writeframes(b"\x00\x00" * n_frames)


def test_transcribe_with_vad_falls_back_to_whisper_when_no_segments(monkeypatch, tmp_path):
    wav = tmp_path / "test.wav"
    _make_silent_wav(wav)

    # Force VAD to report no speech segments so transcribe_with_vad calls transcribe_with_whisper
    monkeypatch.setattr("src.asr.vad.get_speech_segments", lambda *a, **k: [])

    # Monkeypatch the heavy transcribe function to return a predictable result
    def fake_transcribe(wav_path, **kwargs):
        return {"segments": [{"start": 0.0, "end": 0.1, "text": "hello world"}]}

    monkeypatch.setattr("src.asr.asr.transcribe_with_whisper", fake_transcribe)

    out = transcribe_with_vad(wav, model_name="openai/whisper-tiny", device="cpu")

    assert isinstance(out, dict)
    assert "segments" in out
    assert out["segments"] and out["segments"][0]["text"] == "hello world"
