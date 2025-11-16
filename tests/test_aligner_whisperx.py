import sys
from pathlib import Path

import importlib

from src.align.aligner import align_with_whisperx


def test_align_with_whisperx_monkeypatched(tmp_path, monkeypatch):
    # Create a fake whisperx module and inject into sys.modules
    class FakeModel:
        def transcribe(self, path):
            return {"segments": [{"start": 0.0, "end": 1.0, "text": "hello world"}], "language": "en"}

    class FakeWhisperX:
        def load_model(self, name, device=None):
            return FakeModel()

        def load_align_model(self, lang, device=None):
            # return fake align model and metadata
            return ("align_model", {"align": True})

        def align(self, segments, align_model, metadata, wav_path, device=None):
            # return segments with words list
            return {"segments": [{"start": 0.0, "end": 1.0, "text": "hello world", "words": [{"word": "hello", "start": 0.0, "end": 0.5},{"word": "world","start":0.5,"end":1.0}]}]}

    monkeypatch.setitem(sys.modules, "whisperx", FakeWhisperX())

    wav = tmp_path / "dummy.wav"
    wav.write_bytes(b"RIFF----WAVEfmt ")

    aligned = align_with_whisperx(wav, [{"start": 0, "end": 1, "text": "hello world"}], device="cpu")
    assert isinstance(aligned, list)
    assert aligned[0]["words"][0]["word"] == "hello"
