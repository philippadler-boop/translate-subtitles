from types import SimpleNamespace
from pathlib import Path
import srt


def test_run_regeneration_pipeline_basic(monkeypatch, tmp_path):
    # Create a fake input file
    inp = tmp_path / "movie.mkv"
    inp.write_text("fake")

    # Prepare fake args
    args = SimpleNamespace(
        src_lang="en",
        asr_model="small",
        device="cpu",
        align=False,
        align_method=None,
        export_words=None,
        visualize=False,
    )

    # Monkeypatch extract_audio to return a wav path
    fake_wav = tmp_path / "movie.wav"
    fake_wav.write_bytes(b"RIFF....WAVE")
    monkeypatch.setattr("src.io.extract_audio", lambda inp_path, out_path: fake_wav)

    # Monkeypatch ASR return value
    monkeypatch.setattr("src.asr.asr.transcribe_with_vad", lambda *a, **k: {"segments": [{"start": 0.0, "end": 1.0, "text": "hello"}]})

    # Monkeypatch generate_srt_from_asr to return real srt subtitles
    def fake_generate(asr_out):
        return [srt.Subtitle(index=1, start=srt.timedelta(seconds=0), end=srt.timedelta(seconds=1), content="hello")]

    monkeypatch.setattr("src.subtitles.subtitle_sync.generate_srt_from_asr", fake_generate)

    # Capture write_srt_file call
    called = {}

    def fake_write(subs, out_path):
        called['out'] = Path(out_path)
        called['subs'] = subs

    monkeypatch.setattr("src.cli.write_srt_file", fake_write)

    # Run the regeneration helper
    from src.cli import _run_regeneration_pipeline

    subtitles = _run_regeneration_pipeline(args, inp)

    assert isinstance(subtitles, list)
    assert 'out' in called
    assert called['out'].name.endswith('.en.srt')


def test_run_regeneration_with_alignment_and_export(monkeypatch, tmp_path):
    inp = tmp_path / "movie.mkv"
    inp.write_text("fake")

    args = SimpleNamespace(
        src_lang="auto",
        asr_model="small",
        device="cpu",
        align=True,
        align_method="whisperx",
        export_words=str(tmp_path / "words.json"),
        visualize=True,
    )

    fake_wav = tmp_path / "movie.wav"
    fake_wav.write_bytes(b"RIFF....WAVE")
    monkeypatch.setattr("src.io.extract_audio", lambda inp_path, out_path: fake_wav)

    # transcribe returns raw segments
    monkeypatch.setattr("src.asr.asr.transcribe_with_vad", lambda *a, **k: {"segments": [{"start": 0.0, "end": 1.0, "text": "hello"}]})

    # align_with_whisperx should be called and produce aligned segments
    monkeypatch.setattr("src.align.aligner.align_with_whisperx", lambda wav, segs, device=None: segs)

    # visualizer functions
    monkeypatch.setattr("src.visualizer.visualizer.extract_words_from_aligned_segments", lambda segs: [{"word":"hello"}])
    monkeypatch.setattr("src.visualizer.visualizer.write_words_json", lambda words, out: out.write_text('[]'))

    monkeypatch.setattr("src.subtitles.subtitle_sync.generate_srt_from_asr", lambda asr_out: [srt.Subtitle(index=1, start=srt.timedelta(seconds=0), end=srt.timedelta(seconds=1), content="hello")])

    called = {}

    def fake_write(subs, out_path):
        called['out'] = Path(out_path)
        called['subs'] = subs

    monkeypatch.setattr("src.cli.write_srt_file", fake_write)

    from src.cli import _run_regeneration_pipeline

    subtitles = _run_regeneration_pipeline(args, inp)

    assert isinstance(subtitles, list)
    # export_words should have been written (we monkeypatched write_words_json to write '[]')
    exported = Path(args.export_words)
    assert exported.exists()
    assert exported.read_text(encoding='utf-8') == '[]'
    # visualization HTML is no longer produced; JSON export is sufficient.
    html_out = exported.with_suffix('.html')
    assert not html_out.exists()
