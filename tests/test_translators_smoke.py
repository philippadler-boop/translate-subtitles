import srt
from datetime import timedelta

from src.translators import translator_deepl, translator_gpt


def _sample_subtitles():
    return [
        srt.Subtitle(index=1, start=timedelta(seconds=0), end=timedelta(seconds=1), content="Hello")
    ]


def test_deepl_falls_back_to_google_when_no_api_key(monkeypatch):
    subs = _sample_subtitles()

    # Simulate missing DEEPL_API_KEY
    monkeypatch.setattr("src.translators.translator_deepl.get_from_env_or_json", lambda k: None)

    # Provide a fake Google translator to capture call
    def fake_google(subtitles, source_lang, target_lang, progress=None):
        # return modified subtitles to signal fallback path used
        out = []
        for s in subtitles:
            out.append(srt.Subtitle(index=s.index, start=s.start, end=s.end, content=s.content + " (g)", proprietary=s.proprietary))
        return out

    monkeypatch.setattr("src.translators.translator_deepl.translate_subtitles_google", fake_google)

    res = translator_deepl.translate_subtitles_deepl(subs, source_lang="auto", target_lang="DE")
    assert res[0].content.endswith(" (g)")


def test_gpt_raises_when_api_key_missing(monkeypatch):
    subs = _sample_subtitles()
    monkeypatch.setattr("src.translators.translator_gpt.get_from_env_or_json", lambda k: None)

    try:
        translator_gpt.translate_subtitles_gpt(subs, source_lang="en", target_lang="de")
    except RuntimeError as e:
        assert "OPENAI_API_KEY" in str(e)
    else:
        raise AssertionError("Expected RuntimeError when OPENAI_API_KEY is missing")
