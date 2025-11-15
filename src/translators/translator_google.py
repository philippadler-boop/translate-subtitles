from typing import List

import srt
from deep_translator import GoogleTranslator
from tqdm import tqdm


def translate_subtitles_google(
    subtitles: List[srt.Subtitle],
    source_lang: str,
    target_lang: str,
) -> List[srt.Subtitle]:
    """
    Translate subtitles using GoogleTranslator from deep-translator.
    No API key needed (web-based).
    """
    # deep-translator wants short codes like "en", "de", etc.
    src = None if source_lang in ("auto", "", None) else source_lang.lower()
    tgt = target_lang.lower()

    translator = GoogleTranslator(source=src or "auto", target=tgt)

    translated: List[srt.Subtitle] = []

    for sub in tqdm(subtitles, desc="Google Translating", unit="line"):
        text = sub.content
        try:
            result = translator.translate(text)
        except Exception as e:
            print(f"[Google] Error on subtitle {sub.index}: {e}")
            result = text  # fallback

        translated.append(
            srt.Subtitle(
                index=sub.index,
                start=sub.start,
                end=sub.end,
                content=result,
                proprietary=sub.proprietary,
            )
        )

    return translated
