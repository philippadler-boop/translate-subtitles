from typing import List, Optional

import srt
from deep_translator import GoogleTranslator
from tqdm import tqdm

from ..utils import Progress
from ..subtitles.grouping import SubtitleGrouper


def translate_subtitles_google(
    subtitles: List[srt.Subtitle],
    source_lang: str,
    target_lang: str,
    progress: Optional[Progress] = None,
    grouper: Optional[SubtitleGrouper] = None,
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

    def _append_sub(idx: int, orig: srt.Subtitle, content: str):
        translated.append(
            srt.Subtitle(
                index=orig.index,
                start=orig.start,
                end=orig.end,
                content=content,
                proprietary=orig.proprietary,
            )
        )

    # Grouped flow (if a grouper is provided)
    if grouper is not None:
        groups = grouper.group(subtitles)
        iterator = groups if progress is None else groups
        if progress is not None:
            progress.start(total=len(groups))

        for g in iterator:
            try:
                result = translator.translate(g.text)
            except Exception as e:
                # if translation of a whole group fails, fall back to translating
                # each line individually to maximize robustness
                print(f"[Google] Error on group starting at {g.start}: {e}")
                for orig in g.originals:
                    try:
                        r = translator.translate(orig.content)
                    except Exception:
                        r = orig.content
                    _append_sub(orig.index, orig, r)
                if progress is not None:
                    progress.update(1, f"group {g.indices[0]}")
                continue

            # distribute translated text back to original subtitles
            parts = grouper.distribute(result, g.originals)
            for orig, part in zip(g.originals, parts):
                _append_sub(orig.index, orig, part or orig.content)

            if progress is not None:
                progress.update(1, f"group {g.indices[0]}")

        if progress is not None:
            progress.finish("translation complete")
        return translated

    # Ungrouped (original) per-line behavior
    if progress is not None:
        progress.start(total=len(subtitles))
        for sub in subtitles:
            text = sub.content
            try:
                result = translator.translate(text)
            except Exception as e:
                print(f"[Google] Error on subtitle {sub.index}: {e}")
                result = text

            translated.append(
                srt.Subtitle(
                    index=sub.index,
                    start=sub.start,
                    end=sub.end,
                    content=result,
                    proprietary=sub.proprietary,
                )
            )
            progress.update(1, f"line {sub.index}")
        progress.finish("translation complete")
    else:
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
