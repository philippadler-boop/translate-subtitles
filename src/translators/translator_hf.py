from typing import List, Optional

import srt
from tqdm import tqdm
from transformers import pipeline

from ..config.config_loader import get_from_env_or_json
from ..utils import Progress
from ..subtitles.grouping import SubtitleGrouper


def _default_hf_model_for_pair(source_lang: str, target_lang: str) -> str:
    """
    Very simple mapping for common language pairs.
    Extend this as needed.
    """
    pair = f"{source_lang.lower()}-{target_lang.lower()}"

    mapping = {
        "en-de": "Helsinki-NLP/opus-mt-en-de",
        "de-en": "Helsinki-NLP/opus-mt-de-en",
        "en-fr": "Helsinki-NLP/opus-mt-en-fr",
        "fr-en": "Helsinki-NLP/opus-mt-fr-en",
        # add more pairs here as needed
    }

    if pair not in mapping:
        raise RuntimeError(
            f"[HF] No default model configured for language pair: {pair}. "
            "Set HF_MODEL in .env or config.json to override."
        )

    return mapping[pair]


def translate_subtitles_hf(
    subtitles: List[srt.Subtitle],
    source_lang: str,
    target_lang: str,
    progress: Optional[Progress] = None,
    grouper: Optional[SubtitleGrouper] = None,
) -> List[srt.Subtitle]:
    """
    Translate subtitles using a HuggingFace translation pipeline.
    Defaults to a model based on source/target, or uses HF_MODEL.
    """
    model_name = get_from_env_or_json("HF_MODEL")
    if not model_name:
        model_name = _default_hf_model_for_pair(source_lang, target_lang)

    # If HF_MODEL is provided in config but appears to target the opposite
    # language pair (common mistake: `opus-mt-en-de` vs `opus-mt-de-en`),
    # auto-correct and warn the user.
    if model_name:
        mn_lower = model_name.lower()
        src = source_lang.lower()
        tgt = target_lang.lower()

        if "en-de" in mn_lower and src.startswith("de") and tgt.startswith("en"):
            corrected = model_name.replace("en-de", "de-en")
            print(
                f"[HF] Warning: HF_MODEL '{model_name}' looks like an en->de model; "
                f"switching to '{corrected}' for de->en translation."
            )
            model_name = corrected

        elif "de-en" in mn_lower and src.startswith("en") and tgt.startswith("de"):
            corrected = model_name.replace("de-en", "en-de")
            print(
                f"[HF] Warning: HF_MODEL '{model_name}' looks like a de->en model; "
                f"switching to '{corrected}' for en->de translation."
            )
            model_name = corrected

    print(f"[HF] Using model: {model_name}")
    translator = pipeline("translation", model=model_name)

    translated: List[srt.Subtitle] = []
    def _append(orig: srt.Subtitle, text: str):
        translated.append(
            srt.Subtitle(
                index=orig.index,
                start=orig.start,
                end=orig.end,
                content=text,
                proprietary=orig.proprietary,
            )
        )

    # Grouped flow
    if grouper is not None:
        groups = grouper.group(subtitles)
        if progress is not None:
            progress.start(total=len(groups))

        for g in groups:
            try:
                out = translator(g.text)
                result = out[0]["translation_text"]
            except Exception as e:
                print(f"[HF] Error on group starting at {g.start}: {e}")
                # fall back to per-line
                for orig in g.originals:
                    try:
                        out = translator(orig.content)
                        single = out[0]["translation_text"]
                    except Exception:
                        single = orig.content
                    _append(orig, single)
                if progress is not None:
                    progress.update(1, f"group {g.indices[0]}")
                continue

            parts = grouper.distribute(result, g.originals)
            for orig, part in zip(g.originals, parts):
                _append(orig, part or orig.content)

            if progress is not None:
                progress.update(1, f"group {g.indices[0]}")

        if progress is not None:
            progress.finish("translation complete")
        return translated

    # Ungrouped fallback (original behavior)
    if progress is not None:
        progress.start(total=len(subtitles))
        for sub in subtitles:
            text = sub.content
            try:
                out = translator(text)
                result = out[0]["translation_text"]
            except Exception as e:
                print(f"[HF] Error on subtitle {sub.index}: {e}")
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
        return translated

    for sub in tqdm(subtitles, desc="HF Translating", unit="line"):
        text = sub.content
        try:
            out = translator(text)
            result = out[0]["translation_text"]
        except Exception as e:
            print(f"[HF] Error on subtitle {sub.index}: {e}")
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

    return translated
