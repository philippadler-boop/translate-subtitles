from typing import List

import srt
from tqdm import tqdm
from transformers import pipeline

from ..config_loader import get_from_env_or_json


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
) -> List[srt.Subtitle]:
    """
    Translate subtitles using a HuggingFace translation pipeline.
    Defaults to a model based on source/target, or uses HF_MODEL.
    """
    model_name = get_from_env_or_json("HF_MODEL")
    if not model_name:
        model_name = _default_hf_model_for_pair(source_lang, target_lang)

    print(f"[HF] Using model: {model_name}")
    translator = pipeline("translation", model=model_name)

    translated: List[srt.Subtitle] = []

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
