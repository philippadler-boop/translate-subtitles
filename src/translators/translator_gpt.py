from typing import List, Optional

import srt
from tqdm import tqdm
from openai import OpenAI

from ..config_loader import get_from_env_or_json
from ..progress import Progress


def translate_subtitles_gpt(
    subtitles: List[srt.Subtitle],
    source_lang: str,
    target_lang: str,
    model: str = "gpt-4o-mini",
    progress: Optional[Progress] = None,
) -> List[srt.Subtitle]:
    """
    Translate subtitles using GPT-4o / GPT-4o-mini.
    Requires OPENAI_API_KEY in .env or config.json.
    """
    api_key = get_from_env_or_json("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "[GPT] OPENAI_API_KEY not set in .env or config.json. "
            "Cannot use GPT translator."
        )

    client = OpenAI(api_key=api_key)

    translated: List[srt.Subtitle] = []

    # Use our Progress helper when provided
    if progress is not None:
        progress.start(total=len(subtitles))
        for sub in subtitles:
            text = sub.content

            prompt = (
                "Translate the following subtitle text.\n"
                "Requirements:\n"
                f"- Source language: {source_lang or 'auto'}\n"
                f"- Target language: {target_lang}\n"
                "- Preserve line breaks.\n"
                "- Do not add explanations, only return the translated text.\n\n"
                f"Subtitle:\n{text}"
            )

            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500,
                )
                result = resp.choices[0].message.content.strip()
            except Exception as e:
                print(f"[GPT] Error on subtitle {sub.index}: {e}")
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
            progress.update(1, f"line {sub.index}")

        progress.finish("translation complete")
        return translated

    # fallback to tqdm-based loop
    for sub in tqdm(subtitles, desc="GPT Translating", unit="line"):
        text = sub.content

        prompt = (
            "Translate the following subtitle text.\n"
            "Requirements:\n"
            f"- Source language: {source_lang or 'auto'}\n"
            f"- Target language: {target_lang}\n"
            "- Preserve line breaks.\n"
            "- Do not add explanations, only return the translated text.\n\n"
            f"Subtitle:\n{text}"
        )

        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
            )
            result = resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"[GPT] Error on subtitle {sub.index}: {e}")
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
