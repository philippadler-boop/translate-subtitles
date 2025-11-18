from typing import List, Optional

import srt
from tqdm import tqdm
from openai import OpenAI

from ..config.config_loader import get_from_env_or_json
from ..utils import Progress
from ..subtitles.grouping import SubtitleGrouper


def translate_subtitles_gpt(
    subtitles: List[srt.Subtitle],
    source_lang: str,
    target_lang: str,
    model: str = "gpt-4o-mini",
    progress: Optional[Progress] = None,
    grouper: Optional[SubtitleGrouper] = None,
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
    def _append_sub(orig: srt.Subtitle, content: str):
        translated.append(
            srt.Subtitle(
                index=orig.index,
                start=orig.start,
                end=orig.end,
                content=content,
                proprietary=orig.proprietary,
            )
        )

    # Grouped flow if grouper provided
    if grouper is not None:
        groups = grouper.group(subtitles)
        if progress is not None:
            progress.start(total=len(groups))

        for g in groups:
            prompt = (
                "Translate the following subtitle text.\n"
                "Requirements:\n"
                f"- Source language: {source_lang or 'auto'}\n"
                f"- Target language: {target_lang}\n"
                "- Preserve line breaks.\n"
                "- Do not add explanations, only return the translated text.\n\n"
                f"Subtitle:\n{g.text}"
            )

            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1500,
                )
                # openai client shape: choices[0].message.content
                result = resp.choices[0].message.content.strip()
            except Exception as e:
                print(f"[GPT] Error on group starting at {g.start}: {e}")
                # fallback: translate each original line individually
                for orig in g.originals:
                    try:
                        resp = client.chat.completions.create(
                            model=model,
                            messages=[{"role": "user", "content": (
                                "Translate the following subtitle text.\n"
                                "Requirements:\n"
                                f"- Source language: {source_lang or 'auto'}\n"
                                f"- Target language: {target_lang}\n"
                                "- Do not add explanations.\n\n"
                                f"Subtitle:\n{orig.content}") }],
                            max_tokens=500,
                        )
                        single = resp.choices[0].message.content.strip()
                    except Exception:
                        single = orig.content
                    _append_sub(orig, single)
                if progress is not None:
                    progress.update(1, f"group {g.indices[0]}")
                continue

            parts = grouper.distribute(result, g.originals)
            for orig, part in zip(g.originals, parts):
                _append_sub(orig, part or orig.content)

            if progress is not None:
                progress.update(1, f"group {g.indices[0]}")

        if progress is not None:
            progress.finish("translation complete")
        return translated

    # Ungrouped per-line behavior (original)
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
