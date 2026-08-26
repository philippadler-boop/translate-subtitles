import time
from typing import List, Optional

import srt
import deepl
from tqdm import tqdm

from ..config.config_loader import get_from_env_or_json
from .translator_google import translate_subtitles_google
from ..subtitles.grouping import SubtitleGrouper
from ..utils import Progress


MAX_RETRIES = 5  # retry attempts for DeepL rate-limit/high-load errors


def translate_subtitles_deepl(
    subtitles: List[srt.Subtitle],
    source_lang: str,
    target_lang: str,
    progress: Optional[Progress] = None,
    grouper: Optional[SubtitleGrouper] = None,
) -> List[srt.Subtitle]:
    """
    Translate subtitles using DeepL API with retry + exponential backoff.

    Behavior:
    - If no DeepL API key exists → fall back to Google immediately.
    - If key is invalid → fall back to Google.
    - If DeepL rate-limits ("Too many requests") → retry up to MAX_RETRIES.
    - If still failing after retries → fall back to Google for the remaining subtitles.
    """
    api_key = get_from_env_or_json("DEEPL_API_KEY")

    # 1) No API key → use Google
    if not api_key:
        print("[DeepL] No DEEPL_API_KEY found. Falling back to Google.")
        return translate_subtitles_google(subtitles, source_lang, target_lang)

    # 2) Validate key eagerly
    try:
        translator = deepl.Translator(api_key)
        translator.get_usage()  # tests key validity
    except deepl.AuthorizationException as e:
        print(f"[DeepL] Authorization failed: {e}")
        print("[DeepL] Falling back to Google.")
        return translate_subtitles_google(subtitles, source_lang, target_lang)
    except deepl.DeepLException as e:
        print(f"[DeepL] Error initializing DeepL: {e}")
        print("[DeepL] Falling back to Google.")
        return translate_subtitles_google(subtitles, source_lang, target_lang)

    # Prepare DeepL language codes
    src = None if source_lang in ("auto", "", None) else source_lang.upper()
    tgt = target_lang.upper()

    translated: List[srt.Subtitle] = []
    # === MAIN TRANSLATION LOOP ===
    # If a grouper is provided, translate groups as blocks and distribute
    # results back to original subtitles. Otherwise fall back to per-line.
    if grouper is not None:
        groups = grouper.group(subtitles)

        if progress is not None:
            progress.start(total=len(groups))

        for gi, g in enumerate(groups):
            text = g.text

            # RETRY / BACKOFF LOOP for the group
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    result_obj = translator.translate_text(
                        text,
                        source_lang=src,
                        target_lang=tgt,
                    )
                    result = result_obj.text
                    break  # success
                except deepl.TooManyRequestsException:
                    wait_seconds = 2 ** (attempt - 1)
                    print(
                        f"[DeepL] Too many requests (group {g.indices[0]} attempt {attempt}/{MAX_RETRIES}). "
                        f"Retrying in {wait_seconds} seconds..."
                    )
                    time.sleep(wait_seconds)
                except deepl.DeepLException as e:
                    # Other DeepL errors → stop retrying this group and fallback
                    print(f"[DeepL] Error on group starting at {g.start}: {e}")
                    result = None
                    break
            else:
                # Retries exhausted for this group
                print(f"[DeepL] DeepL still overloaded after retries for group starting at {g.start}.")
                result = None

            if result is None:
                # Fallback: translate originals of this group via Google per-line
                try:
                    google_rest = translate_subtitles_google(
                        g.originals,
                        source_lang=source_lang,
                        target_lang=target_lang,
                        progress=None,
                        grouper=None,
                    )
                    # append google_rest to translated
                    translated.extend(google_rest)
                    if progress is not None:
                        progress.update(1, f"group {g.indices[0]} (google fallback)")
                    continue
                except Exception as e:
                    # If google fallback fails, preserve originals
                    print(f"[DeepL] Google fallback failed for group at {g.start}: {e}")
                    for orig in g.originals:
                        translated.append(
                            srt.Subtitle(
                                index=orig.index,
                                start=orig.start,
                                end=orig.end,
                                content=orig.content,
                                proprietary=orig.proprietary,
                            )
                        )
                    if progress is not None:
                        progress.update(1, f"group {g.indices[0]} (preserved)")
                    continue

            # Distribute translated group text back to originals
            parts = grouper.distribute(result, g.originals)
            for orig, part in zip(g.originals, parts):
                translated.append(
                    srt.Subtitle(
                        index=orig.index,
                        start=orig.start,
                        end=orig.end,
                        content=part or orig.content,
                        proprietary=orig.proprietary,
                    )
                )

            if progress is not None:
                progress.update(1, f"group {g.indices[0]}")

        if progress is not None:
            progress.finish("translation complete")
        return translated

    # No grouper: per-line behavior (existing implementation)
    if progress is not None:
        progress.start(total=len(subtitles))
        for idx, sub in enumerate(subtitles):
            text = sub.content

            # RETRY / BACKOFF LOOP
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    result_obj = translator.translate_text(
                        text,
                        source_lang=src,
                        target_lang=tgt,
                    )
                    result = result_obj.text
                    break  # success
                except deepl.TooManyRequestsException:
                    # DeepL is rate-limiting us, so we wait and retry
                    wait_seconds = 2 ** (attempt - 1)
                    print(
                        f"[DeepL] Too many requests (attempt {attempt}/{MAX_RETRIES}). "
                        f"Retrying in {wait_seconds} seconds..."
                    )
                    time.sleep(wait_seconds)
                except deepl.DeepLException as e:
                    # Other DeepL errors → stop retrying this line
                    print(f"[DeepL] Error on subtitle {sub.index}: {e}")
                    result = text  # keep original line
                    break
            else:
                # We only hit this 'else' if all retries failed for rate-limiting
                print("[DeepL] DeepL still overloaded after retries.")
                print("[DeepL] Falling back to Google for remaining subtitles...")

                remaining = subtitles[idx:]
                google_rest = translate_subtitles_google(
                    remaining,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    progress=None,
                    grouper=grouper,
                )
                translated.extend(google_rest)
                progress.finish("fallback to google")
                return translated

            # Append the successfully translated (or fallback) subtitle
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

    # fallback to original tqdm-based behavior
    for idx, sub in enumerate(tqdm(subtitles, desc="DeepL Translating", unit="line")):
        text = sub.content

        # RETRY / BACKOFF LOOP
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                result_obj = translator.translate_text(
                    text,
                    source_lang=src,
                    target_lang=tgt,
                )
                result = result_obj.text
                break  # success
            except deepl.TooManyRequestsException:
                # DeepL is rate-limiting us, so we wait and retry
                wait_seconds = 2 ** (attempt - 1)
                print(
                    f"[DeepL] Too many requests (attempt {attempt}/{MAX_RETRIES}). "
                    f"Retrying in {wait_seconds} seconds..."
                )
                time.sleep(wait_seconds)
            except deepl.DeepLException as e:
                # Other DeepL errors → stop retrying this line
                print(f"[DeepL] Error on subtitle {sub.index}: {e}")
                result = text  # keep original line
                break
        else:
            # We only hit this 'else' if all retries failed for rate-limiting
            print("[DeepL] DeepL still overloaded after retries.")
            print("[DeepL] Falling back to Google for remaining subtitles...")

            remaining = subtitles[idx:]
            google_rest = translate_subtitles_google(
                    remaining,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    grouper=grouper,
                )
            translated.extend(google_rest)
            return translated

        # Append the successfully translated (or fallback) subtitle
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
