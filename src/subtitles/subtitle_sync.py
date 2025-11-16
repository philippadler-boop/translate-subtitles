from typing import List

import srt


def _split_text_into_chunks(text: str, max_chars: int) -> List[str]:
    words = text.split()
    chunks = []
    cur = []
    cur_len = 0
    for w in words:
        if cur_len + len(w) + (1 if cur else 0) <= max_chars:
            cur.append(w)
            cur_len += len(w) + (1 if cur else 0)
        else:
            if cur:
                chunks.append(" ".join(cur))
            cur = [w]
            cur_len = len(w)
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def generate_srt_from_asr(
    asr_output: dict, max_chars: int = 42, min_duration: float = 0.5
) -> List[srt.Subtitle]:
    """Generate a list of `srt.Subtitle` from ASR output segments.

    `asr_output` is expected to be a dict with key "segments" containing dicts with
    `start`, `end`, `text`.
    """
    subs: List[srt.Subtitle] = []

    idx = 1
    for seg in asr_output.get("segments", []):
        start = float(seg["start"])
        end = float(seg["end"])
        text = seg.get("text", "").strip()
        if not text:
            continue

        chunks = _split_text_into_chunks(text, max_chars=max_chars)
        if len(chunks) == 1:
            duration = max(min_duration, end - start)
            subs.append(
                srt.Subtitle(
                    index=idx,
                    start=srt.timedelta(seconds=start),
                    end=srt.timedelta(seconds=start + duration),
                    content=chunks[0],
                )
            )
            idx += 1
        else:
            # distribute timing proportionally to chunk lengths
            total_chars = sum(len(c) for c in chunks)
            if total_chars == 0:
                continue
            cursor = start
            for c in chunks:
                portion = len(c) / total_chars
                dur = max(min_duration, portion * (end - start))
                subs.append(
                    srt.Subtitle(
                        index=idx,
                        start=srt.timedelta(seconds=cursor),
                        end=srt.timedelta(seconds=cursor + dur),
                        content=c,
                    )
                )
                idx += 1
                cursor += dur

    # Reindex and return
    for i, s in enumerate(subs, start=1):
        s.index = i
    return subs
