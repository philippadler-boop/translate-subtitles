from dataclasses import dataclass
from typing import List, Optional

import srt


@dataclass
class SubtitleGroup:
    indices: List[int]
    start: float
    end: float
    text: str
    originals: List[srt.Subtitle]


class SubtitleGrouper:
    """Heuristic subtitle grouper (Approach A).

    - Groups adjacent `srt.Subtitle` items into larger segments using:
      * punctuation-aware merging (end on sentence-ending punctuation)
      * maximum characters per group
      * maximum duration per group

    This class is intentionally small and designed to be extended for
    more advanced approaches (B..E).
    """

    def __init__(
        self,
        max_chars: int = 200,
        max_duration: float = 6.0,
        group_by_punctuation: bool = True,
    ):
        self.max_chars = max_chars
        self.max_duration = max_duration
        self.group_by_punctuation = group_by_punctuation

    def group(self, subtitles: List[srt.Subtitle]) -> List[SubtitleGroup]:
        groups: List[SubtitleGroup] = []
        if not subtitles:
            return groups

        cur_indices: List[int] = []
        cur_texts: List[str] = []
        cur_start = subtitles[0].start.total_seconds()
        cur_end = subtitles[0].end.total_seconds()

        def flush():
            if not cur_indices:
                return
            group = SubtitleGroup(
                indices=list(cur_indices),
                start=cur_start,
                end=cur_end,
                text=" ".join(cur_texts).strip(),
                originals=[subtitles[i] for i in cur_indices],
            )
            groups.append(group)

        for idx, sub in enumerate(subtitles):
            text = sub.content.strip()
            if not cur_indices:
                # start new group
                cur_indices = [idx]
                cur_texts = [text]
                cur_start = sub.start.total_seconds()
                cur_end = sub.end.total_seconds()
                continue

            # calculate prospective sizes
            prospective_text = " ".join(cur_texts + [text]).strip()
            prospective_len = len(prospective_text)
            prospective_end = sub.end.total_seconds()
            prospective_duration = prospective_end - cur_start

            end_sentence = any(text.endswith(p) for p in (".", "?", "!"))

            should_flush = False
            if prospective_len > self.max_chars:
                should_flush = True
            if prospective_duration > self.max_duration:
                should_flush = True
            if self.group_by_punctuation and end_sentence:
                # prefer to end on punctuation when present and group is non-empty
                should_flush = True

            if should_flush:
                # flush current group and start a new one with this subtitle
                flush()
                cur_indices = [idx]
                cur_texts = [text]
                cur_start = sub.start.total_seconds()
                cur_end = sub.end.total_seconds()
            else:
                # append to current group
                cur_indices.append(idx)
                cur_texts.append(text)
                cur_end = sub.end.total_seconds()

        # final flush
        if cur_indices:
            group = SubtitleGroup(
                indices=list(cur_indices),
                start=cur_start,
                end=cur_end,
                text=" ".join(cur_texts).strip(),
                originals=[subtitles[i] for i in cur_indices],
            )
            groups.append(group)

        return groups

    def distribute(self, translated_text: str, originals: List[srt.Subtitle]) -> List[str]:
        """Split a translated block back into pieces aligned with `originals`.

        This simple implementation splits by word counts proportionally to
        the original texts' word counts. It's a best-effort heuristic suitable
        for Approach A and for later refinement.
        """
        if not originals:
            return []

        orig_texts = [s.content.strip() for s in originals]
        orig_word_counts = [len(t.split()) for t in orig_texts]
        total_words = sum(orig_word_counts)
        if total_words == 0:
            # fallback: evenly split by number of originals
            parts = [translated_text.strip()] + [""] * (len(originals) - 1)
            return parts

        translated_words = translated_text.strip().split()
        result_parts: List[str] = []
        widx = 0
        for count in orig_word_counts:
            if count <= 0:
                result_parts.append("")
                continue
            take = max(1, round((count / total_words) * len(translated_words)))
            part_words = translated_words[widx : widx + take]
            result_parts.append(" ".join(part_words))
            widx += take

        # If any words remain, append to the last part
        if widx < len(translated_words):
            remaining = translated_words[widx:]
            if result_parts:
                result_parts[-1] = (result_parts[-1] + " " + " ".join(remaining)).strip()
            else:
                result_parts = [" ".join(remaining)]

        # Ensure we have exactly len(originals) parts
        if len(result_parts) < len(originals):
            result_parts.extend([""] * (len(originals) - len(result_parts)))
        elif len(result_parts) > len(originals):
            # merge extras into last
            extras = result_parts[len(originals) - 1 :]
            result_parts = result_parts[: len(originals) - 1]
            result_parts.append(" ".join(extras).strip())

        return [p.strip() for p in result_parts]
