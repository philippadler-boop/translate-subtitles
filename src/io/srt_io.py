from pathlib import Path
from typing import Iterable, List

import srt


def read_srt_file(path: Path) -> List[srt.Subtitle]:
    """Read an .srt file and return a list of Subtitle objects."""
    text = path.read_text(encoding="utf-8")
    return list(srt.parse(text))


def write_srt_file(subtitles: Iterable[srt.Subtitle], path: Path) -> None:
    """Write a list of Subtitle objects to an .srt file."""
    # Sanitize input: `srt.compose` assumes each subtitle has a non-None
    # `content` string. Some translators or upstream code may produce
    # `None` for content; normalize to empty string here to avoid a
    # runtime AttributeError when srt tries to call `.strip()`.
    sanitized = []
    for sub in subtitles:
        try:
            # If it's already an srt.Subtitle, ensure content is a string.
            if isinstance(sub, srt.Subtitle):
                if sub.content is None:
                    sub.content = ""
                else:
                    sub.content = str(sub.content)
                sanitized.append(sub)
            else:
                # Accept dict-like objects with start/end/text or content keys
                start = getattr(sub, "start", None) or sub.get("start")
                end = getattr(sub, "end", None) or sub.get("end")
                text = getattr(sub, "content", None) or sub.get("text") or ""
                # Convert numeric seconds to timedelta if necessary
                try:
                    s_start = (
                        start
                        if isinstance(start, srt.timedelta)
                        else srt.timedelta(seconds=float(start))
                    )
                except Exception:
                    s_start = srt.timedelta(seconds=0)
                try:
                    s_end = (
                        end
                        if isinstance(end, srt.timedelta)
                        else srt.timedelta(seconds=float(end))
                    )
                except Exception:
                    s_end = srt.timedelta(seconds=0)

                sanitized.append(
                    srt.Subtitle(index=getattr(sub, "index", None) or sub.get("index"), start=s_start, end=s_end, content=str(text))
                )
        except Exception:
            # Skip malformed entries rather than failing the whole write.
            continue

    text = srt.compose(sanitized)
    path.write_text(text, encoding="utf-8")
