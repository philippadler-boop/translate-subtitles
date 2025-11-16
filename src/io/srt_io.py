from pathlib import Path
from typing import Iterable, List

import srt


def read_srt_file(path: Path) -> List[srt.Subtitle]:
    """Read an .srt file and return a list of Subtitle objects."""
    text = path.read_text(encoding="utf-8")
    return list(srt.parse(text))


def write_srt_file(subtitles: Iterable[srt.Subtitle], path: Path) -> None:
    """Write a list of Subtitle objects to an .srt file."""
    text = srt.compose(subtitles)
    path.write_text(text, encoding="utf-8")
