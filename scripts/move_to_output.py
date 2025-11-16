#!/usr/bin/env python3
"""Move processed files into `workspaces/output/{video,audio,subs}` preserving filenames.

Usage: python scripts/move_to_output.py <source_path>
If source_path is a file, it will be moved to the matching output subfolder based on extension.
If source_path is a directory, all files inside will be moved accordingly.
"""
import shutil
from pathlib import Path
import sys


EXT_TO_SUBFOLDER = {
    ".mp4": "video",
    ".mkv": "video",
    ".wav": "audio",
    ".mp3": "audio",
    ".srt": "subs",
}


def move_file_to_output(src: Path) -> None:
    ext = src.suffix.lower()
    sub = EXT_TO_SUBFOLDER.get(ext)
    if not sub:
        print(f"Skipping unknown extension: {src}")
        return
    dst_dir = Path("workspaces") / "output" / sub
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    shutil.move(str(src), str(dst))
    print(f"Moved {src} -> {dst}")


def main(argv):
    if len(argv) < 2:
        print("Usage: move_to_output.py <path>")
        return 1
    p = Path(argv[1])
    if p.is_file():
        move_file_to_output(p)
    elif p.is_dir():
        for f in p.iterdir():
            if f.is_file():
                move_file_to_output(f)
    else:
        print(f"Path not found: {p}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
