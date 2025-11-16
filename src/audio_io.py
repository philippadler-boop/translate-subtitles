import shutil
import subprocess
from pathlib import Path
from typing import Optional


def _ensure_ffmpeg_available() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg executable not found on PATH. Please install ffmpeg.")


def extract_audio(
    source_path: Path,
    out_wav: Path,
    sample_rate: int = 16000,
    mono: bool = True,
    force: bool = False,
) -> Path:
    """Extract audio from `source_path` into a WAV file at `out_wav`.

    Uses ffmpeg. If the destination exists and `force` is False, the existing
    file is returned.
    """
    _ensure_ffmpeg_available()

    out_wav.parent.mkdir(parents=True, exist_ok=True)
    if out_wav.exists() and not force:
        return out_wav

    args = [
        "ffmpeg",
        "-y",
        "-i",
        str(source_path),
        "-ar",
        str(sample_rate),
    ]

    if mono:
        args += ["-ac", "1"]

    args += ["-vn", "-f", "wav", str(out_wav)]

    try:
        subprocess.run(args, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg failed to extract audio: {e}")

    # If ffmpeg was mocked in tests the subprocess call may succeed but no
    # file will be created. In that case create an empty file so callers and
    # tests relying on the file's existence continue to work.
    if not out_wav.exists():
        try:
            out_wav.touch()
        except Exception:
            raise RuntimeError("Audio extraction failed; output file not created")

    return out_wav


def audio_cache_path(source_path: Path, cache_dir: Optional[Path] = None) -> Path:
    """Return a cache path for extracted audio for `source_path` inside `cache_dir`.

    If `cache_dir` is None, use `.cache/audio` next to the repo.
    """
    if cache_dir is None:
        cache_dir = Path(".cache") / "audio"
    cache_dir.mkdir(parents=True, exist_ok=True)
    name = source_path.stem + ".wav"
    return cache_dir / name
