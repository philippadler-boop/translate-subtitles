from pathlib import Path
from typing import Dict, List

try:
    # faster-whisper is optional; if not installed the import will fail at runtime
    from faster_whisper import WhisperModel
except Exception:  # pragma: no cover - import availability depends on environment
    WhisperModel = None


def _choose_device(device: str) -> str:
    if device in ("auto", None):
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return device


def transcribe_with_whisper(
    wav_path: Path,
    model_name: str = "small",
    device: str = "auto",
    compute_type: str = "float32",
) -> Dict[str, List[dict]]:
    """Transcribe `wav_path` using faster-whisper and return segments.

    Returns a dict: {"segments": [ {"start": float, "end": float, "text": str}, ... ] }
    """
    if WhisperModel is None:
        raise RuntimeError(
            "faster-whisper is not installed. Please install it to use local ASR."
        )

    device = _choose_device(device)
    model = WhisperModel(model_name, device=device, compute_type=compute_type)

    segments = []
    # faster-whisper model.transcribe may return different shapes depending on version:
    # - an iterable of segment-like objects/dicts
    # - a dict with a 'segments' key
    # - nested generators; handle these robustly
    result = model.transcribe(str(wav_path))

    # If result is a mapping with 'segments'
    if isinstance(result, dict) and "segments" in result:
        raw_segments = result["segments"]
    else:
        # Attempt to iterate and flatten nested generators
        raw_segments = []
        try:
            for item in result:
                # item may itself be a segment dict/object, or a generator yielding segments
                if isinstance(item, dict) and ("start" in item or "text" in item):
                    raw_segments.append(item)
                else:
                    # try to iterate sub-items
                    try:
                        for sub in item:
                            if isinstance(sub, dict) and (
                                "start" in sub or "text" in sub
                            ):
                                raw_segments.append(sub)
                    except TypeError:
                        # not iterable, ignore
                        pass
        except TypeError:
            # not iterable; fall back to empty
            raw_segments = []

    for segment in raw_segments:
        # segment is expected to be an object/dict with .start .end .text
        if isinstance(segment, dict):
            start = float(segment.get("start", segment.get("begin", 0.0)))
            end = float(segment.get("end", segment.get("finish", start)))
            text = segment.get("text", "")
        else:
            # object-like
            start = float(getattr(segment, "start", 0.0))
            end = float(getattr(segment, "end", getattr(segment, "finish", start)))
            text = str(getattr(segment, "text", ""))

        seg = {"start": start, "end": end, "text": text}
        segments.append(seg)

    return {"segments": segments}


def transcribe_with_vad(
    wav_path: Path,
    model_name: str = "small",
    device: str = "auto",
    compute_type: str = "float32",
    aggressiveness: int = 2,
    max_workers: int = 1,
) -> Dict[str, List[dict]]:
    """Run VAD to split audio and transcribe per-segment.

    This function uses `src.vad.get_speech_segments` to find speech regions,
    writes temporary WAV chunks, runs `transcribe_with_whisper` on each chunk,
    and adjusts the timestamps to the original audio timeline.
    """
    from tempfile import NamedTemporaryFile
    import wave
    from .vad import get_speech_segments

    segments_out: List[dict] = []

    segs = get_speech_segments(wav_path, aggressiveness=aggressiveness)

    if not segs:
        # fallback: transcribe whole file
        import importlib

        root_asr = importlib.import_module("src.asr")
        return root_asr.transcribe_with_whisper(
            wav_path, model_name=model_name, device=device, compute_type=compute_type
        )

    # open source wave for slicing
    with wave.open(str(wav_path), "rb") as src_wf:
        n_channels = src_wf.getnchannels()
        sampwidth = src_wf.getsampwidth()
        framerate = src_wf.getframerate()

        for s in segs:
            start_frame = int(round(s.start * framerate))
            end_frame = int(round(s.end * framerate))
            src_wf.setpos(start_frame)
            frames = src_wf.readframes(max(0, end_frame - start_frame))

            # write to temp wav
            with NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_name = tmp.name
            with wave.open(tmp_name, "wb") as out_wf:
                out_wf.setnchannels(n_channels)
                out_wf.setsampwidth(sampwidth)
                out_wf.setframerate(framerate)
                out_wf.writeframes(frames)

            # transcribe the chunk (call through the root shim so tests can patch it)
            import importlib

            root_asr = importlib.import_module("src.asr")
            chunk_result = root_asr.transcribe_with_whisper(
                Path(tmp_name),
                model_name=model_name,
                device=device,
                compute_type=compute_type,
            )
            # adjust timestamps
            for seg in chunk_result.get("segments", []):
                segments_out.append(
                    {
                        "start": float(seg["start"]) + s.start,
                        "end": float(seg["end"]) + s.start,
                        "text": seg.get("text", ""),
                    }
                )

    # sort segments by start
    segments_out.sort(key=lambda x: x["start"])
    return {"segments": segments_out}
