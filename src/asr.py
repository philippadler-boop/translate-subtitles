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
        raise RuntimeError("faster-whisper is not installed. Please install it to use local ASR.")

    device = _choose_device(device)
    model = WhisperModel(model_name, device=device, compute_type=compute_type)

    segments = []
    # faster-whisper model.transcribe yields segments with start/end/text
    for segment in model.transcribe(str(wav_path)):
        # segment is expected to be an object/dict with .start .end .text
        seg = {
            "start": float(segment["start"]) if isinstance(segment, dict) and "start" in segment else float(segment.start),
            "end": float(segment["end"]) if isinstance(segment, dict) and "end" in segment else float(segment.end),
            "text": segment["text"] if isinstance(segment, dict) and "text" in segment else str(segment.text),
        }
        segments.append(seg)

    return {"segments": segments}
