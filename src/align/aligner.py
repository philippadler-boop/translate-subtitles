from pathlib import Path
from typing import List, Dict, Any


def _choose_device(device: str) -> str:
    if device in ("auto", None):
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return device


def align_with_whisperx(
    wav_path: Path, asr_segments: List[Dict[str, Any]], device: str = "auto"
) -> List[Dict[str, Any]]:
    """Attempt to run WhisperX forced-alignment on the given audio.

    Parameters
    - wav_path: path to the mono 16-bit WAV file
    - asr_segments: list of segments produced by ASR (dicts with 'start','end','text')
    - device: 'auto'|'cpu'|'cuda'

    Returns a list of segments enhanced with word-level timings where available.

    This wrapper is defensive: if `whisperx` is not installed or the API
    is incompatible, it raises a RuntimeError with installation guidance.
    """
    dev = _choose_device(device)

    try:
        import whisperx
    except Exception as e:
        raise RuntimeError(
            "whisperx is required for forced alignment. Install it following the README or: `pip install whisperx`"
        ) from e

    # Try a couple of plausible usage patterns from whisperx API since
    # versions may vary. We keep the code defensive and provide a helpful
    # error if alignment cannot be performed.
    try:
        # load ASR model inside whisperx (may be optional if caller used other ASR)
        model = whisperx.load_model("small", device=dev)
        result = model.transcribe(str(wav_path))

        # load align model (whisperx uses a small alignment model)
        align_model, metadata = whisperx.load_align_model(
            result.get("language", result.get("lang", "en")), device=dev
        )

        # perform alignment
        result_aligned = whisperx.align(
            result["segments"], align_model, metadata, str(wav_path), device=dev
        )

        # result_aligned may contain word-level timings in 'segments'->'words'
        aligned_segments = []
        for seg in result_aligned.get("segments", []):
            aligned_segments.append(seg)

        return aligned_segments

    except Exception:
        # Try alternate calling convention used by other versions
        try:
            result = whisperx.transcribe(str(wav_path), device=dev)
            align_model, metadata = whisperx.load_align_model(
                result.get("language", "en"), device=dev
            )
            result_aligned = whisperx.align(
                result, align_model, metadata, str(wav_path), device=dev
            )
            return result_aligned.get("segments", [])
        except Exception as e:
            raise RuntimeError(
                "whisperx alignment failed. Ensure whisperx is installed and compatible with this wrapper."
            ) from e
