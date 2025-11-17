from pathlib import Path
from typing import Dict, List

try:
    # transformers pipeline for ASR
    from transformers import pipeline as _hf_pipeline
    _TRANSFORMERS_AVAILABLE = True
except Exception:  # pragma: no cover - optional
    _hf_pipeline = None
    _TRANSFORMERS_AVAILABLE = False


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
    model_name: str = "openai/whisper-small",
    device: str = "auto",
    compute_type: str = "float32",
    model_instance=None,
) -> Dict[str, List[dict]]:
    """Transcribe `wav_path` using the Hugging Face `transformers` pipeline.

    Returns a dict: {"segments": [ {"start": float, "end": float, "text": str}, ... ] }

    Note: the transformers ASR pipeline typically returns a single transcription
    without fine-grained timestamps. We preserve the return format by returning
    a single segment that covers the whole file.
    """
    if not _TRANSFORMERS_AVAILABLE:
        raise RuntimeError("transformers package is required for ASR but is not available in the environment.")

    device_choice = _choose_device(device)
    # pipeline expects device index (0-based) for CUDA or -1 for CPU
    hf_device = 0 if device_choice == "cuda" else -1

    # If a pre-created pipeline (model_instance) is provided, reuse it.
    if model_instance is not None:
        pipe = model_instance
    else:
        try:
            pipe = _hf_pipeline("automatic-speech-recognition", model=model_name, device=hf_device)
        except Exception:
            pipe = None

    if pipe is None:
        raise RuntimeError("Failed to initialize transformers ASR pipeline with model '%s'" % model_name)

    # determine duration for a coarse end timestamp
    try:
        import wave as _wave

        with _wave.open(str(wav_path), "rb") as _wf:
            frames = _wf.getnframes()
            rate = _wf.getframerate()
            duration = frames / float(rate) if rate else 0.0
    except Exception:
        duration = 0.0

    result = pipe(str(wav_path))
    if isinstance(result, dict):
        text = result.get("text", "")
    else:
        text = str(result)

    return {"segments": [{"start": 0.0, "end": float(duration), "text": text}]}


def transcribe_with_vad(
    wav_path: Path,
    model_name: str = "small",
    device: str = "auto",
    compute_type: str = "float32",
    aggressiveness: int = 2,
    max_workers: int = 1,
) -> Dict[str, List[dict]]:
    """Run VAD to split audio and transcribe per-segment.

    This function uses :func:`get_speech_segments` to find speech regions,
    writes temporary WAV chunks, runs :func:`transcribe_with_whisper` on each
    chunk, and adjusts the timestamps to the original audio timeline.
    """
    from tempfile import NamedTemporaryFile
    import os
    import wave
    from .vad import get_speech_segments

    segments_out: List[dict] = []

    segs = get_speech_segments(wav_path, aggressiveness=aggressiveness)

    if not segs:
        # fallback: transcribe whole file directly
        return transcribe_with_whisper(
            wav_path,
            model_name=model_name,
            device=device,
            compute_type=compute_type,
        )

    # Prepare a single transformers pipeline instance to reuse across chunks
    model_instance = None
    if _TRANSFORMERS_AVAILABLE:
        try:
            model_device = _choose_device(device)
            hf_device = 0 if model_device == "cuda" else -1
            model_instance = _hf_pipeline("automatic-speech-recognition", model=model_name, device=hf_device)
        except Exception:
            # If pipeline creation fails here, transcribe_with_whisper will raise per-chunk.
            model_instance = None

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
            try:
                with wave.open(tmp_name, "wb") as out_wf:
                    out_wf.setnchannels(n_channels)
                    out_wf.setsampwidth(sampwidth)
                    out_wf.setframerate(framerate)
                    out_wf.writeframes(frames)

                # transcribe the chunk
                chunk_result = transcribe_with_whisper(
                    Path(tmp_name),
                    model_name=model_name,
                    device=device,
                    compute_type=compute_type,
                    model_instance=model_instance,
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
            finally:
                try:
                    os.remove(tmp_name)
                except OSError:
                    pass

    # sort segments by start
    segments_out.sort(key=lambda x: x["start"])
    return {"segments": segments_out}
