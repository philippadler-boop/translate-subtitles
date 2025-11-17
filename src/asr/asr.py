from pathlib import Path
from typing import Dict, List
import concurrent.futures

try:
    # transformers pipeline for ASR
    from transformers import pipeline as _hf_pipeline
    _TRANSFORMERS_AVAILABLE = True
except Exception:  # pragma: no cover - optional
    _hf_pipeline = None
    _TRANSFORMERS_AVAILABLE = False

# Globals used by worker processes
_WORKER_PIPE = None


def _worker_init(model_name: str, device: str, language: str | None = None):
    """Initializer for worker processes: create a module-global pipeline instance.

    This avoids reloading the model for every chunk submitted to the worker.
    Workers run in separate processes and load the pipeline on CPU (device '-1')
    unless explicitly passed a CUDA device index (not recommended for multiple
    workers on a single GPU).
    """
    global _WORKER_PIPE
    try:
        # Import inside worker to avoid pickling heavy objects
        from transformers import pipeline as _local_pipeline

        # If device provided is 'cuda' try to use GPU index 0, but keep CPU default
        if device and device.startswith("cuda"):
            hf_device = 0
        else:
            hf_device = -1

        _WORKER_PIPE = _local_pipeline("automatic-speech-recognition", model=model_name, device=hf_device)
    except Exception:
        _WORKER_PIPE = None


def _worker_transcribe(tmp_wav_path: str, language: str | None = None) -> dict:
    """Transcribe a single temporary WAV file using the worker-global pipeline.

    Returns a dict matching the per-chunk return value: {"segments": [...]}
    """
    global _WORKER_PIPE
    if _WORKER_PIPE is None:
        raise RuntimeError("Worker pipeline not initialized")

    # For longer chunks, Whisper models require timestamp prediction when
    # inputs exceed the short-form length; enabling `return_timestamps=True`
    # ensures the model returns segment timestamps when available.
    if language:
        res = _WORKER_PIPE(str(tmp_wav_path), language=language, return_timestamps=True)
    else:
        res = _WORKER_PIPE(str(tmp_wav_path), return_timestamps=True)

    # Normalize to dict-like with 'text' or 'segments'
    if isinstance(res, dict) and "segments" in res:
        return {"segments": res["segments"]}
    if isinstance(res, dict) and "text" in res:
        return {"segments": [{"start": 0.0, "end": 0.0, "text": res.get("text", "")}]} 
    return {"segments": [{"start": 0.0, "end": 0.0, "text": str(res)}]}


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
    language: str | None = None,
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
            # suppress noisy device-setting prints from transformers/torch internals
            import contextlib, os

            with open(os.devnull, "w") as _devnull:
                with contextlib.redirect_stdout(_devnull), contextlib.redirect_stderr(_devnull):
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

    # Pass language to pipeline call when provided to force transcription language
    import contextlib, os
    # If the audio is long (> ~30s) the Whisper model will enable long-form
    # generation and require timestamp tokens; request timestamps in that case
    # so we get per-segment timing information back from the pipeline.
    need_timestamps = duration > 30.0

    with open(os.devnull, "w") as _devnull:
        with contextlib.redirect_stdout(_devnull), contextlib.redirect_stderr(_devnull):
            if language:
                result = pipe(str(wav_path), language=language, return_timestamps=need_timestamps)
            else:
                result = pipe(str(wav_path), return_timestamps=need_timestamps)

    # If the pipeline returned segments (when return_timestamps=True), normalize
    # them to the expected output format. Otherwise fall back to a single segment
    # spanning the entire file.
    if isinstance(result, dict) and "segments" in result:
        out_segments = []
        for seg in result.get("segments", []):
            out_segments.append(
                {
                    "start": float(seg.get("start", 0.0)),
                    "end": float(seg.get("end", 0.0)),
                    "text": seg.get("text", ""),
                }
            )
        return {"segments": out_segments}

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
    language: str | None = None,
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
            language=language,
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
    temp_chunks = []  # (tmp_name, seg.start, seg.end)
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

            temp_chunks.append((tmp_name, s.start, s.end))

    # Decide on parallelization: only parallelize on CPU to avoid multiple GPU model copies.
    model_device = _choose_device(device)
    use_parallel = max_workers and max_workers > 1 and model_device != "cuda"

    if use_parallel:
        workers = min(max_workers, len(temp_chunks) or 1)
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers,
            initializer=_worker_init,
            initargs=(model_name, "cpu", language),
        ) as exe:
            futures = []
            for tmp_name, seg_start, seg_end in temp_chunks:
                futures.append((exe.submit(_worker_transcribe, tmp_name, language), tmp_name, seg_start))

            for fut, tmp_name, seg_start in futures:
                try:
                    chunk_result = fut.result()
                except Exception:
                    chunk_result = {"segments": []}

                for seg in chunk_result.get("segments", []):
                    segments_out.append(
                        {
                            "start": float(seg.get("start", 0.0)) + seg_start,
                            "end": float(seg.get("end", 0.0)) + seg_start,
                            "text": seg.get("text", ""),
                        }
                    )

        # cleanup temp files
        for tmp_name, _, _ in temp_chunks:
            try:
                os.remove(tmp_name)
            except OSError:
                pass
    else:
        # If we have an in-process pipeline on GPU, it's more efficient to
        # run the pipeline once on the list of audio files (dataset-style)
        # instead of invoking it per-chunk. This reduces kernel launch overhead
        # and improves throughput.
        if model_instance is not None and model_device == "cuda":
            # Call pipeline on list of file paths; suppress noisy output
            import contextlib, os as _os

            paths = [p for p, _, _ in temp_chunks]
            results = []
            with open(_os.devnull, "w") as _devnull:
                with contextlib.redirect_stdout(_devnull), contextlib.redirect_stderr(_devnull):
                    # pipeline can accept a list of inputs and return a list of outputs
                    # Request timestamps for batched inputs to handle longer
                    # chunks (>30s) which require timestamp prediction.
                    if language:
                        results = model_instance(
                            [str(x) for x in paths], language=language, return_timestamps=True
                        )
                    else:
                        results = model_instance([str(x) for x in paths], return_timestamps=True)

            # Normalize and append
            for (tmp_name, seg_start, seg_end), res in zip(temp_chunks, results):
                # res may be dict with 'text' or 'segments', or a string
                if isinstance(res, dict) and "segments" in res:
                    segs = res["segments"]
                elif isinstance(res, dict) and "text" in res:
                    segs = [{"start": 0.0, "end": 0.0, "text": res.get("text", "")}]
                else:
                    segs = [{"start": 0.0, "end": 0.0, "text": str(res)}]

                for seg in segs:
                    segments_out.append(
                        {
                            "start": float(seg.get("start", 0.0)) + seg_start,
                            "end": float(seg.get("end", 0.0)) + seg_start,
                            "text": seg.get("text", ""),
                        }
                    )

            for tmp_name, _, _ in temp_chunks:
                try:
                    os.remove(tmp_name)
                except OSError:
                    pass
        else:
            # Sequential transcription reusing a single in-process pipeline where available
            for tmp_name, seg_start, seg_end in temp_chunks:
                try:
                    chunk_result = transcribe_with_whisper(
                        Path(tmp_name),
                        model_name=model_name,
                        device=device,
                        compute_type=compute_type,
                        model_instance=model_instance,
                        language=language,
                    )
                except Exception:
                    chunk_result = {"segments": []}

                for seg in chunk_result.get("segments", []):
                    segments_out.append(
                        {
                            "start": float(seg.get("start", 0.0)) + seg_start,
                            "end": float(seg.get("end", 0.0)) + seg_start,
                            "text": seg.get("text", ""),
                        }
                    )

            for tmp_name, _, _ in temp_chunks:
                try:
                    os.remove(tmp_name)
                except OSError:
                    pass

    # sort segments by start
    segments_out.sort(key=lambda x: x["start"])
    return {"segments": segments_out}
