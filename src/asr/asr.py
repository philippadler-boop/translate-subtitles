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

import warnings

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

        # Try to load a processor (tokenizer + feature_extractor) and pass
        # its components explicitly to the pipeline so attention masks are
        # produced at preprocessing time instead of being forwarded as
        # generate kwargs which some models reject.
        processor = None
        try:
            from transformers import AutoProcessor

            processor = AutoProcessor.from_pretrained(model_name)
        except Exception:
            processor = None

        if processor is not None:
            try:
                _WORKER_PIPE = _local_pipeline(
                    "automatic-speech-recognition",
                    model=model_name,
                    device=hf_device,
                    tokenizer=processor.tokenizer,
                    feature_extractor=processor.feature_extractor,
                )
            except TypeError:
                # Some test doubles or older pipeline factories may not accept
                # tokenizer/feature_extractor kwargs; fall back to the simple
                # call signature in that case.
                _WORKER_PIPE = _local_pipeline("automatic-speech-recognition", model=model_name, device=hf_device)
        else:
            _WORKER_PIPE = _local_pipeline("automatic-speech-recognition", model=model_name, device=hf_device)

        # Try to configure the pipeline to use attention masks and avoid
        # the deprecated token timestamp behavior so workers don't emit
        # the deprecation warning repeatedly.
        _configure_pipeline_attention(_WORKER_PIPE)
    except Exception:
        _WORKER_PIPE = None


def _safe_pipeline_call(pipe, *args, **kwargs):
    """Call a transformers pipeline while preferring `return_attention_mask`.

    Some pipeline implementations forward unknown kwargs into `model.generate`
    which can raise a ValueError like "model_kwargs not used" when passing
    `return_attention_mask`. Try with `return_attention_mask=True` first and
    fall back to calling without it if the model rejects the kwarg.
    """
    if pipe is None:
        raise RuntimeError("pipeline instance is None")

    try:
        return pipe(*args, **kwargs)
    except ValueError as e:
        # If the pipeline forwarded unknown kwargs into model.generate and
        # the model rejected them, retry by removing commonly problematic
        # keys. Don't add `return_attention_mask` here — instead we prefer
        # to configure the pipeline's feature extractor/tokenizer.
        msg = str(e).lower()
        if "model_kwargs" in msg or "not used by the model" in msg:
            fallback_kw = dict(kwargs)
            # remove attention-mask related kw if present
            for k in ("return_attention_mask", "attention_mask"):
                fallback_kw.pop(k, None)
            return pipe(*args, **fallback_kw)
        raise


def _configure_pipeline_attention(pipe):
    """Configure a transformers pipeline or its components to prefer
    `return_attention_mask` and disable `return_token_timestamps` to
    avoid the deprecation warning.

    This attempts to set attributes on `feature_extractor`, `processor`,
    and `tokenizer` if present. It's best-effort and ignores failures.
    """
    if pipe is None:
        return

    try:
        # Try common attribute names used in speech pipelines
        for comp_name in ("feature_extractor", "processor", "tokenizer"):
            comp = getattr(pipe, comp_name, None)
            if comp is None:
                continue
            # set return_attention_mask if supported
            if hasattr(comp, "return_attention_mask"):
                try:
                    setattr(comp, "return_attention_mask", True)
                except Exception:
                    pass
            # Avoid touching deprecated `return_token_timestamps` attributes;
            # prefer setting `return_attention_mask` on the feature extractor
            # or processor so the pipeline emits attention masks without
            # forwarding unsupported kwargs to model.generate.
    except Exception:
        # best-effort only
        return



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
    # Use a safe caller that prefers `return_attention_mask=True` but falls
    # back when the underlying pipeline forwards unsupported kwargs to
    # `model.generate` (which raises a ValueError on some Transformers
    # versions/implementations).
    if language:
        res = _safe_pipeline_call(_WORKER_PIPE, str(tmp_wav_path), language=language, return_timestamps=True)
    else:
        res = _safe_pipeline_call(_WORKER_PIPE, str(tmp_wav_path), return_timestamps=True)

    # Normalize to dict-like with 'text' or 'segments'
    if isinstance(res, dict) and "segments" in res:
        return {"segments": res["segments"]}
    if isinstance(res, dict) and "text" in res:
        # If the pipeline returned only text (no per-segment timestamps),
        # estimate the duration of this chunk by inspecting the temporary
        # WAV file so we can return a non-zero end timestamp.
        try:
            import wave as _wave

            with _wave.open(str(tmp_wav_path), "rb") as _wf:
                frames = _wf.getnframes()
                rate = _wf.getframerate()
                duration = frames / float(rate) if rate else 0.0
        except Exception:
            duration = 0.0
        return {"segments": [{"start": 0.0, "end": float(duration), "text": res.get("text", "")}]} 
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
                    # Prefer creating a processor and passing its tokenizer +
                    # feature_extractor into the pipeline so attention masks are
                    # emitted by preprocessing rather than forwarded as model
                    # kwargs.
                    processor = None
                    try:
                        from transformers import AutoProcessor

                        processor = AutoProcessor.from_pretrained(model_name)
                    except Exception:
                        processor = None

                    if processor is not None:
                        try:
                            pipe = _hf_pipeline(
                                "automatic-speech-recognition",
                                model=model_name,
                                device=hf_device,
                                tokenizer=processor.tokenizer,
                                feature_extractor=processor.feature_extractor,
                            )
                        except TypeError:
                            pipe = _hf_pipeline("automatic-speech-recognition", model=model_name, device=hf_device)
                    else:
                        pipe = _hf_pipeline("automatic-speech-recognition", model=model_name, device=hf_device)

                # Configure feature-extractor/tokenizer to prefer attention masks and
                # disable deprecated token timestamps when possible to avoid
                # deprecation warnings.
                _configure_pipeline_attention(pipe)
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
            # Prefer `return_attention_mask=True` (via _safe_pipeline_call)
            if language:
                result = _safe_pipeline_call(pipe, str(wav_path), language=language, return_timestamps=need_timestamps)
            else:
                result = _safe_pipeline_call(pipe, str(wav_path), return_timestamps=need_timestamps)

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
    chunk_padding: float = 0.12,
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

            # Try to construct a processor and pass its components explicitly
            # into the pipeline to ensure attention masks are created.
            try:
                from transformers import AutoProcessor

                _processor = AutoProcessor.from_pretrained(model_name)
            except Exception:
                _processor = None

            if _processor is not None:
                try:
                    model_instance = _hf_pipeline(
                        "automatic-speech-recognition",
                        model=model_name,
                        device=hf_device,
                        tokenizer=_processor.tokenizer,
                        feature_extractor=_processor.feature_extractor,
                    )
                except TypeError:
                    model_instance = _hf_pipeline("automatic-speech-recognition", model=model_name, device=hf_device)
            else:
                model_instance = _hf_pipeline("automatic-speech-recognition", model=model_name, device=hf_device)
        except Exception:
            # If pipeline creation fails here, transcribe_with_whisper will raise per-chunk.
            model_instance = None
        else:
            # Configure feature extractor / tokenizer on the instance to
            # prefer attention masks and disable deprecated timestamp API.
            _configure_pipeline_attention(model_instance)

    # open source wave for slicing
    temp_chunks = []  # (tmp_name, seg.start, seg.end)
    with wave.open(str(wav_path), "rb") as src_wf:
        n_channels = src_wf.getnchannels()
        sampwidth = src_wf.getsampwidth()
        framerate = src_wf.getframerate()

        for s in segs:
            # Add a small padding around VAD segments to avoid cutting audio
            # mid-word which can trigger missing timestamp warnings from
            # Whisper's timestamp decoder. Padding is clamped to audio bounds.
            pad = chunk_padding
            start_time = max(0.0, s.start - pad)
            end_time = min((src_wf.getnframes() / framerate), s.end + pad)
            start_frame = int(round(start_time * framerate))
            end_frame = int(round(end_time * framerate))
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

            # Store the original segment start/end and the padding used so
            # we can correct timestamps returned by the model.
            temp_chunks.append((tmp_name, s.start, s.end, pad))

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
            for tmp_name, seg_start, seg_end, seg_pad in temp_chunks:
                futures.append((exe.submit(_worker_transcribe, tmp_name, language), tmp_name, seg_start, seg_end, seg_pad))

            for fut, tmp_name, seg_start, seg_end, seg_pad in futures:
                try:
                    chunk_result = fut.result()
                except Exception:
                    chunk_result = {"segments": []}

                for seg in chunk_result.get("segments", []):
                    # Adjust timestamps: model output timestamps are relative to
                    # the padded chunk. Subtract the padding to align with the
                    # original audio timeline.
                    adj_start = float(seg.get("start", 0.0)) + seg_start - seg_pad
                    adj_end = float(seg.get("end", 0.0)) + seg_start - seg_pad
                    if adj_start < 0:
                        adj_start = 0.0
                    if adj_end < 0:
                        adj_end = 0.0
                    segments_out.append(
                        {
                            "start": adj_start,
                            "end": adj_end,
                            "text": seg.get("text", ""),
                        }
                    )

        # cleanup temp files
        for tmp_name, _, _, _ in temp_chunks:
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

            paths = [p for p, _, _, _ in temp_chunks]
            results = []
            with open(_os.devnull, "w") as _devnull:
                with contextlib.redirect_stdout(_devnull), contextlib.redirect_stderr(_devnull):
                    # pipeline can accept a list of inputs and return a list of outputs
                    # Request timestamps for batched inputs to handle longer
                    # chunks (>30s) which require timestamp prediction.
                    if language:
                        results = _safe_pipeline_call(
                            model_instance, [str(x) for x in paths], language=language, return_timestamps=True
                        )
                    else:
                        results = _safe_pipeline_call(model_instance, [str(x) for x in paths], return_timestamps=True)

            # Normalize and append
            for (tmp_name, seg_start, seg_end, seg_pad), res in zip(temp_chunks, results):
                # res may be dict with 'text' or 'segments', or a string
                if isinstance(res, dict) and "segments" in res:
                    segs = res["segments"]
                elif isinstance(res, dict) and "text" in res:
                    # No timestamps provided by the pipeline for this chunk;
                    # derive chunk duration from the temporary file so the
                    # resulting word timestamps are not zero-length.
                    try:
                        import wave as _wave

                        with _wave.open(str(tmp_name), "rb") as _wf:
                            _frames = _wf.getnframes()
                            _rate = _wf.getframerate()
                            _duration = _frames / float(_rate) if _rate else 0.0
                    except Exception:
                        _duration = 0.0
                    segs = [{"start": 0.0, "end": float(_duration), "text": res.get("text", "")}]
                else:
                    segs = [{"start": 0.0, "end": 0.0, "text": str(res)}]

                for seg in segs:
                    # Correct for padding used when creating the chunk
                    adj_start = float(seg.get("start", 0.0)) + seg_start - seg_pad
                    adj_end = float(seg.get("end", 0.0)) + seg_start - seg_pad
                    if adj_start < 0:
                        adj_start = 0.0
                    if adj_end < 0:
                        adj_end = 0.0
                    segments_out.append(
                        {
                            "start": adj_start,
                            "end": adj_end,
                            "text": seg.get("text", ""),
                        }
                    )

            for tmp_name, _, _, _ in temp_chunks:
                try:
                    os.remove(tmp_name)
                except OSError:
                    pass
        else:
            # Sequential transcription reusing a single in-process pipeline where available
            for tmp_name, seg_start, seg_end, seg_pad in temp_chunks:
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
                    adj_start = float(seg.get("start", 0.0)) + seg_start - seg_pad
                    adj_end = float(seg.get("end", 0.0)) + seg_start - seg_pad
                    if adj_start < 0:
                        adj_start = 0.0
                    if adj_end < 0:
                        adj_end = 0.0
                    segments_out.append(
                        {
                            "start": adj_start,
                            "end": adj_end,
                            "text": seg.get("text", ""),
                        }
                    )

            for tmp_name, _, _, _ in temp_chunks:
                try:
                    os.remove(tmp_name)
                except OSError:
                    pass

    # sort segments by start
    segments_out.sort(key=lambda x: x["start"])
    return {"segments": segments_out}
