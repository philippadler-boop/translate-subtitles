import collections
import contextlib
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

try:
    import webrtcvad  # type: ignore

    _HAVE_WEBRTC = True
except Exception:
    webrtcvad = None  # type: ignore
    _HAVE_WEBRTC = False


@dataclass
class Segment:
    start: float
    end: float


def _read_wave(path: Path) -> Tuple[bytes, int]:
    with contextlib.closing(wave.open(str(path), "rb")) as wf:
        num_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        pcm_data = wf.readframes(wf.getnframes())
    if num_channels != 1:
        raise ValueError("VAD expects mono WAV input")
    if sample_width != 2:
        raise ValueError("VAD expects 16-bit PCM WAV input")
    return pcm_data, sample_rate


def frame_generator(frame_duration_ms: int, audio: bytes, sample_rate: int):
    n = int(
        sample_rate * (frame_duration_ms / 1000.0) * 2
    )  # 2 bytes per sample (16-bit)
    offset = 0
    timestamp = 0.0
    duration = float(n) / (sample_rate * 2)
    while offset + n <= len(audio):
        yield audio[offset : offset + n], timestamp, duration
        timestamp += duration
        offset += n


def vad_collector(
    sample_rate: int,
    frame_duration_ms: int,
    padding_duration_ms: int,
    vad,
    frames,
):
    """Collect voiced regions from frames.

    This function expects `vad` to implement `is_speech(frame, sample_rate)`.
    When `webrtcvad` is not available, a simple energy-based `vad` object
    with `is_speech` will be used by the caller.
    """
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    triggered = False

    voiced_start = 0.0
    last_timestamp = 0.0
    for frame, timestamp, duration in frames:
        is_speech = vad.is_speech(frame, sample_rate)
        last_timestamp = timestamp

        if not triggered:
            ring_buffer.append((frame, timestamp, duration, is_speech))
            num_voiced = len([f for f in ring_buffer if f[3]])
            if len(ring_buffer) and num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                voiced_start = ring_buffer[0][1]
                ring_buffer.clear()
        else:
            ring_buffer.append((frame, timestamp, duration, is_speech))
            num_unvoiced = len([f for f in ring_buffer if not f[3]])
            if len(ring_buffer) and num_unvoiced > 0.9 * ring_buffer.maxlen:
                yield Segment(start=voiced_start, end=timestamp + duration)
                triggered = False
                ring_buffer.clear()

    if triggered:
        yield Segment(
            start=voiced_start,
            end=last_timestamp + (duration if "duration" in locals() else 0),
        )


def _simple_energy_vad(
    frame_bytes: bytes, sample_rate: int, threshold: float = 500.0
) -> bool:
    """Very small RMS-based detector for a single frame (16-bit PCM).

    This is a fallback if `webrtcvad` cannot be built on the platform.
    It is not as robust as `webrtcvad`, but provides a usable segmentation.
    """
    import struct

    if not frame_bytes:
        return False
    # Interpret as signed 16-bit little-endian samples
    count = len(frame_bytes) // 2
    fmt = f"<{count}h"
    try:
        samples = struct.unpack(fmt, frame_bytes)
    except Exception:
        return False
    # compute simple RMS
    s = 0
    for v in samples:
        s += v * v
    rms = (s / max(1, count)) ** 0.5
    return rms > threshold


def get_speech_segments(wav_path: Path, aggressiveness: int = 2) -> List[Segment]:
    """Return speech segments (start/end in seconds) for a mono 16-bit WAV file.

    Tries to use `webrtcvad` when available; otherwise falls back to a
    simple energy-based detector.
    """
    audio, sample_rate = _read_wave(wav_path)

    if _HAVE_WEBRTC:
        vad = webrtcvad.Vad(aggressiveness)
        frames = frame_generator(30, audio, sample_rate)
        segments = list(vad_collector(sample_rate, 30, 300, vad, frames))
        return segments

    # Fallback energy VAD
    class SimpleVad:
        def is_speech(self, frame_bytes, sr):
            return _simple_energy_vad(frame_bytes, sr)

    vad = SimpleVad()
    frames = frame_generator(30, audio, sample_rate)
    segments = list(vad_collector(sample_rate, 30, 300, vad, frames))
    return segments
