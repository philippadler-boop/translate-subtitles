import collections
import contextlib
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import webrtcvad


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
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)  # 2 bytes per sample (16-bit)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / (sample_rate * 2))
    while offset + n <= len(audio):
        yield audio[offset:offset + n], timestamp, duration
        timestamp += duration
        offset += n


def vad_collector(
    sample_rate: int,
    frame_duration_ms: int,
    padding_duration_ms: int,
    vad: webrtcvad.Vad,
    frames,
):
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    triggered = False

    voiced_start = 0.0
    for frame, timestamp, duration in frames:
        is_speech = vad.is_speech(frame, sample_rate)

        if not triggered:
            ring_buffer.append((frame, timestamp, duration, is_speech))
            num_voiced = len([f for f in ring_buffer if f[3]])
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                # use first voiced frame timestamp
                voiced_start = ring_buffer[0][1]
                ring_buffer.clear()
        else:
            if is_speech:
                # continue
                pass
            ring_buffer.append((frame, timestamp, duration, is_speech))
            num_unvoiced = len([f for f in ring_buffer if not f[3]])
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                # end of voiced segment
                voiced_end = ring_buffer[0][1] + ring_buffer[0][2]
                # yield start/end
                yield Segment(start=voiced_start, end=timestamp + duration)
                triggered = False
                ring_buffer.clear()

    if triggered:
        # end the final segment
        yield Segment(start=voiced_start, end=timestamp + duration)


def get_speech_segments(wav_path: Path, aggressiveness: int = 2) -> List[Segment]:
    """Return speech segments (start/end in seconds) for a mono 16-bit WAV file.

    Uses webrtcvad with the provided aggressiveness (0-3).
    """
    audio, sample_rate = _read_wave(wav_path)
    vad = webrtcvad.Vad(aggressiveness)
    frames = frame_generator(30, audio, sample_rate)
    segments = list(vad_collector(sample_rate, 30, 300, vad, frames))
    return segments
