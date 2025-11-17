import wave
from pathlib import Path
import logging
import sys

# Ensure imports from repo work
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def make_silent_wav(path: Path, duration_s: float = 1.0, framerate: int = 16000):
    n_frames = int(duration_s * framerate)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(framerate)
        wf.writeframes(b"\x00\x00" * n_frames)


def main():
    # Enable warnings and INFO from transformers
    logging.basicConfig(level=logging.INFO)
    try:
        import transformers
        print(f"transformers version: {transformers.__version__}")
    except Exception:
        print("transformers not installed")

    out_dir = Path("workspaces") / "output" / "audio"
    out_dir.mkdir(parents=True, exist_ok=True)
    wav = out_dir / "smoke_test.wav"
    make_silent_wav(wav, duration_s=1.0)

    print(f"Created test WAV: {wav} ({wav.stat().st_size} bytes)")

    try:
        from src.asr.asr import transcribe_with_vad

        print("Starting transcribe_with_vad on GPU (device='cuda') with tiny model...")
        out = transcribe_with_vad(wav, model_name="openai/whisper-tiny", device="cuda", max_workers=1)
        print("Transcription result:")
        print(out)
    except Exception as e:
        print(f"Error during ASR: {e}")


if __name__ == '__main__':
    main()
