#!/usr/bin/env python3
"""Quick test of ASR on 5-minute audio snippet."""
from pathlib import Path
import traceback
import os

def main():
    try:
        # Ensure Windows can find CUDA runtime DLLs (e.g. cublas64_12.dll)
        cuda_bin = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin")
        if cuda_bin.exists():
            # Try to register the directory for DLL search (preferred)
            try:
                os.add_dll_directory(str(cuda_bin))
                print(f"Added '{cuda_bin}' to DLL search path")
            except Exception:
                print(f"os.add_dll_directory failed; will prepend to PATH")

            # Also prepend to PATH to cover modules that rely on PATH at import-time
            os.environ["PATH"] = str(cuda_bin) + os.pathsep + os.environ.get("PATH", "")
            print(f"Prepended '{cuda_bin}' to PATH")

        from src.asr.asr import transcribe_with_vad
        from src.subtitles.subtitle_sync import generate_srt_from_asr
        from src.io import write_srt_file
        
        wav_path = Path("workspaces/output/audio/test_5min.wav")
        
        if not wav_path.exists():
            print(f"Error: {wav_path} not found")
            return
        
        print(f"Testing ASR on: {wav_path}")
        print(f"File size: {wav_path.stat().st_size / (1024*1024):.2f} MB")
        print("\nRunning transcription with VAD on CUDA (this may take a minute)...\n")
        
        # Run ASR with a smaller HF Whisper model on CUDA
        # Use a large Whisper model (may be slow and requires substantial GPU RAM)
        asr_result = transcribe_with_vad(
            wav_path,
            model_name="openai/whisper-large-v2",
            device="cuda"
        )
        print("✓ ASR completed!")
    except Exception as e:
        print(f"\n✗ Error during ASR:")
        print(f"  {type(e).__name__}: {e}")
        traceback.print_exc()
        return
        print("✓ ASR completed!")
    except Exception as e:
        print(f"\n✗ Error during ASR:")
        print(f"  {type(e).__name__}: {e}")
        traceback.print_exc()
        return
    
    try:
        segments = asr_result.get("segments", [])
        print(f"  Segments detected: {len(segments)}")
        
        if segments:
            print(f"\nFirst 3 segments:")
            for i, seg in enumerate(segments[:3], 1):
                start = seg.get("start", 0)
                end = seg.get("end", 0)
                text = seg.get("text", "").strip()
                print(f"  {i}. [{start:.2f}s - {end:.2f}s] {text}")
        
        # Generate SRT format
        subtitles = generate_srt_from_asr(asr_result)
        output_srt = Path("workspaces/output/srt/test_5min.srt")
        output_srt.parent.mkdir(parents=True, exist_ok=True)
        write_srt_file(subtitles, output_srt)
        
        print(f"\n✓ SRT saved to: {output_srt}")
        print(f"  Total subtitle entries: {len(subtitles)}")
        
        # Show first subtitle
        if subtitles:
            first = subtitles[0]
            print(f"\nFirst subtitle entry:")
            print(f"  Index: {first.index}")
            print(f"  Time: {first.start} --> {first.end}")
            print(f"  Text: {first.content}")
    except Exception as e:
        print(f"\n✗ Error generating SRT:")
        print(f"  {type(e).__name__}: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
