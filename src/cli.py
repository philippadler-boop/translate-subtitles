import argparse
import os
from pathlib import Path

from .io import read_srt_file, write_srt_file
from .utils import Progress
import sys


ASCII_HEADER = r"""
 __          ___     _                     ______ _               
 \ \        / / |   (_)                   |  ____| |              
  \ \  /\  / /| |__  _ ___ _ __   ___ _ __| |__  | | _____      __
   \ \/  \/ / | '_ \| / __| '_ \ / _ \ '__|  __| | |/ _ \ \ /\ / /
    \  /\  /  | | | | \__ \ |_) |  __/ |  | |    | | (_) \ V  V / 
     \/  \/   |_| |_|_|___/ .__/ \___|_|  |_|    |_|\___/ \_/\_/  
                          | |                                     
                          |_| 
                                                             
WhisperFlow - subtitle translation pipeline
"""

CLI_EXAMPLES = """
examples:
    Translate English subtitles to German using Google Translate:
        python main.py movie.srt --tgt-lang de

    Translate with auto-detected source language and specify output:
        python main.py movie.srt -o movie.de.srt --tgt-lang de

    Translate French subtitles to Spanish using DeepL:
        python main.py movie.fr.srt --src-lang fr --tgt-lang es --engine deepl

    Translate using OpenAI GPT-4o with auto language detection:
        python main.py movie.srt --tgt-lang es --engine gpt

    Regenerate subtitles from a video with Whisper large-v3 on GPU and translate to English:
        python main.py movie.mkv --tgt-lang en --regenerate --device cuda --asr-model large

    Translate using HuggingFace offline model:
        python main.py movie.srt --src-lang en --tgt-lang de --engine hf
"""

# Centralized CLI configuration
ASR_MODEL_CHOICES = ("tiny", "small", "medium", "large")
ASR_DEFAULT_MODEL = "small"
DEVICE_CHOICES = ("auto", "cpu", "cuda")
ENGINE_CHOICES = ("google", "deepl", "gpt", "hf")

# Best-effort CUDA DLL discovery for Windows: ensure the CUDA bin folders
# are on PATH so `ctranslate2` can locate the required CUDA DLLs.
# ctranslate2 4.6.x requires CUDA 12, so try v12.x first, then v13.0
_CUDA_BIN_CANDIDATES = [
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin\x64",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin",
]

for _cuda_bin in _CUDA_BIN_CANDIDATES:
    try:
        p = Path(_cuda_bin)
        if p.is_dir():
            current_path = os.environ.get("PATH", "")
            # Only prepend if not already present to avoid PATH growth.
            if str(p) not in current_path.split(os.pathsep):
                os.environ["PATH"] = str(p) + os.pathsep + current_path
            break
    except Exception:
        # Do not fail CLI import if CUDA folder is missing or inaccessible.
        pass


def _normalize_asr_model(name: str) -> str:
    """Normalize short ASR model names to full HuggingFace model IDs.

    Accepts either a full model id (returned unchanged) or a short name like
    'small' and expands to 'openai/whisper-small'. Special-case 'large'
    to map to the recommended 'openai/whisper-large-v3'.
    """
    if not name:
        return name
    n = name.strip()
    # If user supplied a full HF id (contains a '/'), return as-is
    if "/" in n:
        return n

    mapping = {
        "tiny": "openai/whisper-tiny",
        "small": "openai/whisper-small",
        "medium": "openai/whisper-medium",
        # Recommend the explicit v3 large model by default
        "large": "openai/whisper-large-v3",
        "large-v3": "openai/whisper-large-v3",
    }

    return mapping.get(n.lower(), n)


def _get_workspace_paths(base_name: str):
    """Get standard workspace output paths.
    
    Returns:
        Tuple of (audio_out_dir, srt_out_dir, outputs_root)
    """
    root = Path(__file__).resolve().parents[1]
    workspaces_dir = root / "workspaces"
    outputs_root = workspaces_dir / "output"
    outputs_root.mkdir(parents=True, exist_ok=True)
    
    audio_out_dir = outputs_root / "audio"
    audio_out_dir.mkdir(parents=True, exist_ok=True)
    srt_out_dir = outputs_root / "srt"
    srt_out_dir.mkdir(parents=True, exist_ok=True)
    
    return audio_out_dir, srt_out_dir, outputs_root


def _run_regeneration_pipeline(args, input_path: Path):
    """Run ASR regeneration pipeline: extract audio, transcribe, align, export words.
    
    Returns:
        Generated subtitles from ASR output.
    """
    try:
        from .io import extract_audio
        from .asr.asr import transcribe_with_vad
        from .subtitles.subtitle_sync import generate_srt_from_asr
    except Exception as e:
        raise SystemExit(f"Regenerate requested but required modules missing: {e}")

    base_name = input_path.stem
    audio_out_dir, srt_out_dir, _ = _get_workspace_paths(base_name)
    
    # Audio extraction
    p_audio = Progress("Audio")
    p_audio.start(total=1)
    wav = audio_out_dir / f"{base_name}.wav"
    wav = extract_audio(input_path, wav)
    p_audio.update(1, "extracted")
    p_audio.finish()

    # ASR transcription
    p_asr = Progress("ASR")
    p_asr.start()
    asr_language = None if args.src_lang == "auto" else args.src_lang
    asr_out = transcribe_with_vad(
        wav,
        model_name=_normalize_asr_model(args.asr_model),
        device=args.device,
        language=asr_language,
    )
    p_asr.finish("asr complete")

    # Optional forced alignment
    if args.align:
        method = args.align_method or "whisperx"
        if method == "whisperx":
            try:
                from .align.aligner import align_with_whisperx

                p_align = Progress("Alignment")
                p_align.start()
                aligned = align_with_whisperx(
                    wav, asr_out.get("segments", []), device=args.device
                )
                asr_out = {"segments": aligned}
                p_align.finish("aligned")
            except Exception as e:
                raise SystemExit(f"Alignment failed: {e}")
        else:
            raise SystemExit(f"Unknown alignment method: {method}")

    # Optional word export & visualization
    if args.export_words:
        try:
            from .visualizer.visualizer import (
                extract_words_from_aligned_segments,
                write_words_json,
            )

            p_words = Progress("Export Words")
            p_words.start()
            words = extract_words_from_aligned_segments(asr_out.get("segments", []))
            out_json = Path(args.export_words)
            write_words_json(words, out_json)
            p_words.update(1, "json written")
            # Visualization (words.html) has been removed; JSON is the canonical export.
            p_words.finish("export complete")
        except Exception as e:
            raise SystemExit(f"Exporting words failed: {e}")

    # Generate and save original-language subtitles
    subtitles = generate_srt_from_asr(asr_out)
    src_lang_tag = (args.src_lang or "auto").replace(" ", "_")
    original_srt_path = srt_out_dir / f"{base_name}.{src_lang_tag}.srt"
    write_srt_file(subtitles, original_srt_path)
    
    return subtitles


def _translate_subtitles(engine: str, subtitles, src_lang: str, tgt_lang: str, progress: Progress):
    """Dispatch to the appropriate translation engine.
    
    Returns:
        Translated subtitles.
    """
    translators = {
        "google": ("translator_google", "translate_subtitles_google"),
        "deepl": ("translator_deepl", "translate_subtitles_deepl"),
        "gpt": ("translator_gpt", "translate_subtitles_gpt"),
        "hf": ("translator_hf", "translate_subtitles_hf"),
    }
    
    if engine not in translators:
        raise SystemExit(f"Unknown engine: {engine}")
    
    module_name, func_name = translators[engine]
    translator_module = __import__(f"src.translators.{module_name}", fromlist=[func_name])
    translate_func = getattr(translator_module, func_name)
    
    return translate_func(
        subtitles=subtitles,
        source_lang=src_lang,
        target_lang=tgt_lang,
        progress=progress,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Translate .srt subtitle files between languages.",
        epilog=CLI_EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "input",
        help="Path to the input file (video/audio/.srt). If a video or audio file is provided, use `--regenerate` to extract audio and generate subtitles.",
    )

    parser.add_argument(
        "-o",
        "--output",
        help=(
            "Path to the output .srt file. "
            "Defaults to <input>.<target_lang>.srt in the same folder."
        ),
    )

    parser.add_argument(
        "--src-lang",
        default="auto",
        help=(
            "Source language code (e.g. en, de, fr). "
            "Use 'auto' to let the engine detect it if supported. "
            "Default: auto."
        ),
    )

    parser.add_argument(
        "--tgt-lang",
        required=True,
        help="Target language code (e.g. en, de, fr).",
    )

    parser.add_argument(
        "--engine",
        choices=list(ENGINE_CHOICES),
        default="google",
        help=(
            "Translation engine to use. "
            "google = deep-translator GoogleTranslator (no key needed), "
            "deepl = DeepL API (fallback to google if no key), "
            "gpt = OpenAI GPT-4o (requires OPENAI_API_KEY), "
            "hf = HuggingFace Transformers (offline, requires model). "
            "Default: google."
        ),
    )

    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Extract audio and generate a new .srt from scratch using ASR.",
    )

    parser.add_argument(
        "--align",
        action="store_true",
        help="Run forced alignment on ASR output to get word-level timestamps (requires whisperx).",
    )

    parser.add_argument(
        "--align-method",
        choices=["whisperx"],
        default=None,
        help="Alignment backend to use when --align is specified. Default: whisperx if available.",
    )

    parser.add_argument(
        "--export-words",
        help="Path to write word-level JSON output (defaults to <output>.words.json).",
        default=None,
    )

    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Export a simple HTML timeline visualization of word timings alongside JSON.",
    )

    parser.add_argument(
        "--asr-model",
        default=ASR_DEFAULT_MODEL,
        help=(
            "ASR model to use for regeneration (e.g. small, medium, large). "
            "Short names (tiny/small/medium/large) will be expanded to full HF IDs "
            "(e.g. 'small' -> 'openai/whisper-small', 'large' -> 'openai/whisper-large-v3'). "
            "You may also pass a full HuggingFace model id like 'openai/whisper-large-v3'."
        ),
    )

    parser.add_argument(
        "--device",
        default="auto",
        choices=list(DEVICE_CHOICES),
        help="Device for ASR. Defaults to auto.",
    )

    return parser


def main() -> None:
    parser = build_parser()

    # When no CLI args are provided, show header + full help.
    if len(sys.argv) == 1:
        print(ASCII_HEADER)
        parser.print_help()
        return

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_file():
        raise SystemExit(f"Input file not found: {input_path}")

    print(f"Using engine: {args.engine}")
    print(f"Source language: {args.src_lang} | Target language: {args.tgt_lang}")

    # Get or generate subtitles
    if args.regenerate:
        subtitles = _run_regeneration_pipeline(args, input_path)
    else:
        print(f"Reading:  {input_path}")
        subtitles = read_srt_file(input_path)

    # Translate subtitles
    p_trans = Progress("Translate")
    translated = _translate_subtitles(
        args.engine, subtitles, args.src_lang, args.tgt_lang, p_trans
    )

    # Write translated output
    _, srt_out_dir, _ = _get_workspace_paths(input_path.stem)

    translated_out = srt_out_dir / f"{input_path.stem}.{args.tgt_lang}.srt"
    print(f"Writing:  {translated_out}")
    write_srt_file(translated, translated_out)
    print("Done.")



