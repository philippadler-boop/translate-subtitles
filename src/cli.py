import argparse
import os
from pathlib import Path

from .io import read_srt_file, write_srt_file
from .utils import Progress
import sys
from pathlib import Path


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


def build_parser() -> argparse.ArgumentParser:
    examples = """
examples:
  Translate English subtitles to German using Google Translate:
    python main.py movie.srt --tgt-lang de

  Translate with auto-detected source language and specify output:
    python main.py movie.srt -o movie.de.srt --tgt-lang de

  Translate French subtitles to Spanish using DeepL:
    python main.py movie.fr.srt --src-lang fr --tgt-lang es --engine deepl

  Translate using OpenAI GPT-4o with auto language detection:
    python main.py movie.srt --tgt-lang es --engine gpt

  Translate using HuggingFace offline model:
    python main.py movie.srt --src-lang en --tgt-lang de --engine hf
    """

    parser = argparse.ArgumentParser(
        description="Translate .srt subtitle files between languages.",
        epilog=examples,
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
        choices=["google", "deepl", "gpt", "hf"],
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
        default="small",
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
        help="Device for ASR: auto|cpu|cuda. Defaults to auto.",
    )

    # Hidden flag to suppress secondary interactive prompts in main()
    parser.add_argument(
        "--no-prompt",
        action="store_true",
        help=argparse.SUPPRESS,
    )

    return parser


def main() -> None:
    # Interactive mode when no CLI args provided
    if len(sys.argv) == 1:
        return _interactive_launcher()

    parser = build_parser()
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_file():
        raise SystemExit(f"Input file not found: {input_path}")

    engine = args.engine
    print(f"Using engine: {engine}")
    print(f"Source language: {args.src_lang} | Target language: {args.tgt_lang}")

    # If regenerate requested, run ASR pipeline first to create subtitles
    if args.regenerate:
        # If running interactively, ask the user which device and ASR model to use.
        # This allows choosing GPU/CPU and model size at runtime even when
        # the CLI was invoked with non-interactive defaults.
        try:
            # Only prompt if running in a TTY and no-prompt is not set
            interactive = sys.stdin.isatty() and not getattr(args, "no_prompt", False)
        except Exception:
            interactive = False

        if interactive:
            # Ask for device choice
            dev_prompt = f"ASR device (auto/cpu/cuda) [{args.device}]: "
            dev_in = input(dev_prompt).strip()
            if dev_in:
                if dev_in not in ("auto", "cpu", "cuda"):
                    print("Unknown device choice, using default.")
                else:
                    args.device = dev_in

            # Ask for ASR model
            model_prompt = f"ASR model (tiny/small/medium/whisper-large-v3) [{args.asr_model}]: "
            model_in = input(model_prompt).strip()
            if model_in:
                args.asr_model = model_in

        try:
            from .io import audio_cache_path, extract_audio
            from .asr.asr import transcribe_with_vad
            from .subtitles.subtitle_sync import generate_srt_from_asr
        except Exception as e:
            raise SystemExit(f"Regenerate requested but required modules missing: {e}")

        src_path = input_path

        # Compute per-input output base paths under workspaces/output
        root = Path(__file__).resolve().parents[1]
        workspaces_dir = root / "workspaces"
        outputs_root = workspaces_dir / "output"
        outputs_root.mkdir(parents=True, exist_ok=True)

        base_name = src_path.stem
        # Audio and SRT outputs: reuse base name and place into
        # workspaces/output/audio and workspaces/output/srt
        audio_out_dir = outputs_root / "audio"
        audio_out_dir.mkdir(parents=True, exist_ok=True)
        srt_out_dir = outputs_root / "srt"
        srt_out_dir.mkdir(parents=True, exist_ok=True)
        # Audio extraction progress
        p_audio = Progress("Audio")
        p_audio.start(total=1)
        # Extract audio into workspaces/output/audio/<basename>.wav
        wav = audio_out_dir / f"{base_name}.wav"
        wav = extract_audio(src_path, wav)
        p_audio.update(1, "extracted")
        p_audio.finish()

        # ASR progress (unknown total)
        p_asr = Progress("ASR")
        p_asr.start()
        asr_out = transcribe_with_vad(
            wav, model_name=_normalize_asr_model(args.asr_model), device=args.device
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
                    # aligned may be a list of segments with word timings; adapt to generator
                    asr_out = {"segments": aligned}
                    p_align.finish("aligned")
                except Exception as e:
                    raise SystemExit(f"Alignment failed: {e}")
            else:
                raise SystemExit(f"Unknown alignment method: {method}")

        # Optionally export words & visualization
        if args.export_words:
            try:
                from .visualizer.visualizer import (
                    extract_words_from_aligned_segments,
                    write_words_json,
                    write_simple_html_timeline,
                )

                p_words = Progress("Export Words")
                p_words.start()
                words = extract_words_from_aligned_segments(asr_out.get("segments", []))
                out_json = Path(args.export_words)
                write_words_json(words, out_json)
                p_words.update(1, "json written")
                if args.visualize:
                    html_out = out_json.with_suffix(".html")
                    write_simple_html_timeline(out_json, html_out)
                    p_words.update(1, "html written")
                p_words.finish("export complete")
            except Exception as e:
                raise SystemExit(f"Exporting words/visualization failed: {e}")

        # Generate initial subtitles from ASR output and persist original-language SRT
        subtitles = generate_srt_from_asr(asr_out)
        # Use the configured source language or "auto" in the filename
        src_lang_tag = (args.src_lang or "auto").replace(" ", "_")
        original_srt_path = srt_out_dir / f"{base_name}.{src_lang_tag}.srt"
        write_srt_file(subtitles, original_srt_path)

        # If a translation target is requested, fall through to translation step
    else:
        # No regeneration: read existing SRT file
        print(f"Reading:  {input_path}")
        subtitles = read_srt_file(input_path)

    # Now translate `subtitles` using selected engine; provide Progress to translators
    p_trans = Progress("Translate")
    if engine == "google":
        from .translators.translator_google import translate_subtitles_google

        translated = translate_subtitles_google(
            subtitles=subtitles,
            source_lang=args.src_lang,
            target_lang=args.tgt_lang,
            progress=p_trans,
        )

    elif engine == "deepl":
        from .translators.translator_deepl import translate_subtitles_deepl

        translated = translate_subtitles_deepl(
            subtitles=subtitles,
            source_lang=args.src_lang,
            target_lang=args.tgt_lang,
            progress=p_trans,
        )

    elif engine == "gpt":
        from .translators.translator_gpt import translate_subtitles_gpt

        translated = translate_subtitles_gpt(
            subtitles=subtitles,
            source_lang=args.src_lang,
            target_lang=args.tgt_lang,
            progress=p_trans,
        )

    elif engine == "hf":
        from .translators.translator_hf import translate_subtitles_hf

        translated = translate_subtitles_hf(
            subtitles=subtitles,
            source_lang=args.src_lang,
            target_lang=args.tgt_lang,
            progress=p_trans,
        )
    else:
        raise SystemExit(f"Unknown engine: {engine}")

    # Always write translated SRTs only into workspaces/output/srt
    # so that all outputs are grouped in a single location.
    root = Path(__file__).resolve().parents[1]
    workspaces_dir = root / "workspaces"
    outputs_root = workspaces_dir / "output"
    srt_out_dir = outputs_root / "srt"
    srt_out_dir.mkdir(parents=True, exist_ok=True)

    translated_out = srt_out_dir / f"{input_path.stem}.{args.tgt_lang}.srt"
    print(f"Writing:  {translated_out}")
    write_srt_file(translated, translated_out)
    print("Done.")


def _interactive_launcher() -> None:
    """Simple interactive CLI for choosing input and running the pipeline."""
    print(ASCII_HEADER)

    root = Path(__file__).resolve().parents[1]
    workspace_dir = root / "workspaces"
    if not workspace_dir.is_dir():
        print(f"No workspace directory found at {workspace_dir}")
        return

    input_types = {"video": [".mp4", ".mkv", ".mov"], "audio": [".wav", ".mp3", ".m4a", ".flac"], "srt": [".srt"]}

    # Ask which input type
    while True:
        choice = input("Select input type (video/audio/srt): ").strip().lower()
        if choice in input_types:
            break
        print("Please enter one of: video, audio, srt")

    exts = input_types[choice]

    # Prefer a subdirectory named after the choice (e.g. workspaces/video)
    candidate_dir = workspace_dir / choice
    if candidate_dir.is_dir():
        search_dir = candidate_dir
    else:
        search_dir = workspace_dir

    # Collect matching files (non-recursive if using a dedicated folder,
    # otherwise search recursively so users can organize files arbitrarily)
    if search_dir == candidate_dir:
        files = [p for p in search_dir.iterdir() if p.suffix.lower() in exts]
    else:
        files = [p for p in search_dir.rglob("*") if p.suffix.lower() in exts]

    if not files:
        print(f"No {choice} files found in {search_dir}")
        return

    print(f"Found {len(files)} {choice} files:")
    for i, p in enumerate(files, start=1):
        print(f"  {i}) {p.name}")

    while True:
        sel = input(f"Pick a file [1-{len(files)}]: ").strip()
        try:
            idx = int(sel) - 1
            if 0 <= idx < len(files):
                selected = files[idx]
                break
        except Exception:
            pass
        print("Invalid selection")

    print(f"Selected: {selected}")

    # Only a single action is currently supported; no quit option here.
    if choice == "srt":
        action_label = "Translate subtitles"
    else:
        action_label = "Regenerate + Translate"

    print("Available actions:")
    print(f"  1) {action_label}")

    while True:
        act = input("Choose action [1]: ").strip().lower() or "1"
        if act == "1":
            break
        print("Invalid action")

    # Ask for source and target language
    src = input("Source language code (e.g. de, en or 'auto') [auto]: ").strip() or "auto"
    tgt = input("Target language code (e.g. en, de) [en]: ").strip() or "en"

    # Build args and run parsed pipeline
    parser = build_parser()
    args_list = [str(selected), "--src-lang", src, "--tgt-lang", tgt]
    if choice != "srt":
        # Ask ASR device preference: prefer GPU when available, otherwise CPU
        try:
            import torch

            default_device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            default_device = "cpu"

        # Prompt user for device choice (show default)
        dev_in = input(f"ASR device (auto/cpu/cuda) [{default_device}]: ").strip()
        if not dev_in:
            dev_in = default_device
        if dev_in not in ("auto", "cpu", "cuda"):
            print("Unknown device choice, using default.")
            dev_in = default_device

        # Prompt user for ASR model (suggest 'small')
        model_default = "small"
        model_in = input(
            f"ASR model (tiny/small/medium/large) [{model_default}]: "
        ).strip()
        if not model_in:
            model_in = model_default

        args_list.append("--regenerate")
        args_list.extend(["--device", dev_in, "--asr-model", model_in])

    # Ensure main() does not prompt again for ASR choices
    args_list.append("--no-prompt")

    print(f"Running pipeline with args: {args_list}")
    args = parser.parse_args(args_list)
    # call main logic by reusing the current main() flow: set sys.argv and recurse
    # to avoid duplicating logic we call the non-interactive path by invoking
    # the run via modifying sys.argv temporarily
    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0]] + args_list
        main()
    finally:
        sys.argv = old_argv
