import argparse
from pathlib import Path

from .srt_io import read_srt_file, write_srt_file


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
        help="Path to the input .srt file.",
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

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_file():
        raise SystemExit(f"Input file not found: {input_path}")

    if args.output:
        output_path = Path(args.output)
    else:
        # example: movie.srt -> movie.de.srt
        output_path = input_path.with_suffix(f".{args.tgt_lang}.srt")

    print(f"Reading:  {input_path}")
    subtitles = read_srt_file(input_path)

    engine = args.engine
    print(f"Using engine: {engine}")
    print(f"Source language: {args.src_lang} | Target language: {args.tgt_lang}")

    if engine == "google":
        from .translators.translator_google import translate_subtitles_google

        translated = translate_subtitles_google(
            subtitles=subtitles,
            source_lang=args.src_lang,
            target_lang=args.tgt_lang,
        )

    elif engine == "deepl":
        from .translators.translator_deepl import translate_subtitles_deepl

        translated = translate_subtitles_deepl(
            subtitles=subtitles,
            source_lang=args.src_lang,
            target_lang=args.tgt_lang,
        )

    elif engine == "gpt":
        from .translators.translator_gpt import translate_subtitles_gpt

        translated = translate_subtitles_gpt(
            subtitles=subtitles,
            source_lang=args.src_lang,
            target_lang=args.tgt_lang,
        )

    elif engine == "hf":
        from .translators.translator_hf import translate_subtitles_hf

        translated = translate_subtitles_hf(
            subtitles=subtitles,
            source_lang=args.src_lang,
            target_lang=args.tgt_lang,
        )

    else:
        raise SystemExit(f"Unknown engine: {engine}")

    print(f"Writing:  {output_path}")
    write_srt_file(translated, output_path)
    print("Done.")
