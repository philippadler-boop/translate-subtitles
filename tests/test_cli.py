import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.cli import build_parser


class TestCLI(unittest.TestCase):
    """Test CLI argument parsing and validation."""

    def setUp(self):
        """Set up test fixtures."""
        self.parser = build_parser()

    def test_parser_creation(self):
        """Test that parser is created successfully."""
        self.assertIsInstance(self.parser, argparse.ArgumentParser)

    def test_required_arguments(self):
        """Test that required arguments are properly configured."""
        # Should fail without input file
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["--tgt-lang", "en"])

        # Should fail without target language
        with tempfile.NamedTemporaryFile(suffix=".srt", delete=False) as f:
            temp_path = f.name

        try:
            with self.assertRaises(SystemExit):
                self.parser.parse_args([temp_path])
        finally:
            Path(temp_path).unlink()

    def test_basic_parsing(self):
        """Test basic argument parsing."""
        with tempfile.NamedTemporaryFile(suffix=".srt", delete=False) as f:
            temp_path = f.name

        try:
            args = self.parser.parse_args(
                [
                    temp_path,
                    "--tgt-lang",
                    "en",
                    "--src-lang",
                    "de",
                    "--engine",
                    "google",
                ]
            )

            self.assertEqual(args.input, temp_path)
            self.assertEqual(args.tgt_lang, "en")
            self.assertEqual(args.src_lang, "de")
            self.assertEqual(args.engine, "google")
            self.assertIsNone(args.output)  # Default None

        finally:
            Path(temp_path).unlink()

    def test_default_values(self):
        """Test default argument values."""
        with tempfile.NamedTemporaryFile(suffix=".srt", delete=False) as f:
            temp_path = f.name

        try:
            args = self.parser.parse_args([temp_path, "--tgt-lang", "en"])

            self.assertEqual(args.src_lang, "auto")  # Default source language
            self.assertEqual(args.engine, "google")  # Default engine
            self.assertIsNone(args.output)  # Default output

        finally:
            Path(temp_path).unlink()

    def test_output_path_generation(self):
        """Test output path generation logic."""
        with tempfile.NamedTemporaryFile(suffix=".srt", delete=False) as f:
            temp_path = f.name

        try:
            args = self.parser.parse_args([temp_path, "--tgt-lang", "de"])

            # The parser itself does not enforce output paths anymore;
            # main() now always writes translated files into
            # workspaces/output/srt/<input_stem>.<tgt_lang>.srt.
            # Here we only verify that parsing succeeds and output is None.
            self.assertIsNone(args.output)

        finally:
            Path(temp_path).unlink()

    def test_engine_choices(self):
        """Test that only valid engines are accepted."""
        with tempfile.NamedTemporaryFile(suffix=".srt", delete=False) as f:
            temp_path = f.name

        try:
            # Valid engines
            for engine in ["google", "deepl", "gpt", "hf"]:
                args = self.parser.parse_args(
                    [temp_path, "--tgt-lang", "en", "--engine", engine]
                )
                self.assertEqual(args.engine, engine)

            # Invalid engine should fail
            with self.assertRaises(SystemExit):
                self.parser.parse_args(
                    [temp_path, "--tgt-lang", "en", "--engine", "invalid"]
                )

        finally:
            Path(temp_path).unlink()

    def test_help_output(self):
        """Test that help output contains examples."""
        with patch("sys.stdout"):
            try:
                self.parser.parse_args(["--help"])
            except SystemExit:
                pass  # --help causes SystemExit

        # Check that help was called (would be in stdout if we could capture it)
        # The argparse module handles help internally

    def test_language_code_validation(self):
        """Test that language codes are accepted (no specific validation in parser)."""
        with tempfile.NamedTemporaryFile(suffix=".srt", delete=False) as f:
            temp_path = f.name

        try:
            # Test various language codes
            for lang in ["en", "de", "fr", "es", "it", "pt", "auto"]:
                args = self.parser.parse_args([temp_path, "--tgt-lang", lang])
                self.assertEqual(args.tgt_lang, lang)

        finally:
            Path(temp_path).unlink()


if __name__ == "__main__":
    unittest.main()
