import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import srt

from src.cli import main


class TestMainIntegration(unittest.TestCase):
    """Test main CLI integration functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.sample_srt_content = """1
00:00:00,000 --> 00:00:05,000
Hello world!

2
00:00:05,000 --> 00:00:10,000
This is a test.
"""

        self.sample_subtitles = [
            srt.Subtitle(
                index=1,
                start=srt.srt_timestamp_to_timedelta("00:00:00,000"),
                end=srt.srt_timestamp_to_timedelta("00:00:05,000"),
                content="Hello world!",
            ),
            srt.Subtitle(
                index=2,
                start=srt.srt_timestamp_to_timedelta("00:00:05,000"),
                end=srt.srt_timestamp_to_timedelta("00:00:10,000"),
                content="This is a test.",
            ),
        ]

    def create_temp_srt_file(self, content=None):
        """Helper to create a temporary SRT file."""
        if content is None:
            content = self.sample_srt_content

        with tempfile.NamedTemporaryFile(mode="w", suffix=".srt", delete=False) as f:
            f.write(content)
            return Path(f.name)

    def test_main_google_translation(self):
        """Test main function with Google translator."""
        input_path = self.create_temp_srt_file()
        # Output is now written into workspaces/output/srt/<stem>.de.srt
        root = Path(__file__).resolve().parents[1]
        workspaces_dir = root / "workspaces"
        outputs_root = workspaces_dir / "output"
        srt_out_dir = outputs_root / "srt"
        output_path = srt_out_dir / f"{input_path.stem}.de.srt"

        try:
            with patch(
                "src.translators.translator_google.translate_subtitles_google"
            ) as mock_translate:
                mock_translate.return_value = [
                    srt.Subtitle(
                        index=1,
                        start=srt.srt_timestamp_to_timedelta("00:00:00,000"),
                        end=srt.srt_timestamp_to_timedelta("00:00:05,000"),
                        content="Hallo Welt!",
                    ),
                    srt.Subtitle(
                        index=2,
                        start=srt.srt_timestamp_to_timedelta("00:00:05,000"),
                        end=srt.srt_timestamp_to_timedelta("00:00:10,000"),
                        content="Das ist ein Test.",
                    ),
                ]

                with patch(
                    "sys.argv",
                    [
                        "main.py",
                        str(input_path),
                        "--tgt-lang",
                        "de",
                        "--engine",
                        "google",
                    ],
                ):
                    main()

                # Verify translation was called
                mock_translate.assert_called_once()
                args, kwargs = mock_translate.call_args
                self.assertEqual(kwargs["source_lang"], "auto")  # default
                self.assertEqual(kwargs["target_lang"], "de")

                # Verify output file was created
                self.assertTrue(output_path.exists())

        finally:
            input_path.unlink()
            if output_path.exists():
                output_path.unlink()

    def test_main_hf_translation(self):
        """Test main function with HuggingFace translator."""
        input_path = self.create_temp_srt_file()
        root = Path(__file__).resolve().parents[1]
        workspaces_dir = root / "workspaces"
        outputs_root = workspaces_dir / "output"
        srt_out_dir = outputs_root / "srt"
        output_path = srt_out_dir / f"{input_path.stem}.en.srt"

        try:
            with patch(
                "src.translators.translator_hf.translate_subtitles_hf"
            ) as mock_translate:
                mock_translate.return_value = (
                    self.sample_subtitles
                )  # Mock returns same content

                with patch(
                    "sys.argv",
                    [
                        "main.py",
                        str(input_path),
                        "--src-lang",
                        "de",
                        "--tgt-lang",
                        "en",
                        "--engine",
                        "hf",
                    ],
                ):
                    main()

                mock_translate.assert_called_once()
                args, kwargs = mock_translate.call_args
                self.assertEqual(kwargs["source_lang"], "de")
                self.assertEqual(kwargs["target_lang"], "en")

                self.assertTrue(output_path.exists())

        finally:
            input_path.unlink()
            if output_path.exists():
                output_path.unlink()

    def test_main_invalid_input_file(self):
        """Test main function with nonexistent input file."""
        with patch("sys.argv", ["main.py", "nonexistent.srt", "--tgt-lang", "en"]):
            with self.assertRaises(SystemExit) as cm:
                main()

            self.assertIn("Input file not found", str(cm.exception))

    def test_main_custom_output_path(self):
        """Test main function with custom output path."""
        input_path = self.create_temp_srt_file()
        custom_output = input_path.parent / "custom_output.srt"

        try:
            with patch(
                "src.translators.translator_google.translate_subtitles_google"
            ) as mock_translate:
                mock_translate.return_value = self.sample_subtitles

                with patch(
                    "sys.argv",
                    [
                        "main.py",
                        str(input_path),
                        "--tgt-lang",
                        "en",
                        "--output",
                        str(custom_output),
                    ],
                ):
                    main()

                # Even when a custom output path is provided, main() now
                # writes into workspaces/output/srt/<stem>.<tgt>.srt.
                root = Path(__file__).resolve().parents[1]
                workspaces_dir = root / "workspaces"
                outputs_root = workspaces_dir / "output"
                srt_out_dir = outputs_root / "srt"
                expected = srt_out_dir / f"{input_path.stem}.en.srt"
                self.assertTrue(expected.exists())

        finally:
            input_path.unlink()
            if custom_output.exists():
                custom_output.unlink()

    def test_main_default_output_path_generation(self):
        """Test default output path generation."""
        input_path = self.create_temp_srt_file()
        root = Path(__file__).resolve().parents[1]
        workspaces_dir = root / "workspaces"
        outputs_root = workspaces_dir / "output"
        srt_out_dir = outputs_root / "srt"
        expected_output = srt_out_dir / f"{input_path.stem}.de.srt"

        try:
            with patch(
                "src.translators.translator_google.translate_subtitles_google"
            ) as mock_translate:
                mock_translate.return_value = self.sample_subtitles

                with patch(
                    "sys.argv", ["main.py", str(input_path), "--tgt-lang", "de"]
                ):
                    main()

                # Verify output was created with expected name
                self.assertTrue(expected_output.exists())

        finally:
            input_path.unlink()
            if expected_output.exists():
                expected_output.unlink()

    def test_main_unknown_engine(self):
        """Test main function with unknown engine."""
        input_path = self.create_temp_srt_file()

        try:
            with patch(
                "sys.argv",
                ["main.py", str(input_path), "--tgt-lang", "en", "--engine", "unknown"],
            ):
                # The argument parser will exit with code 2 for invalid choice
                with self.assertRaises(SystemExit):
                    main()

        finally:
            input_path.unlink()


if __name__ == "__main__":
    unittest.main()
