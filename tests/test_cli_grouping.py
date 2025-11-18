import argparse
import tempfile
import unittest
from pathlib import Path

from src.cli import build_parser


class TestCLIGroupingFlags(unittest.TestCase):
    def setUp(self):
        self.parser = build_parser()

    def test_grouping_flags_present(self):
        with tempfile.NamedTemporaryFile(suffix=".srt", delete=False) as f:
            temp_path = f.name
        try:
            args = self.parser.parse_args([
                temp_path,
                "--tgt-lang",
                "en",
                "--group-subtitles",
                "--group-max-chars",
                "150",
                "--group-max-duration",
                "4.5",
                "--group-by-punctuation",
            ])
            self.assertTrue(args.group_subtitles)
            self.assertEqual(args.group_max_chars, 150)
            self.assertAlmostEqual(args.group_max_duration, 4.5)
            self.assertTrue(args.group_by_punctuation)
        finally:
            Path(temp_path).unlink()

    def test_grouping_defaults(self):
        with tempfile.NamedTemporaryFile(suffix=".srt", delete=False) as f:
            temp_path = f.name
        try:
            args = self.parser.parse_args([temp_path, "--tgt-lang", "en"])
            self.assertFalse(args.group_subtitles)
            self.assertEqual(args.group_max_chars, 200)
            self.assertAlmostEqual(args.group_max_duration, 6.0)
            self.assertTrue(args.group_by_punctuation)
        finally:
            Path(temp_path).unlink()


if __name__ == "__main__":
    unittest.main()
