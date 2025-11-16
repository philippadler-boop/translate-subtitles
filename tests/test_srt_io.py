import tempfile
import unittest
from pathlib import Path

import srt

from src.io.srt_io import read_srt_file, write_srt_file


class TestSrtIO(unittest.TestCase):
    """Test SRT file reading and writing functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.sample_subtitles = [
            srt.Subtitle(
                index=1,
                start=srt.srt_timestamp_to_timedelta("00:00:00,000"),
                end=srt.srt_timestamp_to_timedelta("00:00:05,000"),
                content="Hello world!"
            ),
            srt.Subtitle(
                index=2,
                start=srt.srt_timestamp_to_timedelta("00:00:05,000"),
                end=srt.srt_timestamp_to_timedelta("00:00:10,000"),
                content="This is a test subtitle.\nWith multiple lines."
            ),
        ]

    def test_read_srt_file_valid(self):
        """Test reading a valid SRT file."""
        srt_content = """1
00:00:00,000 --> 00:00:05,000
Hello world!

2
00:00:05,000 --> 00:00:10,000
This is a test subtitle.
With multiple lines.
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.srt', delete=False) as f:
            f.write(srt_content)
            temp_path = Path(f.name)

        try:
            subtitles = read_srt_file(temp_path)
            self.assertEqual(len(subtitles), 2)
            self.assertEqual(subtitles[0].content, "Hello world!")
            self.assertEqual(subtitles[1].content, "This is a test subtitle.\nWith multiple lines.")
        finally:
            temp_path.unlink()

    def test_read_srt_file_nonexistent(self):
        """Test reading a nonexistent SRT file raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            read_srt_file(Path("nonexistent.srt"))

    def test_write_srt_file(self):
        """Test writing subtitles to SRT file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.srt', delete=False) as f:
            temp_path = Path(f.name)

        try:
            write_srt_file(self.sample_subtitles, temp_path)

            # Read back and verify
            with open(temp_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Check that the file contains expected content
            self.assertIn("Hello world!", content)
            self.assertIn("This is a test subtitle.", content)
            self.assertIn("With multiple lines.", content)

        finally:
            temp_path.unlink()

    def test_roundtrip_read_write(self):
        """Test that reading and writing preserves subtitle data."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.srt', delete=False) as f:
            temp_path = Path(f.name)

        try:
            # Write original subtitles
            write_srt_file(self.sample_subtitles, temp_path)

            # Read them back
            read_subtitles = read_srt_file(temp_path)

            # Verify they match
            self.assertEqual(len(read_subtitles), len(self.sample_subtitles))
            for original, read_back in zip(self.sample_subtitles, read_subtitles):
                self.assertEqual(original.index, read_back.index)
                self.assertEqual(original.start, read_back.start)
                self.assertEqual(original.end, read_back.end)
                self.assertEqual(original.content, read_back.content)

        finally:
            temp_path.unlink()


if __name__ == '__main__':
    unittest.main()