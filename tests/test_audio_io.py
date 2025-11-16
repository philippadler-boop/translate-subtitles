import unittest
from pathlib import Path
from unittest.mock import patch

from src.audio_io import audio_cache_path, extract_audio
import tempfile


class TestAudioIO(unittest.TestCase):
    def test_audio_cache_path_default(self):
        p = Path("video.mp4")
        cache = audio_cache_path(p)
        self.assertTrue(str(cache).endswith(".cache\\audio\\video.wav") or str(cache).endswith(".cache/audio/video.wav"))

    @patch("src.audio_io.shutil.which")
    @patch("src.audio_io.subprocess.run")
    def test_extract_audio_calls_ffmpeg(self, mock_run, mock_which):
        mock_which.return_value = "/usr/bin/ffmpeg"
        src = Path("in.mp4")
        # write output into a temporary directory to avoid creating files in repo root
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "out.wav"

            # simulate subprocess returning successfully
            mock_run.return_value = None

            # Should not raise (we're mocking ffmpeg)
            r = extract_audio(src, out, force=True)
            self.assertEqual(r, out)


if __name__ == '__main__':
    unittest.main()
