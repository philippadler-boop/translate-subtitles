import unittest
from src.subtitles.subtitle_sync import generate_srt_from_asr


class TestSubtitleSync(unittest.TestCase):
    def test_generate_srt_simple(self):
        asr = {"segments": [{"start": 0.0, "end": 2.0, "text": "Hello world"}]}
        subs = generate_srt_from_asr(asr, max_chars=42)
        self.assertEqual(len(subs), 1)
        self.assertEqual(subs[0].content, "Hello world")

    def test_generate_srt_split(self):
        text = "This is a very long sentence that should be split into multiple subtitle lines because it exceeds the max chars"
        asr = {"segments": [{"start": 0.0, "end": 10.0, "text": text}]}
        subs = generate_srt_from_asr(asr, max_chars=20)
        self.assertTrue(len(subs) > 1)
        # Ensure timings progress
        starts = [s.start.total_seconds() for s in subs]
        self.assertTrue(all(x < y for x, y in zip(starts, starts[1:])))


if __name__ == '__main__':
    unittest.main()
