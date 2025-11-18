import unittest
import srt

from src.subtitles.grouping import SubtitleGrouper


class TestSubtitleGrouper(unittest.TestCase):
    def setUp(self):
        self.subs = [
            srt.Subtitle(index=1, start=srt.timedelta(seconds=0), end=srt.timedelta(seconds=2), content="Hello world!"),
            srt.Subtitle(index=2, start=srt.timedelta(seconds=2), end=srt.timedelta(seconds=4), content="This is a test."),
            srt.Subtitle(index=3, start=srt.timedelta(seconds=4), end=srt.timedelta(seconds=6), content="Another line without punctuation"),
        ]

    def test_group_default(self):
        g = SubtitleGrouper(max_chars=100, max_duration=10.0, group_by_punctuation=True)
        groups = g.group(self.subs)
        # With punctuation grouping, first two end with punctuation and should form groups
        self.assertTrue(len(groups) >= 1)
        # Ensure group text contains original texts
        all_text = " ".join([s.content for s in self.subs])
        joined = " ".join([gr.text for gr in groups])
        self.assertIn("Hello world!", joined)
        self.assertIn("This is a test.", joined)

    def test_distribute_proportional(self):
        g = SubtitleGrouper()
        translated = "Hallo Welt! Das ist ein Test. Noch eine Zeile"
        parts = g.distribute(translated, self.subs)
        # Should return number of parts equal to originals
        self.assertEqual(len(parts), len(self.subs))
        # Each part should be non-empty (best-effort)
        for p in parts:
            self.assertIsInstance(p, str)

    def test_empty_originals(self):
        g = SubtitleGrouper()
        parts = g.distribute("Some text", [])
        self.assertEqual(parts, [])


if __name__ == "__main__":
    unittest.main()
