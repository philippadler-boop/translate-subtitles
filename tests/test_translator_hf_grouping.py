import unittest
from unittest.mock import patch, MagicMock

import srt

from src.subtitles.grouping import SubtitleGrouper
from src.translators.translator_hf import translate_subtitles_hf


class TestTranslatorHFGrouping(unittest.TestCase):
    def setUp(self):
        self.subs = [
            srt.Subtitle(index=1, start=srt.timedelta(seconds=0), end=srt.timedelta(seconds=2), content="Hello world!"),
            srt.Subtitle(index=2, start=srt.timedelta(seconds=2), end=srt.timedelta(seconds=4), content="This is a test."),
        ]

    @patch("src.translators.translator_hf.pipeline")
    @patch("src.translators.translator_hf.get_from_env_or_json")
    def test_grouped_flow(self, mock_get_cfg, mock_pipeline):
        mock_get_cfg.return_value = None
        mock_translator = MagicMock()
        # translator called with group text -> returns translation
        mock_translator.return_value = [{"translation_text": "Hallo Welt! Das ist ein Test."}]
        mock_pipeline.return_value = mock_translator

        grouper = SubtitleGrouper()
        result = translate_subtitles_hf(self.subs, "en", "de", progress=None, grouper=grouper)
        self.assertEqual(len(result), 2)
        self.assertTrue(all(isinstance(r.content, str) for r in result))


if __name__ == "__main__":
    unittest.main()
