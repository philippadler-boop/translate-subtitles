import unittest
from unittest.mock import patch, MagicMock

import srt

from src.subtitles.grouping import SubtitleGrouper
from src.translators.translator_gpt import translate_subtitles_gpt


class TestTranslatorGPT(unittest.TestCase):
    def setUp(self):
        self.subs = [
            srt.Subtitle(index=1, start=srt.timedelta(seconds=0), end=srt.timedelta(seconds=2), content="Hello world!"),
            srt.Subtitle(index=2, start=srt.timedelta(seconds=2), end=srt.timedelta(seconds=4), content="This is a test."),
        ]

    @patch("src.translators.translator_gpt.OpenAI")
    def test_translate_per_line(self, mock_openai_class):
        mock_client = MagicMock()
        # emulate response structure
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=MagicMock(content=MagicMock()))]
        mock_resp.choices[0].message.content = MagicMock()
        mock_resp.choices[0].message.content.strip.return_value = "Hallo Welt!"
        mock_client().chat.completions.create.return_value = mock_resp
        mock_openai_class.return_value = mock_client()

        result = translate_subtitles_gpt(self.subs, "en", "de", progress=None)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].content, "Hallo Welt!")
        self.assertEqual(result[1].content, "Hallo Welt!")

    @patch("src.translators.translator_gpt.OpenAI")
    def test_translate_grouped(self, mock_openai_class):
        mock_client = MagicMock()
        # grouped response for combined text
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=MagicMock(content=MagicMock()))]
        mock_resp.choices[0].message.content = MagicMock()
        mock_resp.choices[0].message.content.strip.return_value = "Hallo Welt! Das ist ein Test."
        mock_client().chat.completions.create.return_value = mock_resp
        mock_openai_class.return_value = mock_client()

        grouper = SubtitleGrouper(max_chars=200, max_duration=6.0, group_by_punctuation=True)
        result = translate_subtitles_gpt(self.subs, "en", "de", progress=None, grouper=grouper)
        self.assertEqual(len(result), 2)
        # Ensure content assigned (best-effort distribution)
        self.assertTrue(all(isinstance(r.content, str) for r in result))


if __name__ == "__main__":
    unittest.main()
