import unittest
from unittest.mock import patch, MagicMock

import srt

from src.translators.translator_google import translate_subtitles_google


class TestTranslatorGoogle(unittest.TestCase):
    """Test Google Translate translator functionality."""

    def setUp(self):
        """Set up test fixtures."""
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
                content="This is a test.\nWith multiple lines.",
            ),
        ]

    @patch("src.translators.translator_google.GoogleTranslator")
    def test_translate_subtitles_google_success(self, mock_google_translator_class):
        """Test successful translation with mocked GoogleTranslator."""
        # Mock the GoogleTranslator instance
        mock_translator_instance = MagicMock()
        mock_translator_instance.translate.return_value = "Hallo Welt!"
        mock_google_translator_class.return_value = mock_translator_instance

        result = translate_subtitles_google(self.sample_subtitles, "en", "de")

        # Verify results
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].content, "Hallo Welt!")
        self.assertEqual(result[1].content, "Hallo Welt!")

        # Verify GoogleTranslator was instantiated once
        mock_google_translator_class.assert_called_once_with(source="en", target="de")

        # Verify translate method was called for each subtitle
        self.assertEqual(mock_translator_instance.translate.call_count, 2)
        mock_translator_instance.translate.assert_any_call("Hello world!")
        mock_translator_instance.translate.assert_any_call(
            "This is a test.\nWith multiple lines."
        )

    @patch("src.translators.translator_google.GoogleTranslator")
    def test_translate_subtitles_google_with_auto_source(
        self, mock_google_translator_class
    ):
        """Test translation with auto-detected source language."""
        mock_translator_instance = MagicMock()
        mock_translator_instance.translate.return_value = "Translated text"
        mock_google_translator_class.return_value = mock_translator_instance

        _result = translate_subtitles_google(self.sample_subtitles, "auto", "de")

        # Verify GoogleTranslator was called with source='auto'
        call_args = mock_google_translator_class.call_args_list[0]
        self.assertEqual(call_args[1]["source"], "auto")
        self.assertEqual(call_args[1]["target"], "de")

    @patch("src.translators.translator_google.GoogleTranslator")
    def test_translate_subtitles_google_preserves_metadata(
        self, mock_google_translator_class
    ):
        """Test that subtitle metadata is preserved during translation."""
        mock_translator_instance = MagicMock()
        mock_translator_instance.translate.return_value = "Translated"
        mock_google_translator_class.return_value = mock_translator_instance

        result = translate_subtitles_google(self.sample_subtitles, "en", "de")

        # Check that all metadata is preserved
        for original, translated in zip(self.sample_subtitles, result):
            self.assertEqual(original.index, translated.index)
            self.assertEqual(original.start, translated.start)
            self.assertEqual(original.end, translated.end)
            self.assertEqual(original.proprietary, translated.proprietary)

    @patch("src.translators.translator_google.GoogleTranslator")
    def test_translate_subtitles_google_handles_multiline(
        self, mock_google_translator_class
    ):
        """Test that multiline subtitles are handled correctly."""
        mock_translator_instance = MagicMock()
        mock_translator_instance.translate.return_value = (
            "Übersetzte Zeile 1\nÜbersetzte Zeile 2"
        )
        mock_google_translator_class.return_value = mock_translator_instance

        multiline_subtitle = srt.Subtitle(
            index=1,
            start=srt.srt_timestamp_to_timedelta("00:00:00,000"),
            end=srt.srt_timestamp_to_timedelta("00:00:05,000"),
            content="Line 1\nLine 2",
        )

        result = translate_subtitles_google([multiline_subtitle], "en", "de")

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].content, "Übersetzte Zeile 1\nÜbersetzte Zeile 2")

        # Verify the entire multiline content was passed to translate
        mock_translator_instance.translate.assert_called_once_with("Line 1\nLine 2")

    @patch("src.translators.translator_google.GoogleTranslator")
    def test_translate_subtitles_google_error_handling(
        self, mock_google_translator_class
    ):
        """Test error handling when GoogleTranslator fails."""
        mock_translator_instance = MagicMock()
        mock_translator_instance.translate.side_effect = Exception("Translation failed")
        mock_google_translator_class.return_value = mock_translator_instance

        result = translate_subtitles_google(self.sample_subtitles, "en", "de")

        # Should fall back to original text on error
        self.assertEqual(result[0].content, "Hello world!")
        self.assertEqual(result[1].content, "This is a test.\nWith multiple lines.")

    @patch("src.translators.translator_google.GoogleTranslator")
    def test_translate_subtitles_google_empty_subtitles(
        self, mock_google_translator_class
    ):
        """Test handling of empty subtitle list."""
        mock_translator_instance = MagicMock()
        mock_google_translator_class.return_value = mock_translator_instance

        result = translate_subtitles_google([], "en", "de")

        self.assertEqual(result, [])
        # GoogleTranslator should still be instantiated (but not used)
        mock_google_translator_class.assert_called_once_with(source="en", target="de")
        # translate method should not be called
        mock_translator_instance.translate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
