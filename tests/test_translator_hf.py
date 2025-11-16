import unittest
from unittest.mock import patch, MagicMock

import srt
import pytest

from src.translators.translator_hf import (
    translate_subtitles_hf,
    _default_hf_model_for_pair,
)
pytest.importorskip("transformers")
pytest.importorskip("torchvision")


class TestTranslatorHF(unittest.TestCase):
    """Test HuggingFace translator functionality."""

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
                content="This is a test.",
            ),
        ]

    def test_default_hf_model_for_pair_known_pairs(self):
        """Test model selection for known language pairs."""
        test_cases = [
            ("en", "de", "Helsinki-NLP/opus-mt-en-de"),
            ("de", "en", "Helsinki-NLP/opus-mt-de-en"),
            ("en", "fr", "Helsinki-NLP/opus-mt-en-fr"),
            ("fr", "en", "Helsinki-NLP/opus-mt-fr-en"),
        ]

        for src, tgt, expected in test_cases:
            with self.subTest(src=src, tgt=tgt):
                result = _default_hf_model_for_pair(src, tgt)
                self.assertEqual(result, expected)

    def test_default_hf_model_for_pair_unknown_pair(self):
        """Test that unknown language pairs raise RuntimeError."""
        with self.assertRaises(RuntimeError) as cm:
            _default_hf_model_for_pair("unknown", "lang")

        self.assertIn("No default model configured", str(cm.exception))
        self.assertIn("unknown-lang", str(cm.exception))

    @patch("src.translators.translator_hf.get_from_env_or_json")
    @patch("src.translators.translator_hf.pipeline")
    def test_translate_subtitles_hf_success(self, mock_pipeline, mock_get_config):
        """Test successful translation with mocked pipeline."""
        # Mock configuration - no HF_MODEL set, use default
        mock_get_config.return_value = None

        # Mock the translation pipeline
        mock_translator = MagicMock()
        mock_translator.return_value = [{"translation_text": "Hallo Welt!"}]
        mock_pipeline.return_value = mock_translator

        result = translate_subtitles_hf(self.sample_subtitles, "en", "de")

        # Verify results
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].content, "Hallo Welt!")
        self.assertEqual(result[1].content, "Hallo Welt!")  # Same mock result

        # Verify pipeline was called with correct model
        mock_pipeline.assert_called_once_with(
            "translation", model="Helsinki-NLP/opus-mt-en-de"
        )

    @patch("src.translators.translator_hf.get_from_env_or_json")
    @patch("src.translators.translator_hf.pipeline")
    def test_translate_subtitles_hf_with_custom_model(
        self, mock_pipeline, mock_get_config
    ):
        """Test translation with custom HF_MODEL from config."""
        custom_model = "custom/model-name"
        mock_get_config.return_value = custom_model

        mock_translator = MagicMock()
        mock_translator.return_value = [{"translation_text": "Translated text"}]
        mock_pipeline.return_value = mock_translator

        _result = translate_subtitles_hf(self.sample_subtitles, "en", "de")

        # Verify custom model was used
        mock_pipeline.assert_called_once_with("translation", model=custom_model)

    @patch("src.translators.translator_hf.get_from_env_or_json")
    @patch("src.translators.translator_hf.pipeline")
    def test_translate_subtitles_hf_auto_correct_en_de_to_de_en(
        self, mock_pipeline, mock_get_config
    ):
        """Test auto-correction when HF_MODEL is en-de but translating de->en."""
        # Config has en->de model but we're translating de->en
        mock_get_config.return_value = "Helsinki-NLP/opus-mt-en-de"

        mock_translator = MagicMock()
        mock_translator.return_value = [{"translation_text": "Translated"}]
        mock_pipeline.return_value = mock_translator

        with patch("builtins.print") as mock_print:
            _result = translate_subtitles_hf(self.sample_subtitles, "de", "en")

            # Should auto-correct to de-en model
            mock_pipeline.assert_called_once_with(
                "translation", model="Helsinki-NLP/opus-mt-de-en"
            )

            # Should print warning and model usage
            self.assertEqual(mock_print.call_count, 2)
            warning_call = mock_print.call_args_list[0]
            model_call = mock_print.call_args_list[1]
            self.assertIn("Warning", warning_call[0][0])
            self.assertIn("en->de model", warning_call[0][0])
            self.assertIn("de->en translation", warning_call[0][0])
            self.assertIn("Using model", model_call[0][0])

    @patch("src.translators.translator_hf.get_from_env_or_json")
    @patch("src.translators.translator_hf.pipeline")
    def test_translate_subtitles_hf_auto_correct_de_en_to_en_de(
        self, mock_pipeline, mock_get_config
    ):
        """Test auto-correction when HF_MODEL is de-en but translating en->de."""
        mock_get_config.return_value = "Helsinki-NLP/opus-mt-de-en"

        mock_translator = MagicMock()
        mock_translator.return_value = [{"translation_text": "Translated"}]
        mock_pipeline.return_value = mock_translator

        with patch("builtins.print") as mock_print:
            _result = translate_subtitles_hf(self.sample_subtitles, "en", "de")

            mock_pipeline.assert_called_once_with(
                "translation", model="Helsinki-NLP/opus-mt-en-de"
            )

            # Should print warning and model usage
            self.assertEqual(mock_print.call_count, 2)
            warning_call = mock_print.call_args_list[0]
            model_call = mock_print.call_args_list[1]
            self.assertIn("Warning", warning_call[0][0])
            self.assertIn("de->en model", warning_call[0][0])
            self.assertIn("en->de translation", warning_call[0][0])
            self.assertIn("Using model", model_call[0][0])

    @patch("src.translators.translator_hf.get_from_env_or_json")
    @patch("src.translators.translator_hf.pipeline")
    def test_translate_subtitles_hf_translation_error(
        self, mock_pipeline, mock_get_config
    ):
        """Test handling of translation errors."""
        mock_get_config.return_value = None

        mock_translator = MagicMock()
        # First call succeeds, second fails
        mock_translator.side_effect = [
            [{"translation_text": "Success"}],
            Exception("Translation failed"),
        ]
        mock_pipeline.return_value = mock_translator

        result = translate_subtitles_hf(self.sample_subtitles, "en", "de")

        # First subtitle should be translated
        self.assertEqual(result[0].content, "Success")
        # Second should fall back to original text
        self.assertEqual(result[1].content, "This is a test.")

    def test_translate_subtitles_hf_preserves_metadata(self):
        """Test that subtitle metadata is preserved during translation."""
        with patch(
            "src.translators.translator_hf.get_from_env_or_json"
        ) as mock_get_config:
            with patch("src.translators.translator_hf.pipeline") as mock_pipeline:
                mock_get_config.return_value = None

                mock_translator = MagicMock()
                mock_translator.return_value = [{"translation_text": "Translated"}]
                mock_pipeline.return_value = mock_translator

                result = translate_subtitles_hf(self.sample_subtitles, "en", "de")

                # Check that all metadata is preserved
                for original, translated in zip(self.sample_subtitles, result):
                    self.assertEqual(original.index, translated.index)
                    self.assertEqual(original.start, translated.start)
                    self.assertEqual(original.end, translated.end)
                    self.assertEqual(original.proprietary, translated.proprietary)


if __name__ == "__main__":
    unittest.main()
