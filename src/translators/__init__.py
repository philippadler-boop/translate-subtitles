"""Translator engines package.

Expose module attributes so unittest.mock.patch paths like
`src.translators.translator_hf.translate_subtitles_hf` resolve correctly.
"""

from . import translator_google  # noqa: F401
from . import translator_deepl   # noqa: F401
from . import translator_gpt     # noqa: F401
from . import translator_hf      # noqa: F401

__all__ = [
    "translator_google",
    "translator_deepl",
    "translator_gpt",
    "translator_hf",
]
