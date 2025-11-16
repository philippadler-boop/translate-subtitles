"""Top-level package initializer.

Expose subpackages needed by tests that use unittest.mock.patch with
attribute traversal (e.g. patch("src.translators.translator_google.translate_subtitles_google")).

Importing the subpackage here makes it available as an attribute of the
`src` module so the patch resolution logic can access it.
"""

from . import translators  # noqa: F401

__all__ = ["translators"]
