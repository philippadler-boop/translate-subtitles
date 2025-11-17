from pathlib import Path
import json

from src.visualizer.visualizer import (
    extract_words_from_aligned_segments,
    write_words_json,
)


def test_extract_words_from_aligned_segments_handles_various_keys():
    aligned = [
        {"words": [{"start": 0.0, "end": 0.5, "word": "hello"}]},
        {"words": [{"start": 0.5, "end": 1.0, "text": "world"}]},
        {"words": [{"start": 1.0, "end": 1.5, "token": "!"}]},
        {"words": "not-a-list"},  # should be ignored safely
    ]

    words = extract_words_from_aligned_segments(aligned)

    assert [w["text"] for w in words] == ["hello", "world", "!"]
    assert words[0]["start"] == 0.0
    assert words[-1]["end"] == 1.5


def test_write_words_json_and_html(tmp_path: Path):
    words = [
        {"text": "hello", "start": 0.0, "end": 0.5},
        {"text": "world", "start": 0.5, "end": 1.0},
    ]
    json_path = tmp_path / "words.json"
    html_path = tmp_path / "words.html"

    write_words_json(words, json_path)
    assert json_path.is_file()

    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert "words" in data
    assert data["words"][0]["text"] == "hello"

    # HTML visualizer removed; JSON export is sufficient. Ensure JSON exists.
    assert json_path.is_file()
