from pathlib import Path
import json
from typing import List, Dict, Any


def extract_words_from_aligned_segments(
    aligned_segments: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Flatten aligned segments into a list of word dicts with start/end/text."""
    words = []
    for seg in aligned_segments:
        # whisperx segments sometimes contain 'words' list
        for w in seg.get("words", []) if isinstance(seg.get("words", []), list) else []:
            # expect word dict with 'start','end','word'
            word_text = w.get("word") or w.get("text") or w.get("token") or ""
            words.append(
                {
                    "text": word_text,
                    "start": float(w.get("start", 0.0)),
                    "end": float(w.get("end", 0.0)),
                }
            )
    return words


def write_words_json(words: List[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"words": words}, f, ensure_ascii=False, indent=2)

