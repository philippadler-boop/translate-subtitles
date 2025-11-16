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


def write_simple_html_timeline(words_json_path: Path, html_path: Path) -> None:
    """Produce a tiny HTML that loads the JSON and visualizes word boxes on a timeline.

    This is intentionally minimal and dependency-free so it can be opened locally.
    """
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Word Timeline</title>
  <style>
    body {{ font-family: Arial, sans-serif; padding: 1rem; }}
    .timeline {{ position: relative; height: 120px; border: 1px solid #ccc; padding: 8px; }}
    .word {{ position: absolute; top: 20px; height: 24px; background: #007acc; color: #fff; padding: 2px 4px; border-radius: 3px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
    .label {{ margin-top: 8px; font-size: 0.9rem; color: #333; }}
  </style>
  </head>
<body>
  <h2>Word Timeline</h2>
  <div id="timeline" class="timeline"></div>
  <div class="label">Loaded from: {words_json_path.name}</div>
  <script>
    fetch('{words_json_path.name}').then(r=>r.json()).then(data=>{{
      const words = data.words || [];
      if(!words.length) return;
      const maxEnd = Math.max(...words.map(w=>w.end));
      const container = document.getElementById('timeline');
      const W = container.clientWidth || 800;
      words.forEach(w=>{{
        const left = (w.start / maxEnd) * 100;
        const width = Math.max(( (w.end - w.start) / maxEnd) * 100, 0.5);
        const div = document.createElement('div');
        div.className = 'word';
        div.style.left = left + '%';
        div.style.width = width + '%';
        div.title = `${{w.text}} (${{w.start.toFixed(2)}} - ${{w.end.toFixed(2)}})`;
        div.textContent = w.text;
        container.appendChild(div);
      }});
    }}).catch(e=>console.error(e));
  </script>
</body>
</html>"""

    html_path.parent.mkdir(parents=True, exist_ok=True)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
