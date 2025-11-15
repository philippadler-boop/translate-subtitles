import json
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load .env once at import time
load_dotenv()


def get_from_env_or_json(key: str, json_path: Path | None = None) -> Optional[str]:
    """
    Read a secret / config value from environment first,
    then from a JSON config file if provided.
    """
    value = os.getenv(key)
    if value:
        return value

    if json_path is None:
        json_path = Path("config.json")

    if json_path.is_file():
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            return data.get(key)
        except Exception:
            # Ignore JSON parse errors for now
            return None

    return None
