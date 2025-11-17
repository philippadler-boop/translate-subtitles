import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv

# Load .env once at import time so users can put secrets in a .env file
load_dotenv()


def _deep_get(d: Dict[str, Any], keys: str, default: Any = None) -> Any:
    """Get a nested value from a dict using a dot-separated key path.

    Example: _deep_get(cfg, "translation.deepl.auth_key")
    """
    parts = keys.split(".") if isinstance(keys, str) else list(keys)
    cur = d
    for p in parts:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(p, default)
        if cur is default:
            return default
    return cur


def _deep_set(d: Dict[str, Any], keys: str, value: Any) -> None:
    parts = keys.split(".") if isinstance(keys, str) else list(keys)
    cur = d
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def load_config(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load configuration from `config.json` (defaults to project root).

    Environment variables take precedence for secrets:
      - `DEEPL_AUTH_KEY` -> `translation.deepl.auth_key`
      - `OPENAI_API_KEY` -> `translation.openai.api_key`

    Returns a plain dict. If no file exists, returns an empty dict.
    """
    if path is None:
        path = Path.cwd() / "config.json"

    if not path.is_file():
        return {}

    try:
        cfg = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    # Apply environment overrides for known secret keys
    deepl_key = os.getenv("DEEPL_AUTH_KEY") or os.getenv("DEEPL_AUTH")
    if deepl_key:
        _deep_set(cfg, "translation.deepl.auth_key", deepl_key)

    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        _deep_set(cfg, "translation.openai.api_key", openai_key)

    return cfg


def get_from_env_or_config(dot_path: str, cfg: Optional[Dict[str, Any]] = None, default: Any = None) -> Any:
    """Get a value from environment first, then from the loaded config dict.

    `dot_path` is a dot-separated path in the JSON config, e.g. "translation.deepl.auth_key".
    """
    # Environment shortcut: support top-level env var names matching important keys
    env_map = {
        "translation.deepl.auth_key": "DEEPL_AUTH_KEY",
        "translation.openai.api_key": "OPENAI_API_KEY",
    }

    env_name = env_map.get(dot_path)
    if env_name:
        env_val = os.getenv(env_name)
        if env_val:
            return env_val

    if cfg is None:
        cfg = load_config()

    return _deep_get(cfg, dot_path, default)


def get_from_env_or_json(key: str, json_path: Optional[Path] = None) -> Optional[str]:
    """Backward-compatible helper used by older modules.

    Behavior:
      - Check environment variable named `key` first.
      - Then check top-level `key` in `config.json` (or provided `json_path`).
    """
    # First check environment variable directly
    value = os.getenv(key)
    if value:
        return value

    # Support legacy top-level names mapping to nested config paths
    legacy_map = {
        "DEEPL_API_KEY": "translation.deepl.auth_key",
        "OPENAI_API_KEY": "translation.openai.api_key",
    }
    mapped = legacy_map.get(key)
    if mapped:
        # Prefer env var mapping (already checked), then config.json nested path
        cfg = load_config() if (json_path is None or not Path(json_path).is_file()) else json.loads(Path(json_path).read_text(encoding="utf-8"))
        return _deep_get(cfg, mapped)

    if json_path is None:
        json_path = Path.cwd() / "config.json"

    if json_path.is_file():
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            return data.get(key)
        except Exception:
            return None

    return None

