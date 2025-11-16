import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.config.config_loader import get_from_env_or_json


class TestConfigLoader(unittest.TestCase):
    """Test configuration loading from environment and JSON files."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_key = "TEST_CONFIG_KEY"
        self.test_value = "test_value"

    def tearDown(self):
        """Clean up after tests."""
        # Remove test environment variable if it exists
        if self.test_key in os.environ:
            del os.environ[self.test_key]

    def test_get_from_env(self):
        """Test getting value from environment variable."""
        os.environ[self.test_key] = self.test_value
        result = get_from_env_or_json(self.test_key)
        self.assertEqual(result, self.test_value)

    def test_get_from_json_file(self):
        """Test getting value from JSON config file."""
        config_data = {self.test_key: self.test_value}

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_path = Path(f.name)

        try:
            result = get_from_env_or_json(self.test_key, temp_path)
            self.assertEqual(result, self.test_value)
        finally:
            temp_path.unlink()

    def test_env_takes_precedence_over_json(self):
        """Test that environment variable takes precedence over JSON file."""
        env_value = "env_value"
        json_value = "json_value"

        os.environ[self.test_key] = env_value

        config_data = {self.test_key: json_value}

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_path = Path(f.name)

        try:
            result = get_from_env_or_json(self.test_key, temp_path)
            self.assertEqual(result, env_value)  # Should be env value, not json
        finally:
            temp_path.unlink()

    def test_key_not_found(self):
        """Test returning None when key is not found in env or JSON."""
        result = get_from_env_or_json("NONEXISTENT_KEY")
        self.assertIsNone(result)

    def test_json_file_not_found(self):
        """Test handling when JSON file doesn't exist."""
        result = get_from_env_or_json(self.test_key, Path("nonexistent.json"))
        self.assertIsNone(result)

    def test_invalid_json_file(self):
        """Test handling of invalid JSON file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content {")
            temp_path = Path(f.name)

        try:
            result = get_from_env_or_json(self.test_key, temp_path)
            self.assertIsNone(result)
        finally:
            temp_path.unlink()

    def test_default_config_path(self):
        """Test using default config.json path when no path provided."""
        config_data = {self.test_key: self.test_value}

        # Create config.json in current directory
        config_path = Path("config.json")
        with open(config_path, 'w') as f:
            json.dump(config_data, f)

        try:
            result = get_from_env_or_json(self.test_key)
            self.assertEqual(result, self.test_value)
        finally:
            config_path.unlink()


if __name__ == '__main__':
    unittest.main()