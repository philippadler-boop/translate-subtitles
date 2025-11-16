# Testing

This directory contains comprehensive unit tests for the translate-subtitles project.

## Running Tests

### Using the test runner
```bash
python run_tests.py
```

### Using pytest directly
```bash
pytest
```

### With coverage
```bash
pytest --cov=src --cov-report=html
```

## Test Structure

- `test_srt_io.py` - Tests for SRT file reading/writing
- `test_config_loader.py` - Tests for configuration loading
- `test_cli.py` - Tests for command-line argument parsing
- `test_translator_hf.py` - Tests for HuggingFace translator
- `test_translator_google.py` - Tests for Google Translate translator
- `test_main_integration.py` - Integration tests for main CLI functionality

## Test Coverage

The tests cover:
- ✅ SRT file I/O operations
- ✅ Configuration loading from environment and JSON
- ✅ CLI argument parsing and validation
- ✅ HuggingFace model selection and auto-correction
- ✅ Google Translate integration (mocked)
- ✅ Error handling and fallback behavior
- ✅ Integration testing of main CLI workflow

## Mocking Strategy

External API calls are mocked to ensure:
- Tests run quickly without network dependencies
- Tests are deterministic and reliable
- No API rate limits or costs during testing
- Tests can run in CI/CD environments