**Overview**

This document describes how to set up the translate-subtitles project on Windows (PowerShell). It includes instructions for creating a Python virtual environment, installing runtime and optional native dependencies (notably `webrtcvad`), installing Visual Studio Build Tools if you want the native `webrtcvad` wheel, and running tests and common CLI commands.

**Prerequisites**

- **Python 3.10+ installed**: Ensure `python` is on your PATH.
- **Git**: for cloning/pushing the repo.
- **PowerShell** (Windows) — commands in this guide assume PowerShell 5.1.
- **ffmpeg**: required for audio extraction. Download a Windows build and add the `ffmpeg` binary directory to your PATH.

**Workspace layout (what we created)**

- `workspaces/input/{video,audio,subs}`: Put input files here (video/audio/subs).
- `workspaces/output/{video,audio,subs}`: Destination for processed files.
- `src/` : source code (CLI, audio_io, asr, vad, subtitle_sync, translators).
- `tests/` : unit tests.

**1) Create and activate virtual environment (PowerShell)**

```
python -m venv .venv
.\.venv\Scripts\Activate
python -m pip install -U pip
```

**2) Install Python dependencies**

Install the project's required Python packages. This will attempt to build any packages that need compilation (e.g., `webrtcvad`) unless you install a prebuilt wheel first.

```
python -m pip install -r requirements.txt
```

If you encounter build errors for `webrtcvad` (common on Windows), see section "Installing webrtcvad on Windows" below.

**3) ffmpeg (audio extraction)**

- Download a Windows build from https://ffmpeg.org/download.html or use a package manager
- Add the folder containing `ffmpeg.exe` to your PATH
- Verify with:

```
ffmpeg -version
```

**4) Installing VS Build Tools (if you want to build native wheels)**

If you prefer to install `webrtcvad` from source on Windows, you need Microsoft C++ Build Tools.

- Download and install "Build Tools for Visual Studio" from:
  https://visualstudio.microsoft.com/visual-cpp-build-tools/
- During installation, select **"C++ build tools"** (include MSVC, Windows SDK, and CMake if offered).
- After install, restart PowerShell and reinstall dependencies:

```
python -m pip install --upgrade pip setuptools wheel
python -m pip install webrtcvad
python -m pip install -r requirements.txt
```

Note: Building may take time and requires disk space.

**5) Installing `webrtcvad` without Build Tools (recommended quick option)**

Option A: `pipwin` (installs prebuilt Windows wheels):

```
python -m pip install pipwin
python -m pipwin install webrtcvad
```

Option B: Conda (if you use Anaconda/Miniconda):

```
conda install -c conda-forge webrtcvad
```

Either option avoids the need for Visual C++ build tools.

**6) Notes about `webrtcvad` in this project**

- The project attempts to import `webrtcvad` in `src/vad.py`. If `webrtcvad` is not available the code falls back to a simple energy-based VAD so the pipeline still runs (but with less accuracy).
- If you want the best segmentation quality, prefer installing `webrtcvad` via `pipwin` or installing VS Build Tools and then `pip install webrtcvad`.

**7) GPU support and ASR (Hugging Face transformers pipeline)**

- This project now uses the Hugging Face `transformers` ASR pipeline for local
  transcription. To enable GPU acceleration install a CUDA-enabled `torch` build
  first (matching your CUDA toolkit/driver), then install `transformers`:

```
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu118 torch torchvision --upgrade
python -m pip install transformers
```

If you do not have a GPU, CPU mode works but is slower.

**8) Running tests**

Run the test suite from the repository root (PowerShell):

```
python -m pytest -q
```

If tests create temporary artifacts, this repo uses `.gitignore` entries to avoid committing them. If a test still writes to the repo root, check test fixtures — we updated `tests/test_audio_io.py` to use temporary directories.

**9) Common CLI usage**

- Translate an SRT using HuggingFace offline model:
```
python main.py "workspaces/input/subs/movie.srt" --src-lang en --tgt-lang de --engine hf
```

- Regenerate SRT from video using ASR (local Whisper):
```
python main.py "workspaces/input/video/movie.mp4" --tgt-lang en --regenerate --asr-model small --device auto
```

Notes:
- `--device auto|cpu|cuda` selects ASR device.
- Use `--asr-model` to pick a local Whisper model (small/medium/large). Large models require more RAM.

**10) Utility scripts**

- `scripts/move_to_output.py <path>` — move processed files into `workspaces/output/{video,audio,subs}` by extension.

Example:
```
python scripts/move_to_output.py workspaces/input/subs
```

**11) Troubleshooting**

- "error: Microsoft Visual C++ 14.0 or greater is required" — install Visual C++ Build Tools (see section 4) or use `pipwin` to get prebuilt `webrtcvad` wheel.
- If `ffmpeg` is not found, ensure `ffmpeg.exe` directory is on your PATH and restart PowerShell.
- If ASR fails with memory errors on GPU, try a smaller `--asr-model` or use `--device cpu`.

**12) Next recommended steps**

- If you plan on production use, install `webrtcvad` via `pipwin` or conda for best VAD quality.
- Add a GitHub Actions workflow to run tests on push/PR.
- Optionally add a pre-commit hook to prevent committing large binaries.

If you want, I can add a short `docs/README.md` or expand this into a `CONTRIBUTING.md` with developer workflows.
