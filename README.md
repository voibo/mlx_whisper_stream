# mlx_whisper_stream

Real-time speech recognition tool optimized for macOS Apple Silicon, used in Voibo. It receives audio data from standard input, detects speech segments with Silero VAD, and transcribes them using mlx_whisper.

## Features

- High-speed processing with MLX framework optimized for Apple Silicon
- Efficient speech segment detection using Silero VAD
- High-accuracy Japanese speech recognition with Whisper Large-v3-turbo model
- Simple pipeline processing using standard input/output
- Debug features (speech segment saving, log output)

## Requirements

- macOS (with Apple Silicon chip)
- Python 3.11 or higher
- The following dependencies:
  - numpy
  - torch
  - torchaudio
  - mlx-whisper
  - silero-vad
  - librosa
  - soundfile

## Installation

```bash
# Clone the repository
git clone https://github.com/voibo/mlx_whisper_stream.git
cd mlx_whisper_stream

# Create and configure virtual environment for Voibo integration
python3 -m venv venv
source ./venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download Whisper model (first time only)
mkdir -p model

# Download the MLX-optimized Whisper model from Hugging Face
# Option 1: Using git clone (requires git-lfs)
git lfs install
git clone https://huggingface.co/mlx-community/whisper-large-v3-turbo model/whisper-large-v3-turbo

# Option 2: Manual download
# Visit https://huggingface.co/mlx-community/whisper-large-v3-turbo
# Download the necessary model files (weights.safetensors, config.json, etc.)
# Place them in the model/whisper-large-v3-turbo directory
```

## Usage

### Basic Usage

```bash
# Example using ffmpeg:
ffmpeg -i sample.wav -ar 16000 -ac 1 -f f32le - | python mlx_whisper_stream.py
```

### Input Format

- Reads PCM audio data in 32-bit floating point (f32le), 16kHz, mono from standard input
- Data is processed in continuous chunks

### Output Format

- Recognition results are output in JSON format to standard output
- Each segment is output in the following format:

```json
{ "start": 1.25, "end": 3.75, "text": "Transcribed text" }
```

## Configuration

Main settings are defined at the beginning of mlx_whisper_stream.py:

```python
DEBUG = False                          # Enable/disable debug mode
BASE_PATH = "/path/to/mlx_whisper_stream"  # Base path of the project
DEBUG_AUDIO_PATH = f"{BASE_PATH}/output/"  # Debug output location
SAMPLE_RATE = 16000                    # Sample rate
```

VAD settings:

```python
vad_iterator = FixedVADIterator(
    model,
    threshold=0.5,                    # Speech detection threshold
    sampling_rate=SAMPLE_RATE,
    min_silence_duration_ms=500,      # Minimum duration to determine as silence (ms)
    speech_pad_ms=100                 # Padding added before and after speech segments (ms)
)
```

## Debug Mode

When `DEBUG = True` is set, the following features are enabled:

- Creates an `output/session_YYYYMMDD_HHMMSS/` directory at session start
- Saves detected speech segments as NumPy arrays (.npy)
  - File name format: `segment_NNNN_start_end.npy`
- Saves raw input audio data as binary files (.bin) every 10 seconds
  - File name format: `raw_SSSS.bin` (SSSS is the time in seconds)
- Outputs detailed logs to standard error output

## Troubleshooting

- **Error:** "UserWarning: The given NumPy array is not writable..."

  - This is a harmless warning. The program will work normally.

- **Low recognition accuracy:**
  - Try adjusting the VAD threshold (`threshold`)
  - Try adjusting the `min_silence_duration_ms` and `speech_pad_ms` parameters
  - Check the quality of the input audio

## License

This project is released under the MIT License.

## Acknowledgements

- [Silero VAD](https://github.com/snakers4/silero-vad) - Voice Activity Detection
- [MLX Whisper](https://huggingface.co/mlx-community/whisper-large-v3-turbo) - Fast speech recognition for Apple Silicon
- [OpenAI Whisper](https://github.com/openai/whisper) - Base speech recognition model
