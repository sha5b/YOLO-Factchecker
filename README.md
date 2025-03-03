# YOLO Factchecker

A Flask-based web application that processes videos using YOLO object detection, speech recognition, and local LLM (Ollama) for fact-checking. The application also analyzes facial expressions and body language when people are detected in the video.

## Features

- Upload and process videos
- Real-time YOLO object detection with visualization
- Automatic facial expression and body language analysis
- Intelligent video pausing when significant expressions are detected
- Sequential video processing with frame-by-frame analysis
- Offline speech transcription using Vosk
- Fact-checking using local Ollama LLM
- Live processing visualization with detailed analytics
- Downloadable summary of results

## Requirements

- Python 3.9+
- FFmpeg (required for audio extraction)
- CUDA-compatible GPU (optional but recommended)
- [Ollama](https://ollama.ai/) installed locally

## Installation

1. Install FFmpeg (required for audio extraction):
   - Windows:
     ```bash
     # Using Chocolatey (recommended)
     choco install ffmpeg -y
     
     # Or download from https://ffmpeg.org/download.html and add to PATH
     ```
   - Linux:
     ```bash
     # Ubuntu/Debian
     sudo apt update && sudo apt install ffmpeg
     
     # CentOS/RHEL
     sudo dnf install ffmpeg
     ```
   - macOS:
     ```bash
     # Using Homebrew
     brew install ffmpeg
     ```

2. Clone the repository:
```bash
git clone https://github.com/yourusername/YOLO-Factchecker.git
cd YOLO-Factchecker
```

3. Create a virtual environment:
```bash
python -m venv venv
```

4. Activate the virtual environment:
   - Windows:
   ```bash
   venv\Scripts\activate
   ```
   - Linux/Mac:
   ```bash
   source venv/bin/activate
   ```

5. Install dependencies:
```bash
pip install -r requirements.txt
```

6. Make sure Ollama is running with the required model:
```bash
ollama run llama3
```

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open your browser and navigate to:
```
http://localhost:5000
```

3. Upload a video file and watch the processing in real-time.

4. Download the summary when processing is complete.

## Configuration

You can customize the application by editing the `config.yml` file:

- YOLO model and settings
- Speech recognition settings (language, sample rate)
- Ollama LLM settings
- Processing parameters

## Project Structure

```
YOLO-Factchecker/
├── app.py                  # Main Flask application
├── config.yml              # Configuration file
├── requirements.txt        # Dependencies
├── src/                    # Source code
│   ├── __init__.py
│   ├── video_processor.py  # Video processing utilities
│   ├── yolo_detector.py    # YOLO implementation
│   ├── speech_recognizer.py # Speech recognition
│   └── ollama_interface.py # Interface to local Ollama
├── models/                 # Model files
│   └── yolo/               # YOLO model files
├── static/                 # Static assets
│   ├── css/                # Stylesheets
│   ├── js/                 # JavaScript files
│   ├── img/                # Images
│   ├── uploads/            # Uploaded videos
│   └── results/            # Processing results
└── templates/              # Flask templates
    ├── index.html          # Main page
    └── process.html        # Processing page
```

## How It Works

1. **Video Upload**: User uploads a video through the web interface.
2. **Processing**:
   - Video frames are processed with YOLO to detect objects
   - Audio is extracted and transcribed with Vosk (offline speech recognition)
   - Transcribed text is analyzed by Ollama LLM
3. **Visualization**:
   - Detected objects are highlighted in video frames
   - Transcription is displayed in real-time
   - LLM analysis is shown alongside
4. **Summary**: A downloadable JSON summary is generated with all results.

## Speech Recognition

The application uses the Vosk library for offline speech recognition to transcribe audio from videos. The transcription process:

1. Extracts audio from the video using FFmpeg
2. Converts the audio to the proper format (WAV, 16kHz, mono)
3. Loads the Vosk model from the `models/vosk` directory
4. Processes the audio in small chunks using Vosk's streaming API
5. Creates segments with transcription text and timestamps

You can configure the speech recognition settings in the `config.yml` file:

```yaml
speech_recognition:
  language: "en-US"    # Language code (e.g., en-US, fr-FR, de-DE)
  sample_rate: 16000   # Audio sample rate in Hz
  word_timestamps: true # Whether to include word timestamps
```

### Vosk Model Setup

The application requires a Vosk model to be placed in the `models/vosk` directory. You can download models from the [Vosk website](https://alphacephei.com/vosk/models). For English, we recommend using the `vosk-model-small-en-us-0.22` model.

1. Download the model from the Vosk website
2. Extract the model files to the `models/vosk` directory
3. Make sure the model files are directly in the `models/vosk` directory (not in a subdirectory)

Note: Vosk provides completely offline speech recognition, so no internet connection is required for transcription. This makes the application more reliable and privacy-friendly. If no speech is detected or transcription fails, a placeholder segment will be created to allow the rest of the processing to continue.

## Troubleshooting

- **Model Download Issues**: The first run may take time as models are downloaded.
- **Memory Errors**: Reduce model sizes in config.yml if you encounter memory issues.
- **Performance**: Adjust the sample_rate in config.yml to process fewer frames for better performance.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
