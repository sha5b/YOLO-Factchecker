# YOLO Factchecker

A Flask-based web application that processes videos using YOLO object detection, speech recognition, and local LLM (Ollama) for fact-checking. The application also analyzes facial expressions and body language when people are detected in the video.

## Features

- Upload and process videos
- Real-time YOLO object detection with visualization
- Automatic facial expression and body language analysis
- Intelligent video pausing when significant expressions are detected
- Sequential video processing with frame-by-frame analysis
- Speech transcription using Vosk (lightweight offline speech recognition)
- Fact-checking using local Ollama LLM
- Live processing visualization with detailed analytics
- Downloadable summary of results

## Requirements

- Python 3.9+
- FFmpeg (required for audio extraction)
- CUDA-compatible GPU (optional but recommended)
- [Ollama](https://ollama.ai/) installed locally

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/YOLO-Factchecker.git
cd YOLO-Factchecker
```

2. Create a virtual environment:
```bash
python -m venv venv
```

3. Activate the virtual environment:
   - Windows:
   ```bash
   venv\Scripts\activate
   ```
   - Linux/Mac:
   ```bash
   source venv/bin/activate
   ```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

5. Make sure Ollama is running with the required model:
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
- Vosk speech recognition settings
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
   - Audio is extracted and transcribed with Vosk
   - Transcribed text is analyzed by Ollama LLM
3. **Visualization**:
   - Detected objects are highlighted in video frames
   - Transcription is displayed in real-time
   - LLM analysis is shown alongside
4. **Summary**: A downloadable JSON summary is generated with all results.

## Troubleshooting

- **Model Download Issues**: The first run may take time as models are downloaded.
- **Memory Errors**: Reduce model sizes in config.yml if you encounter memory issues.
- **Performance**: Adjust the sample_rate in config.yml to process fewer frames for better performance.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
