# Truth Detector

A local-first video analysis system that combines computer vision (YOLO), speech recognition, and local LLM capabilities to detect inconsistencies and fact-check content.

## Overview

Truth Detector analyzes videos by:
1. **Detecting visual elements** using YOLOv8
2. **Transcribing speech** with local Whisper models
3. **Identifying inconsistencies** between visual content and speech
4. **Fact-checking claims** against a local knowledge base
5. **Generating reports** that highlight potential misinformation

All processing is done locally without relying on external APIs, making this suitable for private analysis, research projects, and artistic installations.

## System Requirements

- **Python**: 3.9+ 
- **CUDA support** (optional but recommended for GPU acceleration)
- **Storage**: 
  - ~5GB for models (YOLO, Whisper, embeddings)
  - 8-20GB for LLM (depending on model choice)
- **Memory**: 
  - Minimum: 16GB RAM
  - Recommended: 32GB+ RAM for larger models
- **GPU**: 
  - Optional but recommended (8GB+ VRAM for decent performance)
  - Required for real-time analysis

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/truth-detector.git
cd truth-detector
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download models

```bash
python scripts/download_models.py
```

This will download:
- YOLOv8 model (default: YOLOv8n)
- Whisper model (default: base)
- Sentence transformer for embeddings
- Default LLM (if specified in config)

### 5. Build knowledge base

```bash
python scripts/build_knowledge_base.py --source path/to/knowledge/files
```

## Usage

### Basic Analysis

```bash
python pipelines/full_pipeline.py --video path/to/video.mp4 --output path/to/output
```

### Batch Processing

```bash
python pipelines/full_pipeline.py --video_dir path/to/videos --output path/to/output
```

### Configuration

Edit `config/config.yml` to customize:
- Model sizes and parameters
- Analysis thresholds
- Visualization options
- Hardware utilization

## Components

### 1. YOLO Detection (`src/yolo_detector.py`)

Detects objects, people, and scenes in video frames using YOLOv8. Customizable with different model sizes for speed/accuracy tradeoffs.

### 2. Speech Recognition (`src/speech_recognizer.py`)

Transcribes speech from video using Whisper models, with optional speaker diarization. Supports multiple languages and optimized inference.

### 3. Inconsistency Detection (`src/inconsistency_detector.py`)

Identifies mismatches between visual content and spoken claims using a combination of rule-based filtering and local LLM analysis.

### 4. Fact Checking (`src/fact_checker.py`)

Verifies claims against a local knowledge base using vector similarity search and LLM-based verification.

### 5. Knowledge Base (`src/knowledge_base.py`)

Local vector database for storing factual information, built using FAISS or Chroma DB for efficient similarity search.

### 6. LLM Interface (`src/llm_interface.py`)

Interface to run language models locally for analysis, reasoning, and fact verification.

## LLM Integration

The system is designed to work with various open-source LLMs:

- **Default**: Llama 3 8B or similar compact model
- **Options**: Mistral, Gemma, or other compatible models
- **Quantization**: int4/int8 quantization supported for faster inference
- **Inference engines**: llama.cpp, vLLM, ExLLama2

## Knowledge Base Sources

To build an effective knowledge base for fact-checking:

1. Gather text documents with factual information
2. Place them in `data/knowledge_base/sources/`
3. Run `python scripts/build_knowledge_base.py`

Recommended sources:
- Wikipedia dumps (filtered to topics of interest)
- Trusted reference materials
- Domain-specific databases
- Curated fact repositories

## Customization

### Training Custom YOLO Models

```bash
python scripts/train_yolo.py --data path/to/dataset --epochs 100 --batch 16
```

### Extending the System

The modular design allows adding:
- Additional detection models
- Custom fact-checking logic
- Alternative LLMs
- Specialized visualizations

## Development Roadmap

- [ ] Implement core pipeline with YOLOv8 and Whisper
- [ ] Add basic fact-checking with local vector search
- [ ] Integrate LLM for reasoning and analysis
- [ ] Develop visualization and reporting
- [ ] Add web interface (optional)
- [ ] Optimize for performance on limited hardware

## Troubleshooting

### Common Issues

1. **Out of memory errors**: 
   - Reduce model sizes in config
   - Process video in smaller segments
   - Use more aggressive quantization for LLM

2. **Slow processing**:
   - Increase frame sampling rate (process fewer frames)
   - Use smaller models
   - Enable GPU acceleration if available

3. **Missing dependencies**:
   - Install system libraries: `apt-get install ffmpeg libsndfile1`
   - Update GPU drivers for CUDA support

## Resources and References

- YOLOv8: [Ultralytics Documentation](https://docs.ultralytics.com/)
- Whisper: [OpenAI Whisper Documentation](https://github.com/openai/whisper)
- Faster-Whisper: [Faster Whisper Implementation](https://github.com/guillaumekln/faster-whisper)
- LLMs: [llama.cpp](https://github.com/ggerganov/llama.cpp)
- Vector Search: [FAISS Documentation](https://github.com/facebookresearch/faiss)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

This project combines multiple open-source technologies to create a privacy-focused video analysis system.
