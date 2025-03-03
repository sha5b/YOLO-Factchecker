import os
import time
import json
import uuid
import logging
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file, Response
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import yaml

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("yolo_factchecker")

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'yolo-factchecker-secret-key'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max upload size
socketio = SocketIO(app)

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Create other necessary directories
os.makedirs('static/results', exist_ok=True)
os.makedirs('models/yolo', exist_ok=True)
os.makedirs('models/vosk', exist_ok=True)
os.makedirs('data/knowledge_base', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)

# Load configuration
CONFIG_PATH = 'config.yml'
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
else:
    # Default configuration
    config = {
        "yolo": {
            "model": "yolov8s.pt",
            "confidence_threshold": 0.5,
            "img_size": 640,
            "sample_rate": 5  # Process every Nth frame
        },
        "vosk": {
            "model_name": "vosk-model-en-us-0.22",
            "sample_rate": 16000,
            "enable_words": True,
            "language": "en-us",
            "alternative_results": 1
        },
        "ollama": {
            "base_url": "http://localhost:11434",
            "model": "llama3",
            "temperature": 0.2,
            "max_tokens": 1024
        }
    }
    # Save default config
    with open(CONFIG_PATH, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

# Import custom modules
from src.video_processor import VideoProcessor
from src.yolo_detector import YOLODetector
from src.speech_recognizer import VoskTranscriber
from src.ollama_interface import OllamaInterface

# Initialize components (lazy loading)
video_processor = None
yolo_detector = None
transcriber = None
ollama_interface = None

def initialize_components():
    """Initialize processing components when needed"""
    global video_processor, yolo_detector, transcriber, ollama_interface
    
    try:
        if yolo_detector is None:
            logger.info("Initializing YOLO detector")
            yolo_detector = YOLODetector(
                model_path=config["yolo"]["model"],
                confidence_threshold=config["yolo"]["confidence_threshold"],
                img_size=config["yolo"]["img_size"]
            )
        
        if transcriber is None:
            logger.info("Initializing Vosk transcriber")
            transcriber = VoskTranscriber(
                model_name=config["vosk"]["model_name"],
                sample_rate=config["vosk"]["sample_rate"],
                enable_words=config["vosk"]["enable_words"],
                language=config["vosk"]["language"],
                alternative_results=config["vosk"]["alternative_results"]
            )
        
        if ollama_interface is None:
            logger.info("Initializing Ollama interface")
            ollama_interface = OllamaInterface(
                base_url=config["ollama"]["base_url"],
                model=config["ollama"]["model"],
                temperature=config["ollama"]["temperature"],
                max_tokens=config["ollama"]["max_tokens"]
            )
    except Exception as e:
        logger.error(f"Error initializing components: {str(e)}")
        raise

@app.route('/')
def index():
    """Render the main page with upload form"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    """Handle video upload and start processing"""
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({'error': 'No video selected'}), 400
    
    # Generate unique ID for this processing session
    session_id = str(uuid.uuid4())
    
    # Save the uploaded video
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_{video_file.filename}")
    video_file.save(video_path)
    
    # Create results directory for this session
    results_dir = os.path.join('static/results', session_id)
    os.makedirs(results_dir, exist_ok=True)
    
    # Redirect to processing page
    return jsonify({
        'success': True,
        'session_id': session_id,
        'video_path': video_path,
        'redirect': f'/process/{session_id}'
    })

@app.route('/process/<session_id>')
def process_page(session_id):
    """Render the processing page"""
    # Find the video file for this session
    video_files = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if f.startswith(session_id)]
    if not video_files:
        return "Session not found", 404
    
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_files[0])
    video_filename = video_files[0].replace(f"{session_id}_", "")
    
    return render_template('process.html', 
                          session_id=session_id, 
                          video_path=video_path,
                          video_filename=video_filename)

@socketio.on('start_processing')
def handle_start_processing(data):
    """Start processing the video"""
    session_id = data['session_id']
    video_path = data['video_path']
    
    # Initialize components if needed
    initialize_components()
    
    # Normalize path to ensure proper format
    video_path = os.path.normpath(video_path)
    
    # If path is relative, make it absolute
    if not os.path.isabs(video_path):
        video_path = os.path.join(os.getcwd(), video_path)
    
    # Check if file exists
    if not os.path.exists(video_path):
        # Try to find the file in the uploads directory
        uploads_dir = os.path.join(os.getcwd(), 'static', 'uploads')
        for filename in os.listdir(uploads_dir):
            if session_id in filename:
                video_path = os.path.join(uploads_dir, filename)
                logger.info(f"Found video file: {video_path}")
                break
    
    # Create video processor for this session
    global video_processor
    try:
        video_processor = VideoProcessor(video_path)
    except ValueError as e:
        logger.error(f"Error creating video processor: {str(e)}")
        emit('processing_error', {
            'session_id': session_id,
            'error': f"Could not open video file. Please try uploading again.",
            'component': 'video'
        })
        return
    
    # Get video info
    total_frames = video_processor.total_frames
    fps = video_processor.fps
    duration = video_processor.duration
    
    # Send video info to client
    emit('video_info', {
        'total_frames': total_frames,
        'fps': fps,
        'duration': duration,
        'width': video_processor.width,
        'height': video_processor.height
    })
    
    # Create results directory
    results_dir = os.path.join('static/results', session_id)
    os.makedirs(results_dir, exist_ok=True)
    
    # Extract audio for transcription
    audio_path = os.path.join(results_dir, "audio.wav")
    video_processor.extract_audio(audio_path)
    
    # Start transcription in background
    socketio.start_background_task(
        process_audio, 
        session_id=session_id, 
        audio_path=audio_path
    )
    
    # Start video processing in background
    socketio.start_background_task(
        process_video, 
        session_id=session_id, 
        video_path=video_path,
        results_dir=results_dir
    )

def process_audio(session_id, audio_path):
    """Process audio for transcription"""
    try:
        # Transcribe audio
        transcription = transcriber.transcribe(audio_path)
        
        # Save transcription
        results_dir = os.path.join('static/results', session_id)
        transcript_path = os.path.join(results_dir, "transcript.json")
        transcriber.save_transcription(transcription, transcript_path)
        
        # Send transcription segments to client
        for segment in transcription['segments']:
            socketio.emit('transcription_segment', {
                'segment': segment,
                'session_id': session_id
            })
            
            # Send to Ollama for analysis
            prompt = f"Analyze this statement for factual accuracy: '{segment['text']}'"
            response = ollama_interface.generate(prompt)
            
            # Send LLM response to client
            socketio.emit('llm_response', {
                'segment_id': segment['id'],
                'response': response,
                'session_id': session_id
            })
            
            # Small delay to avoid overwhelming the client
            time.sleep(0.1)
        
        # Signal completion
        socketio.emit('transcription_complete', {
            'session_id': session_id,
            'transcript_path': transcript_path
        })
        
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        socketio.emit('processing_error', {
            'session_id': session_id,
            'error': str(e),
            'component': 'audio'
        })

def process_video(session_id, video_path, results_dir):
    """Process video frames with YOLO"""
    try:
        # Process in segments to manage memory
        sample_rate = config["yolo"]["sample_rate"]
        all_detections = []
        
        # Get total frames
        total_frames = video_processor.total_frames
        processed_count = 0
        
        # Current frame index (for sequential processing)
        current_frame_idx = 0
        
        # Process frames sequentially
        while current_frame_idx < total_frames:
            # Get frame
            frame = video_processor.get_frame(current_frame_idx)
            if frame is None:
                current_frame_idx += sample_rate
                continue
            
            # Run detection
            detections = yolo_detector.detect(frame)
            
            # Check for faces with expressions and analyze body language
            has_face = False
            has_significant_expression = False
            body_language_analysis = []
            facial_expression_analysis = []
            
            for det in detections:
                # Add frame information
                det['frame'] = current_frame_idx
                det['timestamp'] = current_frame_idx / video_processor.fps
                
                # Check if this is a person detection
                if det['class_name'] == 'person':
                    has_face = True
                    
                    # Extract person bounding box
                    x1, y1, x2, y2 = map(int, det["bbox"])
                    person_height = y2 - y1
                    
                    # Estimate face region (top 20% of person bounding box)
                    face_y1 = y1
                    face_y2 = y1 + int(person_height * 0.2)
                    face_x1 = x1
                    face_x2 = x2
                    
                    # Analyze facial expression (simplified for now)
                    # In a real implementation, you would use a facial expression detection model
                    facial_expression = "Neutral"  # Default
                    expression_confidence = 0.7
                    
                    # Add facial expression analysis
                    facial_expression_analysis.append({
                        'bbox': [face_x1, face_y1, face_x2, face_y2],
                        'expression': facial_expression,
                        'confidence': expression_confidence
                    })
                    
                    # Analyze body language (simplified for now)
                    # In a real implementation, you would use pose estimation
                    body_posture = "Standing"  # Default
                    posture_confidence = 0.8
                    
                    # Add body language analysis
                    body_language_analysis.append({
                        'bbox': det["bbox"],
                        'posture': body_posture,
                        'confidence': posture_confidence
                    })
                    
                    # For now, we'll consider all person detections as potentially having expressions
                    has_significant_expression = True
            
            all_detections.extend(detections)
            
            # Draw detections on frame with enhanced visibility
            drawn_frame = yolo_detector._draw_detections(frame, detections, enhanced=True)
            
            # Convert to JPEG for sending
            _, buffer = cv2.imencode('.jpg', drawn_frame)
            img_str = buffer.tobytes()
            
            # Send frame to client
            socketio.emit('processed_frame', {
                'frame': img_str.hex(),
                'frame_idx': current_frame_idx,
                'detections': detections,
                'session_id': session_id,
                'progress': min(100, int(100 * current_frame_idx / total_frames)),
                'has_face': has_face,
                'has_expression': has_significant_expression,
                'facial_expressions': facial_expression_analysis,
                'body_language': body_language_analysis
            })
            
            processed_count += 1
            
            # If we detect a face with significant expression, analyze the audio at this timestamp
            if has_face and has_significant_expression:
                # Get the timestamp for this frame
                timestamp = current_frame_idx / video_processor.fps
                
                # Find transcription segments that overlap with this timestamp
                matching_segments = []
                try:
                    for segment in transcriber.get_segments_at_time(timestamp):
                        matching_segments.append(segment)
                except Exception as e:
                    logger.warning(f"Error getting segments at time {timestamp}: {str(e)}")
                
                # If we have matching segments, analyze them with the LLM
                if matching_segments:
                    for segment in matching_segments:
                        # Send to Ollama for analysis
                        prompt = f"Analyze this statement for factual accuracy: '{segment['text']}'"
                        response = ollama_interface.generate(prompt)
                        
                        # Send LLM response to client
                        socketio.emit('llm_response', {
                            'segment_id': segment['id'],
                            'response': response,
                            'session_id': session_id,
                            'frame_idx': current_frame_idx,
                            'timestamp': timestamp,
                            'facial_expressions': facial_expression_analysis,
                            'body_language': body_language_analysis
                        })
                        
                        # Wait for LLM to complete before continuing
                        # This ensures we don't overwhelm the system and provides synchronization
                        time.sleep(0.5)
                
                # Instead of pausing for user approval, we wait for the LLM response
                # to complete before continuing with the next frame
                
                # Add a small delay after processing a significant frame
                # to allow the client to display the results
                time.sleep(0.5)  # Shorter delay, just enough to display results
            else:
                # Smaller delay for frames without significant expressions
                time.sleep(0.05)
            
            # Move to next frame
            current_frame_idx += sample_rate
        
        # Save all detections
        detection_path = os.path.join(results_dir, "detections.json")
        with open(detection_path, 'w') as f:
            json.dump(all_detections, f, indent=2)
        
        # Signal completion
        socketio.emit('video_processing_complete', {
            'session_id': session_id,
            'processed_frames': processed_count,
            'total_detections': len(all_detections),
            'detection_path': detection_path
        })
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        socketio.emit('processing_error', {
            'session_id': session_id,
            'error': str(e),
            'component': 'video'
        })

@app.route('/download/<session_id>')
def download_summary(session_id):
    """Generate and download summary"""
    results_dir = os.path.join('static/results', session_id)
    
    # Check if results exist
    if not os.path.exists(results_dir):
        return "Results not found", 404
    
    # Load detections and transcription
    detection_path = os.path.join(results_dir, "detections.json")
    transcript_path = os.path.join(results_dir, "transcript.json")
    
    detections = []
    transcription = {"segments": []}
    
    if os.path.exists(detection_path):
        with open(detection_path, 'r') as f:
            detections = json.load(f)
    
    if os.path.exists(transcript_path):
        with open(transcript_path, 'r') as f:
            transcription = json.load(f)
    
    # Generate summary
    summary = {
        "session_id": session_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "detections": {
            "count": len(detections),
            "by_class": {}
        },
        "transcription": {
            "segments": len(transcription["segments"]),
            "text": transcription.get("text", "")
        },
        "analysis": []
    }
    
    # Count detections by class
    for det in detections:
        class_name = det["class_name"]
        if class_name not in summary["detections"]["by_class"]:
            summary["detections"]["by_class"][class_name] = 0
        summary["detections"]["by_class"][class_name] += 1
    
    # Save summary
    summary_path = os.path.join(results_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Return the file
    return send_file(summary_path, as_attachment=True, download_name="factcheck_summary.json")

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
