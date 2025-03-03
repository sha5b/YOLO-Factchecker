import os
import sys
import argparse
import yaml
import torch
import numpy as np
import cv2
from datetime import timedelta
from pathlib import Path
from tqdm import tqdm
import logging
from concurrent.futures import ThreadPoolExecutor

# Local imports
from src.yolo_detector import YOLODetector
from src.speech_recognizer import WhisperTranscriber
from src.inconsistency_detector import InconsistencyDetector
from src.fact_checker import FactChecker
from src.knowledge_base import KnowledgeBase
from src.llm_interface import LocalLLM
from src.visualizer import ResultVisualizer
from src.utils.video_utils import VideoProcessor
from src.utils.timing import Timer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("pipeline")

class TruthDetectionPipeline:
    """
    End-to-end pipeline for analyzing videos, detecting inconsistencies,
    and fact-checking claims using local models.
    """
    
    def __init__(self, config_path="config/config.yml"):
        """Initialize the pipeline with configuration."""
        self.timer = Timer()
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        logger.info("Initializing Truth Detection Pipeline")
        
        # Load models based on configuration
        self._initialize_components()
        
    def _initialize_components(self):
        """Initialize all pipeline components."""
        with self.timer.measure("initialization"):
            # Initialize vision component
            model_path = os.path.join(self.config["paths"]["models"], "yolo")
            self.yolo = YOLODetector(
                model_path=model_path,
                confidence_threshold=self.config["yolo"]["confidence_threshold"],
                device=self.config["device"]
            )
            
            # Initialize speech recognition
            whisper_model = self.config["whisper"]["model_size"]
            whisper_path = os.path.join(self.config["paths"]["models"], "whisper")
            self.transcriber = WhisperTranscriber(
                model_size=whisper_model,
                model_path=whisper_path,
                device=self.config["device"]
            )
            
            # Initialize knowledge base
            kb_path = self.config["paths"]["knowledge_base"]
            self.kb = KnowledgeBase(
                db_path=kb_path,
                embedding_model=self.config["knowledge_base"]["embedding_model"]
            )
            
            # Initialize LLM for analysis
            llm_path = os.path.join(self.config["paths"]["models"], "llm")
            self.llm = LocalLLM(
                model_path=llm_path,
                model_type=self.config["llm"]["model_type"],
                quantization=self.config["llm"]["quantization"],
                max_tokens=self.config["llm"]["max_tokens"],
                temperature=self.config["llm"]["temperature"]
            )
            
            # Initialize analyzers
            self.inconsistency_detector = InconsistencyDetector(
                llm=self.llm,
                threshold=self.config["analysis"]["inconsistency_threshold"]
            )
            
            self.fact_checker = FactChecker(
                knowledge_base=self.kb,
                llm=self.llm,
                confidence_threshold=self.config["analysis"]["fact_check_threshold"]
            )
            
            # Initialize visualizer if needed
            if self.config["visualize"]["enabled"]:
                self.visualizer = ResultVisualizer(
                    output_dir=self.config["paths"]["output"],
                    visualization_type=self.config["visualize"]["type"]
                )
            else:
                self.visualizer = None
                
            logger.info("All components initialized successfully")
    
    def process_video(self, video_path, output_dir=None, segment_length=60):
        """
        Process a video file through the entire pipeline.
        
        Args:
            video_path: Path to video file
            output_dir: Directory to save results (default: config output path)
            segment_length: Length of video segments to process in seconds
        
        Returns:
            Dictionary containing analysis results
        """
        if output_dir is None:
            output_dir = self.config["paths"]["output"]
        
        os.makedirs(output_dir, exist_ok=True)
        
        video_path = Path(video_path)
        filename = video_path.stem
        
        logger.info(f"Processing video: {video_path}")
        
        # Create video processor
        video_processor = VideoProcessor(str(video_path))
        total_frames = video_processor.total_frames
        fps = video_processor.fps
        duration = video_processor.duration
        
        logger.info(f"Video info: {duration:.2f}s, {fps:.2f} fps, {total_frames} frames")
        
        # Extract audio for speech recognition
        with self.timer.measure("audio_extraction"):
            audio_path = os.path.join(output_dir, f"{filename}_audio.wav")
            video_processor.extract_audio(audio_path)
        
        # Perform speech recognition
        with self.timer.measure("speech_recognition"):
            transcription = self.transcriber.transcribe(audio_path)
            
            # Save transcription
            transcript_path = os.path.join(output_dir, f"{filename}_transcript.json")
            self.transcriber.save_transcription(transcription, transcript_path)
            
            logger.info(f"Transcription completed: {len(transcription['segments'])} segments")
        
        # Process video frames with YOLO
        with self.timer.measure("yolo_detection"):
            # Process in segments to manage memory
            segment_frames = int(segment_length * fps)
            num_segments = (total_frames + segment_frames - 1) // segment_frames
            
            all_detections = []
            
            for i in tqdm(range(num_segments), desc="Processing video segments"):
                start_frame = i * segment_frames
                end_frame = min(start_frame + segment_frames, total_frames)
                
                frames = video_processor.get_frames(start_frame, end_frame)
                detections = self.yolo.detect_batch(frames)
                
                # Add frame numbers to detections
                for j, frame_dets in enumerate(detections):
                    frame_num = start_frame + j
                    frame_time = frame_num / fps
                    
                    for det in frame_dets:
                        det['frame'] = frame_num
                        det['timestamp'] = frame_time
                    
                    all_detections.extend(frame_dets)
            
            # Save detections
            detection_path = os.path.join(output_dir, f"{filename}_detections.json")
            self.yolo.save_detections(all_detections, detection_path)
            
            logger.info(f"YOLO detection completed: {len(all_detections)} detections")
        
        # Analyze inconsistencies between speech and vision
        with self.timer.measure("inconsistency_analysis"):
            inconsistencies = self.inconsistency_detector.analyze(
                transcription=transcription,
                detections=all_detections,
                fps=fps
            )
            
            # Save inconsistencies
            inconsistency_path = os.path.join(output_dir, f"{filename}_inconsistencies.json")
            self.inconsistency_detector.save_results(inconsistencies, inconsistency_path)
            
            logger.info(f"Inconsistency analysis completed: {len(inconsistencies)} potential issues")
        
        # Fact-check claims
        with self.timer.measure("fact_checking"):
            claims = self._extract_claims(transcription, inconsistencies)
            fact_checks = self.fact_checker.check_claims(claims)
            
            # Save fact checks
            factcheck_path = os.path.join(output_dir, f"{filename}_factchecks.json")
            self.fact_checker.save_results(fact_checks, factcheck_path)
            
            logger.info(f"Fact-checking completed: {len(fact_checks)} claims analyzed")
        
        # Visualize results if requested
        if self.visualizer:
            with self.timer.measure("visualization"):
                output_video_path = os.path.join(output_dir, f"{filename}_analyzed.mp4")
                self.visualizer.create_visualization(
                    video_path=str(video_path),
                    transcription=transcription,
                    detections=all_detections,
                    inconsistencies=inconsistencies,
                    fact_checks=fact_checks,
                    output_path=output_video_path
                )
                logger.info(f"Visualization saved to {output_video_path}")
        
        # Compile all results
        results = {
            "video_info": {
                "path": str(video_path),
                "duration": duration,
                "fps": fps,
                "total_frames": total_frames
            },
            "transcription": {
                "path": transcript_path,
                "segments": len(transcription["segments"])
            },
            "detections": {
                "path": detection_path,
                "count": len(all_detections)
            },
            "inconsistencies": {
                "path": inconsistency_path,
                "count": len(inconsistencies)
            },
            "fact_checks": {
                "path": factcheck_path,
                "count": len(fact_checks)
            },
            "timing": self.timer.get_measurements(),
            "output_dir": output_dir
        }
        
        # Save overall results summary
        summary_path = os.path.join(output_dir, f"{filename}_summary.json")
        with open(summary_path, 'w') as f:
            import json
            json.dump(results, f, indent=2)
        
        logger.info(f"Processing completed. Results saved to {output_dir}")
        return results
    
    def _extract_claims(self, transcription, inconsistencies):
        """
        Extract claims from transcription based on inconsistencies.
        
        Args:
            transcription: Whisper transcription dict
            inconsistencies: List of detected inconsistencies
            
        Returns:
            List of claim dictionaries to fact-check
        """
        claims = []
        
        # Extract claims from all segments with inconsistencies
        inconsistent_segments = set([inc["segment_id"] for inc in inconsistencies])
        
        # Also include claims with high confidence regardless of visual inconsistency
        for segment in transcription["segments"]:
            segment_id = segment["id"]
            text = segment["text"]
            start = segment["start"]
            end = segment["end"]
            
            # Ask LLM to identify claims in this text
            is_inconsistent = segment_id in inconsistent_segments
            
            # LLM-based claim extraction
            extracted_claims = self.llm.extract_claims(text)
            
            for claim in extracted_claims:
                claims.append({
                    "text": claim,
                    "source_text": text,
                    "segment_id": segment_id,
                    "start_time": start,
                    "end_time": end,
                    "from_inconsistency": is_inconsistent,
                    "confidence": 1.0 if is_inconsistent else 0.7  # Higher priority for inconsistent segments
                })
        
        return claims
    
    def process_batch(self, video_dir, output_dir=None, extensions=[".mp4", ".avi", ".mov"]):
        """
        Process multiple videos in a directory.
        
        Args:
            video_dir: Directory containing videos
            output_dir: Directory to save results
            extensions: Video file extensions to process
        
        Returns:
            Dictionary with results for each video
        """
        video_dir = Path(video_dir)
        if output_dir is None:
            output_dir = self.config["paths"]["output"]
        
        # Find all video files
        video_files = []
        for ext in extensions:
            video_files.extend(list(video_dir.glob(f"*{ext}")))
        
        logger.info(f"Found {len(video_files)} videos to process")
        
        results = {}
        for video_path in tqdm(video_files, desc="Processing videos"):
            video_output_dir = os.path.join(output_dir, video_path.stem)
            os.makedirs(video_output_dir, exist_ok=True)
            
            try:
                video_results = self.process_video(
                    video_path=str(video_path),
                    output_dir=video_output_dir
                )
                results[video_path.name] = video_results
            except Exception as e:
                logger.error(f"Error processing {video_path}: {str(e)}")
                results[video_path.name] = {"error": str(e)}
        
        # Save batch results
        batch_summary_path = os.path.join(output_dir, "batch_summary.json")
        with open(batch_summary_path, 'w') as f:
            import json
            json.dump(results, f, indent=2)
            
        return results

def main():
    parser = argparse.ArgumentParser(description="Truth Detection Pipeline")
    parser.add_argument("--video", type=str, help="Path to video file")
    parser.add_argument("--video_dir", type=str, help="Directory of videos for batch processing")
    parser.add_argument("--output", type=str, help="Output directory")
    parser.add_argument("--config", type=str, default="config/config.yml", help="Configuration file")
    
    args = parser.parse_args()
    
    if not args.video and not args.video_dir:
        parser.error("Either --video or --video_dir must be provided")
    
    # Initialize pipeline
    pipeline = TruthDetectionPipeline(config_path=args.config)
    
    if args.video:
        pipeline.process_video(
            video_path=args.video,
            output_dir=args.output
        )
    elif args.video_dir:
        pipeline.process_batch(
            video_dir=args.video_dir,
            output_dir=args.output
        )

if __name__ == "__main__":
    main()
