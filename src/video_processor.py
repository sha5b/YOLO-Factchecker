import os
import cv2
import numpy as np
import subprocess
import logging
from typing import List, Optional, Tuple

logger = logging.getLogger("video_processor")

class VideoProcessor:
    """
    Utility class for video processing operations.
    Handles frame extraction, audio extraction, and basic video operations.
    """
    
    def __init__(self, video_path: str):
        """
        Initialize video processor.
        
        Args:
            video_path: Path to video file
        """
        self.video_path = video_path
        
        # Open video file
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.total_frames / self.fps if self.fps > 0 else 0
        
        logger.info(f"Loaded video: {video_path}")
        logger.info(f"Properties: {self.width}x{self.height}, {self.fps} fps, {self.duration:.2f}s, {self.total_frames} frames")
    
    def __del__(self):
        """Clean up resources."""
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
    
    def get_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        """
        Get a specific frame from the video.
        
        Args:
            frame_idx: Frame index
            
        Returns:
            Frame as numpy array or None if frame could not be read
        """
        if frame_idx >= self.total_frames:
            logger.warning(f"Frame index {frame_idx} exceeds total frames {self.total_frames}")
            return None
        
        # Set position
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        
        # Read frame
        ret, frame = self.cap.read()
        if not ret:
            logger.warning(f"Could not read frame {frame_idx}")
            return None
        
        return frame
    
    def get_frames(self, start_frame: int, end_frame: int) -> List[np.ndarray]:
        """
        Get a range of frames from the video.
        
        Args:
            start_frame: Starting frame index (inclusive)
            end_frame: Ending frame index (exclusive)
            
        Returns:
            List of frames as numpy arrays
        """
        frames = []
        
        # Set position to start frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Read frames
        for _ in range(end_frame - start_frame):
            ret, frame = self.cap.read()
            if not ret:
                break
            frames.append(frame)
        
        return frames
    
    def extract_audio(self, output_path: str) -> bool:
        """
        Extract audio from video using FFmpeg.
        
        Args:
            output_path: Path to save extracted audio
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            try:
                # Extract audio using ffmpeg with more conservative settings to avoid threading issues
                command = [
                    "ffmpeg", "-i", self.video_path, 
                    "-vn", "-acodec", "pcm_s16le", 
                    "-ar", "16000", "-ac", "1", 
                    "-threads", "1",  # Use single thread to avoid multithreading issues
                    "-filter_threads", "1",  # Single thread for filters
                    "-filter_complex_threads", "1",  # Single thread for filter complex
                    "-safe", "0",  # Avoid safety checks that might cause issues
                    "-f", "wav",  # Explicitly specify WAV format
                    "-y",  # Overwrite output file if it exists
                    "-loglevel", "error",  # Only show errors
                    output_path
                ]
                
                logger.info(f"Extracting audio to {output_path} using ffmpeg")
                subprocess.run(command, check=True)
                
                return os.path.exists(output_path)
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                logger.error(f"FFmpeg error: {str(e)}")
                # Create an empty WAV file if FFmpeg fails
                self._create_empty_wav(output_path)
                logger.warning(f"Created empty WAV file at {output_path} (FFmpeg failed)")
                return False
        
        except Exception as e:
            logger.error(f"Error extracting audio: {str(e)}")
            # Create an empty audio file in case of error
            try:
                self._create_empty_wav(output_path)
                logger.warning(f"Created empty WAV file at {output_path} after error")
            except Exception as e2:
                logger.error(f"Error creating empty WAV file: {str(e2)}")
            return False
    
    def get_frame_at_time(self, timestamp: float) -> Optional[np.ndarray]:
        """
        Get frame at specific timestamp.
        
        Args:
            timestamp: Time in seconds
            
        Returns:
            Frame as numpy array or None if frame could not be read
        """
        if timestamp < 0 or timestamp > self.duration:
            logger.warning(f"Timestamp {timestamp} out of range [0, {self.duration}]")
            return None
        
        # Convert timestamp to frame index
        frame_idx = int(timestamp * self.fps)
        return self.get_frame(frame_idx)
    
    def save_frame(self, frame: np.ndarray, output_path: str) -> bool:
        """
        Save frame to file.
        
        Args:
            frame: Frame as numpy array
            output_path: Path to save frame
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # Save frame
            cv2.imwrite(output_path, frame)
            return os.path.exists(output_path)
        
        except Exception as e:
            logger.error(f"Error saving frame: {str(e)}")
            return False
    
    def _create_empty_wav(self, output_path: str) -> None:
        """
        Create an empty WAV file with a valid header.
        
        Args:
            output_path: Path to save the empty WAV file
        """
        try:
            with open(output_path, 'wb') as f:
                # Write a minimal valid WAV file header
                f.write(b'RIFF')
                f.write((36).to_bytes(4, byteorder='little'))  # File size - 8
                f.write(b'WAVE')
                f.write(b'fmt ')
                f.write((16).to_bytes(4, byteorder='little'))  # Chunk size
                f.write((1).to_bytes(2, byteorder='little'))   # PCM format
                f.write((1).to_bytes(2, byteorder='little'))   # Mono
                f.write((16000).to_bytes(4, byteorder='little'))  # Sample rate
                f.write((32000).to_bytes(4, byteorder='little'))  # Byte rate
                f.write((2).to_bytes(2, byteorder='little'))   # Block align
                f.write((16).to_bytes(2, byteorder='little'))  # Bits per sample
                f.write(b'data')
                f.write((0).to_bytes(4, byteorder='little'))  # Chunk size
            logger.warning(f"Created empty WAV file at {output_path}")
        except Exception as e:
            logger.error(f"Error creating empty WAV file: {str(e)}")
    
    def get_metadata(self) -> dict:
        """
        Get video metadata using ffprobe.
        
        Returns:
            Dictionary of metadata
        """
        try:
            # Get metadata using ffprobe
            command = [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_format", "-show_streams", self.video_path
            ]
            
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            
            # Parse JSON output
            import json
            metadata = json.loads(result.stdout)
            
            return metadata
        
        except subprocess.CalledProcessError as e:
            logger.error(f"Error getting metadata: {str(e)}")
            return {}
        except Exception as e:
            logger.error(f"Unexpected error getting metadata: {str(e)}")
            return {}
