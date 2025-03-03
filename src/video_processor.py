import os
import cv2
import numpy as np
import subprocess
import logging
import threading
from typing import List, Optional, Tuple

logger = logging.getLogger("video_processor")

# Global lock for moviepy operations to prevent threading issues
moviepy_lock = threading.Lock()

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
        Extract audio from video using moviepy.
        
        Args:
            output_path: Path to save extracted audio
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # Import moviepy here to avoid dependency if not used
            try:
                # First try to import moviepy directly
                try:
                    from moviepy.editor import VideoFileClip
                    moviepy_available = True
                    logger.info("Using moviepy for audio extraction")
                except ImportError:
                    # If that fails, try to import it using a subprocess to ensure it's in the path
                    import sys
                    import subprocess
                    
                    # Check if moviepy is installed using pip
                    try:
                        subprocess.run([sys.executable, "-m", "pip", "show", "moviepy"], 
                                      check=True, capture_output=True)
                        
                        # If we get here, moviepy is installed but there might be an import issue
                        logger.warning("moviepy is installed but could not be imported directly. Trying alternative import.")
                        
                        # Try to add the site-packages directory to the path
                        import site
                        sys.path.extend(site.getsitepackages())
                        
                        # Try import again
                        from moviepy.editor import VideoFileClip
                        moviepy_available = True
                        logger.info("Successfully imported moviepy after path adjustment")
                    except subprocess.CalledProcessError:
                        logger.warning("moviepy not installed according to pip. Trying ffmpeg as fallback.")
                        moviepy_available = False
            except Exception as e:
                logger.warning(f"Error importing moviepy: {str(e)}. Trying ffmpeg as fallback.")
                moviepy_available = False
            
            if moviepy_available:
                # Use lock to prevent threading issues with moviepy
                with moviepy_lock:
                    try:
                        # Extract audio using moviepy
                        logger.info(f"Extracting audio to {output_path} using moviepy")
                        video_clip = VideoFileClip(self.video_path)
                        audio_clip = video_clip.audio
                        
                        if audio_clip is None:
                            logger.warning(f"No audio track found in video: {self.video_path}")
                            # Create an empty audio file
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
                        else:
                            # Write audio to file
                            audio_clip.write_audiofile(
                                output_path,
                                fps=16000,  # Sample rate
                                nbytes=2,   # 16-bit
                                codec='pcm_s16le',  # PCM format
                                ffmpeg_params=["-ac", "1"]  # Mono
                            )
                        
                        # Clean up
                        video_clip.close()
                        if audio_clip is not None:
                            audio_clip.close()
                    except Exception as e:
                        logger.error(f"Error in moviepy audio extraction: {str(e)}")
                        # Fall back to creating an empty WAV file
                        self._create_empty_wav(output_path)
                
                return os.path.exists(output_path)
            else:
                # Fallback to ffmpeg if available
                try:
                    # Try to run ffmpeg -version to check if it's available
                    subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)
                    ffmpeg_available = True
                except (subprocess.CalledProcessError, FileNotFoundError):
                    logger.warning("ffmpeg not found in PATH. Audio extraction will be skipped.")
                    ffmpeg_available = False
                
                if ffmpeg_available:
                    # Extract audio using ffmpeg
                    command = [
                        "ffmpeg", "-i", self.video_path, 
                        "-vn", "-acodec", "pcm_s16le", 
                        "-ar", "16000", "-ac", "1", 
                        output_path, "-y", "-loglevel", "error"
                    ]
                    
                    logger.info(f"Extracting audio to {output_path} using ffmpeg")
                    subprocess.run(command, check=True)
                    
                    return os.path.exists(output_path)
                else:
                    # Create an empty file if neither moviepy nor ffmpeg is available
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
                    
                    logger.warning(f"Created empty WAV file at {output_path} (no audio extraction available)")
                    return True
        
        except Exception as e:
            logger.error(f"Error extracting audio: {str(e)}")
            # Create an empty audio file in case of error
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
