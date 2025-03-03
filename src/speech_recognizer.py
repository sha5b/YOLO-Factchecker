import os
import json
import logging
import subprocess
import tempfile
import time
import wave
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple

from vosk import Model, KaldiRecognizer, SetLogLevel
import numpy as np

logger = logging.getLogger("speech_recognizer")

# Set Vosk log level (0 for most verbose, higher numbers for less verbose)
SetLogLevel(0)

class VoskTranscriber:
    """
    Speech recognition using Vosk, running locally without internet connection.
    """
    
    def __init__(
        self,
        model_size: str = "small",
        device: str = "cpu",
        language: Optional[str] = "en-US",
        beam_size: int = 5,
        sample_rate: int = 16000,
        word_timestamps: bool = True
    ):
        """
        Initialize Vosk transcriber.
        
        Args:
            model_size: Size of the model ('small', 'medium', 'large')
            device: Not used for Vosk
            language: Language code (e.g., 'en-US')
            beam_size: Not used for Vosk
            sample_rate: Audio sample rate in Hz
            word_timestamps: Whether to include word timestamps
        """
        self.language = language or "en-US"
        self.sample_rate = sample_rate
        self.word_timestamps = word_timestamps
        self._last_transcription = None
        self.model = None
        
        # Initialize the model
        model_path = self._get_model_path()
        if not os.path.exists(model_path):
            logger.warning(f"Vosk model not found at {model_path}. Please download it manually.")
        else:
            logger.info(f"Loading Vosk model from {model_path}")
            try:
                self.model = Model(model_path)
                logger.info("Vosk model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading Vosk model: {str(e)}")
    
    def _get_model_path(self) -> str:
        """
        Get the path to the Vosk model.
        
        Returns:
            Path to the Vosk model
        """
        # Default model path
        base_path = os.path.join("models", "vosk")
        
        # Check if model directory exists
        if not os.path.exists(base_path):
            os.makedirs(base_path, exist_ok=True)
            logger.warning(f"Created Vosk model directory at {base_path}")
        
        # Check if there are any model files in the directory
        model_files = os.listdir(base_path)
        if not model_files:
            logger.error(f"No model files found in {base_path}. Please download a Vosk model and extract it to this directory.")
            # Create a README file with instructions
            readme_path = os.path.join(base_path, "README.md")
            with open(readme_path, "w") as f:
                f.write("""# Vosk Model Directory

This directory should contain the Vosk model files. Please download a model from the [Vosk website](https://alphacephei.com/vosk/models) and extract it to this directory.

For English, we recommend using the `vosk-model-small-en-us-0.22` model.

1. Download the model from https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.22.zip
2. Extract the ZIP file
3. Copy all the files from the extracted directory directly into this directory (not in a subdirectory)
""")
            logger.info(f"Created README file at {readme_path} with instructions for downloading the Vosk model")
        else:
            logger.info(f"Found {len(model_files)} files in Vosk model directory")
            # Check if the model files are in a subdirectory
            for item in model_files:
                item_path = os.path.join(base_path, item)
                if os.path.isdir(item_path) and "model" in item.lower():
                    # If there's a subdirectory with "model" in the name, use that
                    logger.info(f"Found model subdirectory: {item}")
                    return item_path
        
        # Return the path to the model directory
        return base_path
    
    def transcribe(
        self,
        audio_path: str,
        output_format: str = "json",
        segment_level: bool = True,
        word_timestamps: bool = None
    ) -> Dict:
        """
        Transcribe audio file.
        
        Args:
            audio_path: Path to audio file
            output_format: Output format ('json', 'text', 'srt', 'vtt')
            segment_level: Whether to return segment-level timestamps
            word_timestamps: Not used, kept for compatibility
            
        Returns:
            Transcription dictionary
        """
        start_time = time.time()
        logger.info(f"Transcribing audio: {audio_path}")
        
        # Check if file exists and has content
        if not os.path.exists(audio_path):
            logger.error(f"Audio file not found: {audio_path}")
            return {"text": "", "segments": []}
        
        # Check file size
        file_size = os.path.getsize(audio_path)
        if file_size < 100:  # Very small file, likely empty or just a header
            logger.warning(f"Audio file is too small ({file_size} bytes), likely empty: {audio_path}")
            return {"text": "", "segments": []}
        
        try:
            # Prepare audio file
            audio_path = self._prepare_audio(audio_path)
            
            # Transcribe
            result = self._transcribe_audio(audio_path)
            
            # Format the results
            formatted_result = self._format_result(result, output_format)
            
            elapsed_time = time.time() - start_time
            audio_duration = self._get_audio_duration(audio_path)
            
            if audio_duration > 0:
                logger.info(f"Transcription completed in {elapsed_time:.2f}s for {audio_duration:.2f}s audio")
                logger.info(f"Real-time factor: {elapsed_time / audio_duration:.2f}x")
            else:
                logger.warning(f"Audio duration is 0, possibly empty file: {audio_path}")
            
            # Store the transcription for later use
            self._last_transcription = formatted_result
            
            return formatted_result
        except Exception as e:
            logger.error(f"Error during transcription: {str(e)}")
            return {"text": "", "segments": []}
    
    def _prepare_audio(self, audio_path: str) -> str:
        """
        Prepare audio file for transcription.
        Converts to proper format if needed.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Path to prepared audio file
        """
        # Check if file exists
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Get file extension
        _, ext = os.path.splitext(audio_path)
        
        # Always convert to ensure proper format
        logger.info(f"Converting audio file to proper format for transcription")
        
        # Create temp file
        temp_dir = tempfile.gettempdir()
        output_path = os.path.join(temp_dir, f"speech_audio_{int(time.time())}.wav")
        
        # Convert using ffmpeg
        try:
            subprocess.run([
                "ffmpeg", "-i", audio_path, 
                "-ar", str(self.sample_rate), 
                "-ac", "1", 
                "-c:a", "pcm_s16le", 
                "-threads", "1",  # Use single thread to avoid multithreading issues
                "-f", "wav",  # Explicitly specify WAV format
                output_path, "-y", "-loglevel", "error"
            ], check=True)
            
            # Verify the file exists and has content
            if os.path.exists(output_path) and os.path.getsize(output_path) > 100:
                logger.info(f"Audio converted successfully: {output_path}")
                return output_path
            else:
                logger.warning(f"Converted audio file is too small or doesn't exist: {output_path}")
                return audio_path
        
        except subprocess.CalledProcessError as e:
            logger.error(f"Error converting audio: {str(e)}")
            logger.warning("Using original file instead")
            return audio_path
    
    def _transcribe_audio(self, audio_path: str) -> Dict:
        """
        Transcribe audio file using Vosk.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Transcription result
        """
        if self.model is None:
            logger.error("Vosk model not loaded")
            return {"text": "", "segments": []}
        
        try:
            # Get audio duration
            audio_duration = self._get_audio_duration(audio_path)
            logger.info(f"Audio duration: {audio_duration:.2f} seconds")
            
            # Open the audio file
            wf = wave.open(audio_path, "rb")
            
            # Check if the audio format is compatible with Vosk
            if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
                logger.warning(f"Audio file has unsupported format, channels={wf.getnchannels()}, width={wf.getsampwidth()}")
                # Try to convert the file again with stricter parameters
                audio_path = self._prepare_audio(audio_path)
                wf = wave.open(audio_path, "rb")
            
            # Create a recognizer
            rec = KaldiRecognizer(self.model, wf.getframerate())
            
            # Enable word timestamps if requested
            if self.word_timestamps:
                rec.SetWords(True)
            
            # Process the audio file
            logger.info("Processing audio with Vosk")
            
            # Variables to store results
            full_text = ""
            segments = []
            segment_id = 0
            current_segment_start = 0
            
            # Read and process audio data in chunks
            chunk_size = 4000  # Process 4000 frames at a time
            while True:
                data = wf.readframes(chunk_size)
                if len(data) == 0:
                    break
                
                # Process audio chunk
                if rec.AcceptWaveform(data):
                    # Get result for this chunk
                    result_json = rec.Result()
                    result = json.loads(result_json)
                    
                    # Extract text
                    text = result.get("text", "").strip()
                    
                    if text:
                        # Calculate timestamps
                        if "result" in result and len(result["result"]) > 0:
                            # If we have detailed word results
                            words = result["result"]
                            if words:
                                segment_start = words[0].get("start", current_segment_start)
                                segment_end = words[-1].get("end", segment_start + 5.0)  # Default 5 seconds if no end time
                            else:
                                segment_start = current_segment_start
                                segment_end = current_segment_start + 5.0
                        else:
                            # Estimate timestamps based on audio position
                            segment_start = current_segment_start
                            segment_end = segment_start + 5.0  # Default 5 seconds if no timestamps
                        
                        # Create segment
                        segment = {
                            "id": segment_id,
                            "start": segment_start,
                            "end": segment_end,
                            "text": text,
                            "avg_logprob": 0.0,  # Not provided by Vosk
                            "compression_ratio": 1.0,  # Not provided by Vosk
                            "no_speech_prob": 0.0  # Not provided by Vosk
                        }
                        
                        segments.append(segment)
                        full_text += text + " "
                        logger.info(f"Transcribed segment {segment_id}: '{text}'")
                        
                        # Update for next segment
                        segment_id += 1
                        current_segment_start = segment_end
            
            # Get final result
            final_json = rec.FinalResult()
            final_result = json.loads(final_json)
            
            # Extract text from final result
            final_text = final_result.get("text", "").strip()
            
            if final_text and final_text != full_text.strip():
                # Calculate timestamps for final segment
                if "result" in final_result and len(final_result["result"]) > 0:
                    # If we have detailed word results
                    words = final_result["result"]
                    if words:
                        segment_start = words[0].get("start", current_segment_start)
                        segment_end = words[-1].get("end", segment_start + 5.0)
                    else:
                        segment_start = current_segment_start
                        segment_end = current_segment_start + 5.0
                else:
                    # Estimate timestamps based on audio position
                    segment_start = current_segment_start
                    segment_end = audio_duration
                
                # Create final segment
                segment = {
                    "id": segment_id,
                    "start": segment_start,
                    "end": segment_end,
                    "text": final_text,
                    "avg_logprob": 0.0,
                    "compression_ratio": 1.0,
                    "no_speech_prob": 0.0
                }
                
                segments.append(segment)
                full_text += final_text + " "
                logger.info(f"Transcribed final segment: '{final_text}'")
            
            # Close the audio file
            wf.close()
            
            # If no segments were created, add a placeholder
            if not segments:
                logger.warning("No speech detected or transcription failed")
                segment = {
                    "id": 0,
                    "start": 0,
                    "end": audio_duration,
                    "text": "[No speech detected or transcription failed]",
                    "avg_logprob": 0.0,
                    "compression_ratio": 1.0,
                    "no_speech_prob": 1.0
                }
                segments.append(segment)
                full_text = "[No speech detected or transcription failed]"
            
            return {
                "text": full_text.strip(),
                "segments": segments,
                "language": self.language
            }
            
        except Exception as e:
            logger.error(f"Error during Vosk transcription: {str(e)}")
            return {"text": "", "segments": []}
    
    def _format_result(self, result: Dict, output_format: str) -> Dict:
        """
        Format the transcription result.
        
        Args:
            result: Transcription result
            output_format: Output format
            
        Returns:
            Formatted result
        """
        if output_format == "json":
            return result
        
        elif output_format == "text":
            return {"text": result["text"]}
        
        elif output_format == "srt":
            srt_content = ""
            for i, segment in enumerate(result["segments"]):
                start_time = self._format_timestamp(segment["start"], output_format)
                end_time = self._format_timestamp(segment["end"], output_format)
                text = segment["text"].strip()
                
                srt_content += f"{i+1}\n{start_time} --> {end_time}\n{text}\n\n"
            
            return {"text": result["text"], "srt": srt_content}
        
        elif output_format == "vtt":
            vtt_content = "WEBVTT\n\n"
            for i, segment in enumerate(result["segments"]):
                start_time = self._format_timestamp(segment["start"], output_format)
                end_time = self._format_timestamp(segment["end"], output_format)
                text = segment["text"].strip()
                
                vtt_content += f"{start_time} --> {end_time}\n{text}\n\n"
            
            return {"text": result["text"], "vtt": vtt_content}
        
        else:
            logger.warning(f"Unknown output format: {output_format}, using json")
            return result
    
    def _format_timestamp(self, seconds: float, format_type: str) -> str:
        """
        Format timestamp.
        
        Args:
            seconds: Timestamp in seconds
            format_type: Format type ('srt' or 'vtt')
            
        Returns:
            Formatted timestamp
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        
        if format_type == "srt":
            # SRT format: 00:00:00,000
            return f"{hours:02d}:{minutes:02d}:{int(seconds):02d},{int((seconds % 1) * 1000):03d}"
        else:
            # VTT format: 00:00:00.000
            return f"{hours:02d}:{minutes:02d}:{int(seconds):02d}.{int((seconds % 1) * 1000):03d}"
    
    def _get_audio_duration(self, audio_path: str) -> float:
        """
        Get audio duration in seconds.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Duration in seconds
        """
        try:
            # Use ffprobe to get duration
            result = subprocess.run([
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", audio_path
            ], capture_output=True, text=True, check=True)
            
            duration = float(result.stdout.strip())
            return duration
        
        except (subprocess.CalledProcessError, ValueError):
            logger.warning("Could not determine audio duration")
            return 0.0
    
    def get_segments_at_time(self, timestamp: float, margin: float = 0.5) -> List[Dict]:
        """
        Get transcription segments that overlap with a given timestamp.
        
        Args:
            timestamp: Time in seconds
            margin: Time margin in seconds to consider for overlap
            
        Returns:
            List of segments that overlap with the timestamp
        """
        # Check if transcription is available
        if not hasattr(self, '_last_transcription') or self._last_transcription is None:
            logger.warning("No transcription available")
            return []
        
        # Check if there are any segments
        segments = self._last_transcription.get('segments', [])
        if not segments:
            logger.warning("Transcription has no segments")
            return []
        
        # Find matching segments
        matching_segments = []
        
        for segment in segments:
            # Verify segment has required fields
            if 'start' not in segment or 'end' not in segment:
                logger.warning(f"Segment missing start/end times: {segment}")
                continue
                
            start_time = segment.get('start', 0)
            end_time = segment.get('end', 0)
            
            # Check if timestamp falls within segment (with margin)
            if (start_time - margin) <= timestamp <= (end_time + margin):
                # Verify segment has text
                if not segment.get('text'):
                    logger.warning(f"Matching segment has no text: {segment}")
                    continue
                    
                matching_segments.append(segment)
        
        return matching_segments
    
    def save_transcription(self, transcription: Dict, output_path: str) -> None:
        """
        Save transcription to file.
        
        Args:
            transcription: Transcription result
            output_path: Output file path
        """
        # Store the transcription for later use
        self._last_transcription = transcription
        
        try:
            # Determine format from extension
            _, ext = os.path.splitext(output_path)
            ext = ext.lower()
            
            if ext == ".json":
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(transcription, f, indent=2, ensure_ascii=False)
            
            elif ext == ".txt":
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(transcription["text"])
            
            elif ext == ".srt":
                # Format as SRT if not already
                if "srt" not in transcription:
                    transcription = self._format_result(transcription, "srt")
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(transcription["srt"])
            
            elif ext == ".vtt":
                # Format as VTT if not already
                if "vtt" not in transcription:
                    transcription = self._format_result(transcription, "vtt")
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(transcription["vtt"])
            
            else:
                # Default to JSON
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(transcription, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Transcription saved to {output_path}")
        
        except Exception as e:
            logger.error(f"Error saving transcription: {str(e)}")


# For backward compatibility
WhisperTranscriber = VoskTranscriber
SpeechRecognitionTranscriber = VoskTranscriber


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Vosk Speech Transcriber")
    parser.add_argument("--audio", type=str, required=True, help="Path to audio file")
    parser.add_argument("--output", type=str, required=True, help="Output file path")
    parser.add_argument("--language", type=str, default="en-US", help="Language code")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Initialize transcriber
    transcriber = VoskTranscriber(
        language=args.language
    )
    
    # Transcribe audio
    result = transcriber.transcribe(
        audio_path=args.audio
    )
    
    # Save result
    transcriber.save_transcription(result, args.output)
