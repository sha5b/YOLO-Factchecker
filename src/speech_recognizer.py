import os
import json
import logging
import subprocess
import tempfile
import time
import wave
import requests
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple

from vosk import Model, KaldiRecognizer, SetLogLevel

logger = logging.getLogger("speech_recognizer")

# Set Vosk logging level (0 for most verbose, higher numbers for less verbose)
SetLogLevel(0)

class VoskTranscriber:
    """
    Speech recognition using Vosk, running locally.
    """
    
    def __init__(
        self,
        model_name: str = "vosk-model-en-us-0.22",
        model_path: Optional[str] = None,
        sample_rate: int = 16000,
        enable_words: bool = True,
        language: str = "en-us",
        alternative_results: int = 1
    ):
        """
        Initialize Vosk transcriber.
        
        Args:
            model_name: Name of the Vosk model to use
            model_path: Path to custom model or directory (if None, will download)
            sample_rate: Audio sample rate in Hz
            enable_words: Whether to include word-level timestamps
            language: Language code (e.g., 'en-us')
            alternative_results: Number of alternative results to return
        """
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.enable_words = enable_words
        self.language = language
        self.alternative_results = alternative_results
        
        # Path operations
        if model_path is None:
            # Use default model location or download
            self.model_path = self._get_model_path(model_name)
        else:
            self.model_path = model_path
        
        # Load the model
        logger.info(f"Initializing Vosk model: {model_name}")
        self._load_model()
    
    def _get_model_path(self, model_name: str) -> str:
        """Get path to model, downloading if necessary."""
        # Default path is models/vosk/
        model_dir = Path("models/vosk")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = model_dir / model_name
        
        # Check if model exists, if not download it
        if not model_path.exists():
            self._download_model(model_name, model_path)
        
        return str(model_path)
    
    def _download_model(self, model_name: str, model_path: Path) -> None:
        """Download Vosk model."""
        logger.info(f"Downloading Vosk model: {model_name}")
        
        # Model URL
        base_url = "https://alphacephei.com/vosk/models/"
        model_url = f"{base_url}{model_name}.zip"
        
        try:
            # Download the model
            zip_path = Path(f"models/vosk/{model_name}.zip")
            
            logger.info(f"Downloading from {model_url}")
            response = requests.get(model_url, stream=True)
            response.raise_for_status()
            
            # Save the zip file
            with open(zip_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Extract the zip file
            import zipfile
            import shutil
            
            # Create a temporary extraction directory
            extract_dir = Path(f"models/vosk/temp_{model_name}")
            extract_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Extracting model to {extract_dir}")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_dir)
            
            # Find the model directory inside the extracted content
            # Vosk models typically have a single directory inside the zip
            extracted_dirs = [d for d in extract_dir.iterdir() if d.is_dir()]
            if extracted_dirs:
                # Move the contents to the target model path
                if model_path.exists():
                    shutil.rmtree(model_path)
                shutil.move(str(extracted_dirs[0]), str(model_path))
            else:
                # If no subdirectory, move all files directly
                model_path.mkdir(parents=True, exist_ok=True)
                for item in extract_dir.iterdir():
                    if item.is_file():
                        shutil.move(str(item), str(model_path / item.name))
            
            # Clean up
            if extract_dir.exists():
                shutil.rmtree(extract_dir)
            
            # Remove the zip file
            zip_path.unlink()
            
            logger.info(f"Model downloaded and extracted to {model_path}")
        
        except Exception as e:
            logger.error(f"Error downloading model: {str(e)}")
            raise
    
    def _load_model(self) -> None:
        """Load Vosk model."""
        try:
            self.model = Model(self.model_path)
            logger.info(f"Vosk model loaded successfully from {self.model_path}")
        except Exception as e:
            logger.error(f"Error loading Vosk model: {str(e)}")
            raise
    
    def transcribe(
        self,
        audio_path: str,
        output_format: str = "json",
        segment_level: bool = True,
        word_timestamps: bool = False
    ) -> Dict:
        """
        Transcribe audio file.
        
        Args:
            audio_path: Path to audio file
            output_format: Output format ('json', 'text', 'srt', 'vtt')
            segment_level: Whether to return segment-level timestamps
            word_timestamps: Whether to include word-level timestamps
            
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
            result = self._transcribe_audio(audio_path, word_timestamps or self.enable_words)
            
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
        
        # If not already WAV, convert to WAV
        if ext.lower() != '.wav':
            logger.info(f"Converting {ext} file to WAV")
            
            # Create temp file
            temp_dir = tempfile.gettempdir()
            output_path = os.path.join(temp_dir, f"vosk_audio_{int(time.time())}.wav")
            
            # Convert using ffmpeg
            try:
                subprocess.run([
                    "ffmpeg", "-i", audio_path, "-ar", str(self.sample_rate), "-ac", "1", "-c:a", "pcm_s16le", 
                    output_path, "-y", "-loglevel", "error"
                ], check=True)
                
                return output_path
            
            except subprocess.CalledProcessError as e:
                logger.error(f"Error converting audio: {str(e)}")
                logger.warning("Using original file instead")
        
        return audio_path
    
    def _transcribe_audio(self, audio_path: str, word_timestamps: bool = False) -> Dict:
        """
        Transcribe audio file using Vosk.
        
        Args:
            audio_path: Path to audio file
            word_timestamps: Whether to include word-level timestamps
            
        Returns:
            Transcription result
        """
        try:
            # Open audio file
            wf = wave.open(audio_path, "rb")
            
            # Check if audio format is compatible
            if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
                logger.warning("Audio file must be WAV format mono PCM.")
                return {"text": "", "segments": []}
            
            # Create recognizer
            rec = KaldiRecognizer(self.model, wf.getframerate())
            rec.SetWords(word_timestamps)
            rec.SetMaxAlternatives(self.alternative_results)
            
            # Process audio in chunks
            results = []
            chunk_size = 4000  # Process 4000 frames at a time
            
            while True:
                data = wf.readframes(chunk_size)
                if len(data) == 0:
                    break
                
                if rec.AcceptWaveform(data):
                    result_json = rec.Result()
                    result = json.loads(result_json)
                    if "result" in result:
                        results.append(result)
            
            # Get final result
            final_json = rec.FinalResult()
            final_result = json.loads(final_json)
            if "result" in final_result:
                results.append(final_result)
            
            # Process results into a format similar to Whisper
            return self._process_results(results, wf.getframerate())
            
        except Exception as e:
            logger.error(f"Error during transcription: {str(e)}")
            return {"text": "", "segments": []}
    
    def _process_results(self, results: List[Dict], frame_rate: int) -> Dict:
        """
        Process Vosk results into a standardized format.
        
        Args:
            results: List of Vosk result dictionaries
            frame_rate: Audio frame rate
            
        Returns:
            Processed results in a format similar to Whisper
        """
        full_text = ""
        segments = []
        
        for i, res in enumerate(results):
            if "result" not in res:
                continue
            
            # Extract segment text
            segment_text = res.get("text", "")
            full_text += segment_text + " "
            
            # Get start and end times
            words = res.get("result", [])
            if not words:
                continue
            
            start_time = words[0].get("start", 0)
            end_time = words[-1].get("end", 0)
            
            # Create segment
            segment = {
                "id": i,
                "start": start_time,
                "end": end_time,
                "text": segment_text.strip(),
                "avg_logprob": 0.0,  # Vosk doesn't provide this
                "compression_ratio": 1.0,  # Vosk doesn't provide this
                "no_speech_prob": 0.0  # Vosk doesn't provide this
            }
            
            # Add word timestamps if available
            if words and len(words) > 0:
                segment["words"] = [
                    {"word": w.get("word", ""), "start": w.get("start", 0), "end": w.get("end", 0), "probability": 1.0}
                    for w in words
                ]
            
            segments.append(segment)
        
        return {
            "text": full_text.strip(),
            "segments": segments,
            "language": self.language
        }
    
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
        # Store the last transcription result
        if not hasattr(self, '_last_transcription') or self._last_transcription is None:
            logger.warning("No transcription available")
            return []
        
        matching_segments = []
        
        for segment in self._last_transcription.get('segments', []):
            start_time = segment.get('start', 0)
            end_time = segment.get('end', 0)
            
            # Check if timestamp falls within segment (with margin)
            if (start_time - margin) <= timestamp <= (end_time + margin):
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


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Vosk Transcriber")
    parser.add_argument("--audio", type=str, required=True, help="Path to audio file")
    parser.add_argument("--output", type=str, required=True, help="Output file path")
    parser.add_argument("--model", type=str, default="vosk-model-en-us-0.22", help="Vosk model name")
    parser.add_argument("--language", type=str, default="en-us", help="Language code")
    parser.add_argument("--words", action="store_true", help="Include word timestamps")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Initialize transcriber
    transcriber = VoskTranscriber(
        model_name=args.model,
        language=args.language,
        enable_words=args.words
    )
    
    # Transcribe audio
    result = transcriber.transcribe(
        audio_path=args.audio,
        word_timestamps=args.words
    )
    
    # Save result
    transcriber.save_transcription(result, args.output)
