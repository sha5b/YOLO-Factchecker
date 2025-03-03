import os
import json
import numpy as np
import torch
import logging
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
import subprocess
import tempfile
import time

logger = logging.getLogger("speech_recognizer")

class WhisperTranscriber:
    """
    Speech recognition using OpenAI's Whisper model, running locally.
    Supports both standard Whisper and Faster-Whisper implementations.
    """
    
    def __init__(
        self,
        model_size: str = "base",
        model_path: Optional[str] = None,
        device: str = "auto",
        compute_type: str = "float16",
        use_faster_whisper: bool = True,
        language: Optional[str] = None,
        beam_size: int = 5,
        vad_filter: bool = True,
        diarize: bool = False,
        diarization_speakers: int = 2
    ):
        """
        Initialize Whisper transcriber.
        
        Args:
            model_size: Size of Whisper model ('tiny', 'base', 'small', 'medium', 'large-v3')
            model_path: Path to custom model or directory
            device: Device to run on ('cpu', 'cuda', 'auto')
            compute_type: Computation type ('float32', 'float16', 'int8')
            use_faster_whisper: Whether to use Faster Whisper implementation
            language: Language code (e.g., 'en', 'fr') or None for auto-detection
            beam_size: Beam size for decoding
            vad_filter: Whether to use voice activity detection
            diarize: Whether to perform speaker diarization
            diarization_speakers: Expected number of speakers for diarization
        """
        self.model_size = model_size
        self.device = self._get_device(device)
        self.compute_type = compute_type
        self.use_faster_whisper = use_faster_whisper
        self.language = language
        self.beam_size = beam_size
        self.vad_filter = vad_filter
        self.diarize = diarize
        self.diarization_speakers = diarization_speakers
        
        # Path operations
        if model_path is None:
            # Use default model location or download
            self.model_path = self._get_default_model_path(model_size)
        else:
            self.model_path = model_path
        
        # Load the model
        logger.info(f"Initializing Whisper model: {model_size} on {self.device}")
        self._load_model()
        
        # Initialize diarization model if needed
        if self.diarize:
            self._init_diarization()
    
    def _get_device(self, device: str) -> str:
        """Determine the device to run on."""
        if device.lower() == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device.lower()
    
    def _get_default_model_path(self, model_size: str) -> str:
        """Get default path for model or use the model name for download."""
        # Default path is ~/.cache/whisper
        cache_dir = os.path.expanduser("~/.cache/whisper")
        os.makedirs(cache_dir, exist_ok=True)
        
        # For faster-whisper models, the directory structure is different
        if self.use_faster_whisper:
            model_dir = os.path.join(cache_dir, f"faster-whisper-{model_size}")
            if not os.path.exists(model_dir):
                return model_size  # Return model name for download
            return model_dir
        else:
            # Standard Whisper
            return model_size  # OpenAI's Whisper will handle caching
    
    def _load_model(self) -> None:
        """Load the Whisper model."""
        try:
            if self.use_faster_whisper:
                self._load_faster_whisper()
            else:
                self._load_standard_whisper()
        except Exception as e:
            logger.error(f"Error loading Whisper model: {str(e)}")
            raise
    
    def _load_faster_whisper(self) -> None:
        """Load the Faster Whisper implementation."""
        try:
            # Import here to avoid dependencies if not used
            from faster_whisper import WhisperModel
            
            # Determine compute type
            if self.compute_type == "float16" and self.device == "cpu":
                logger.warning("float16 not supported on CPU, using float32 instead")
                compute_type = "float32"
            else:
                compute_type = self.compute_type
            
            # Load the model
            self.model = WhisperModel(
                self.model_path,
                device=self.device,
                compute_type=compute_type,
                cpu_threads=4,  # Adjust based on your system
                download_root=os.path.expanduser("~/.cache/whisper")
            )
            
            logger.info(f"Faster Whisper model loaded: {self.model_size}")
        
        except ImportError:
            logger.error("Failed to import faster_whisper. Install with: pip install faster-whisper")
            self.use_faster_whisper = False
            self._load_standard_whisper()
    
    def _load_standard_whisper(self) -> None:
        """Load the standard OpenAI Whisper implementation."""
        try:
            # Import here to avoid dependencies if not used
            import whisper
            
            # Load the model
            self.model = whisper.load_model(
                self.model_size,
                device=self.device,
                download_root=os.path.expanduser("~/.cache/whisper")
            )
            
            logger.info(f"Standard Whisper model loaded: {self.model_size}")
        
        except ImportError:
            logger.error("Failed to import whisper. Install with: pip install openai-whisper")
            raise
    
    def _init_diarization(self) -> None:
        """Initialize speaker diarization model."""
        try:
            # Import here to avoid dependencies if not used
            import pyannote.audio
            from pyannote.audio import Pipeline
            
            # Check for authentication
            token_path = os.path.expanduser("~/.cache/huggingface/token")
            if not os.path.exists(token_path):
                logger.warning(
                    "HuggingFace token not found. Speaker diarization might fail. "
                    "Please set up HuggingFace authentication for pyannote.audio access."
                )
            
            # Load diarization pipeline
            self.diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization@2.1",
                use_auth_token=True
            )
            
            # Move to appropriate device
            if self.device == "cuda":
                self.diarization_pipeline = self.diarization_pipeline.to(torch.device("cuda"))
            
            logger.info("Speaker diarization model loaded successfully")
        
        except ImportError:
            logger.error(
                "Failed to import pyannote.audio. Install with: "
                "pip install pyannote.audio"
            )
            self.diarize = False
        except Exception as e:
            logger.error(f"Error initializing diarization: {str(e)}")
            self.diarize = False
    
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
        
        # Prepare audio file
        audio_path = self._prepare_audio(audio_path)
        
        # Transcribe based on implementation
        if self.use_faster_whisper:
            result = self._transcribe_faster_whisper(audio_path, word_timestamps)
        else:
            result = self._transcribe_standard_whisper(audio_path, word_timestamps)
        
        # Perform diarization if requested
        if self.diarize:
            result = self._apply_diarization(audio_path, result)
        
        # Format the results
        formatted_result = self._format_result(result, output_format)
        
        elapsed_time = time.time() - start_time
        audio_duration = self._get_audio_duration(audio_path)
        
        logger.info(f"Transcription completed in {elapsed_time:.2f}s for {audio_duration:.2f}s audio")
        logger.info(f"Real-time factor: {elapsed_time / audio_duration:.2f}x")
        
        return formatted_result
    
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
        
        # If not already WAV/MP3/FLAC, convert to WAV
        if ext.lower() not in ['.wav', '.mp3', '.flac']:
            logger.info(f"Converting {ext} file to WAV")
            
            # Create temp file
            temp_dir = tempfile.gettempdir()
            output_path = os.path.join(temp_dir, f"whisper_audio_{int(time.time())}.wav")
            
            # Convert using ffmpeg
            try:
                subprocess.run([
                    "ffmpeg", "-i", audio_path, "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", 
                    output_path, "-y", "-loglevel", "error"
                ], check=True)
                
                return output_path
            
            except subprocess.CalledProcessError as e:
                logger.error(f"Error converting audio: {str(e)}")
                logger.warning("Using original file instead")
        
        return audio_path
    
    def _transcribe_faster_whisper(
        self,
        audio_path: str,
        word_timestamps: bool = False
    ) -> Dict:
        """
        Transcribe using Faster Whisper implementation.
        
        Args:
            audio_path: Path to audio file
            word_timestamps: Whether to include word-level timestamps
            
        Returns:
            Transcription result
        """
        # Transcribe with Faster Whisper
        segments, info = self.model.transcribe(
            audio_path,
            beam_size=self.beam_size,
            language=self.language,
            vad_filter=self.vad_filter,
            word_timestamps=word_timestamps
        )
        
        # Convert to standard format
        result = {
            "text": "",
            "segments": [],
            "language": info.language,
            "language_probability": info.language_probability
        }
        
        # Process segments
        for i, segment in enumerate(segments):
            seg_dict = {
                "id": i,
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip(),
                "tokens": segment.tokens,
                "temperature": segment.temperature,
                "avg_logprob": segment.avg_logprob,
                "compression_ratio": segment.compression_ratio,
                "no_speech_prob": segment.no_speech_prob
            }
            
            # Add word timestamps if available
            if word_timestamps and segment.words:
                seg_dict["words"] = [
                    {"word": word.word, "start": word.start, "end": word.end, "probability": word.probability}
                    for word in segment.words
                ]
            
            result["segments"].append(seg_dict)
            result["text"] += segment.text.strip() + " "
        
        result["text"] = result["text"].strip()
        return result
    
    def _transcribe_standard_whisper(
        self,
        audio_path: str,
        word_timestamps: bool = False
    ) -> Dict:
        """
        Transcribe using standard Whisper implementation.
        
        Args:
            audio_path: Path to audio file
            word_timestamps: Whether to include word-level timestamps
            
        Returns:
            Transcription result
        """
        # Set transcription options
        transcribe_options = {
            "language": self.language,
            "beam_size": self.beam_size,
            "word_timestamps": word_timestamps,
            "vad_filter": self.vad_filter
        }
        
        # Remove None values
        transcribe_options = {k: v for k, v in transcribe_options.items() if v is not None}
        
        # Transcribe with standard Whisper
        result = self.model.transcribe(audio_path, **transcribe_options)
        
        # Ensure consistent format with faster-whisper
        for i, segment in enumerate(result["segments"]):
            segment["id"] = i
        
        return result
    
    def _apply_diarization(self, audio_path: str, transcription: Dict) -> Dict:
        """
        Apply speaker diarization to transcription.
        
        Args:
            audio_path: Path to audio file
            transcription: Transcription result
            
        Returns:
            Transcription with speaker information
        """
        if not hasattr(self, 'diarization_pipeline'):
            logger.warning("Diarization pipeline not initialized, skipping diarization")
            return transcription
        
        logger.info("Applying speaker diarization")
        
        try:
            # Run diarization
            diarization = self.diarization_pipeline(
                audio_path,
                num_speakers=self.diarization_speakers
            )
            
            # Add speaker information to segments
            for segment in transcription["segments"]:
                segment_start = segment["start"]
                segment_end = segment["end"]
                
                # Find overlapping speaker turns
                speakers = {}
                
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    # Check overlap
                    if max(segment_start, turn.start) < min(segment_end, turn.end):
                        overlap_duration = min(segment_end, turn.end) - max(segment_start, turn.start)
                        speakers[speaker] = speakers.get(speaker, 0) + overlap_duration
                
                # Assign most active speaker
                if speakers:
                    segment["speaker"] = max(speakers.items(), key=lambda x: x[1])[0]
                else:
                    segment["speaker"] = "UNKNOWN"
            
            logger.info(f"Diarization completed: {len(set([s.get('speaker', 'UNKNOWN') for s in transcription['segments']]))} speakers identified")
            
        except Exception as e:
            logger.error(f"Error during diarization: {str(e)}")
        
        return transcription
    
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
    
    def save_transcription(self, transcription: Dict, output_path: str) -> None:
        """
        Save transcription to file.
        
        Args:
            transcription: Transcription result
            output_path: Output file path
        """
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
    
    parser = argparse.ArgumentParser(description="Whisper Transcriber")
    parser.add_argument("--audio", type=str, required=True, help="Path to audio file")
    parser.add_argument("--output", type=str, required=True, help="Output file path")
    parser.add_argument("--model", type=str, default="base", help="Whisper model size")
    parser.add_argument("--faster", action="store_true", help="Use Faster Whisper")
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu, cuda, auto)")
    parser.add_argument("--language", type=str, default=None, help="Language code")
    parser.add_argument("--diarize", action="store_true", help="Perform speaker diarization")
    parser.add_argument("--speakers", type=int, default=2, help="Number of speakers for diarization")
    parser.add_argument("--words", action="store_true", help="Include word timestamps")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Initialize transcriber
    transcriber = WhisperTranscriber(
        model_size=args.model,
        device=args.device,
        use_faster_whisper=args.faster,
        language=args.language,
        diarize=args.diarize,
        diarization_speakers=args.speakers
    )
    
    # Transcribe audio
    result = transcriber.transcribe(
        audio_path=args.audio,
        word_timestamps=args.words
    )
    
    # Save result
    transcriber.save_transcription(result, args.output)
