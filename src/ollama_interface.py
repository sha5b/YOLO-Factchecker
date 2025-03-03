import requests
import json
import logging
from typing import Dict, List, Optional, Union, Any

logger = logging.getLogger("ollama_interface")

class OllamaInterface:
    """
    Interface to interact with local Ollama LLM.
    Provides methods for generating text, analyzing content, and fact-checking.
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3",
        temperature: float = 0.2,
        max_tokens: int = 1024
    ):
        """
        Initialize Ollama interface.
        
        Args:
            base_url: Base URL for Ollama API
            model: Model name to use
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
        """
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Check if Ollama is available
        self._check_availability()
        
        logger.info(f"Initialized Ollama interface with model: {model}")
    
    def _check_availability(self) -> bool:
        """
        Check if Ollama is available.
        
        Returns:
            True if available, False otherwise
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                logger.info(f"Ollama available with {len(models)} models")
                
                # Check if our model is available
                model_names = [model.get("name") for model in models]
                if self.model not in model_names:
                    logger.warning(f"Model {self.model} not found in available models: {model_names}")
                
                return True
            else:
                logger.warning(f"Ollama returned status code {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Error checking Ollama availability: {str(e)}")
            return False
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text using Ollama.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional parameters to pass to Ollama
            
        Returns:
            Generated text
        """
        try:
            # Prepare request
            url = f"{self.base_url}/api/generate"
            data = {
                "model": self.model,
                "prompt": prompt,
                "temperature": kwargs.get("temperature", self.temperature),
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                "stream": False
            }
            
            # Add any additional parameters
            for key, value in kwargs.items():
                if key not in data:
                    data[key] = value
            
            # Make request
            response = requests.post(url, json=data)
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "")
            else:
                logger.error(f"Ollama returned status code {response.status_code}: {response.text}")
                return f"Error: Ollama returned status code {response.status_code}"
        
        except Exception as e:
            logger.error(f"Error generating text with Ollama: {str(e)}")
            return f"Error: {str(e)}"
    
    def generate_stream(self, prompt: str, callback=None, **kwargs) -> str:
        """
        Generate text using Ollama with streaming.
        
        Args:
            prompt: Input prompt
            callback: Callback function to call with each chunk
            **kwargs: Additional parameters to pass to Ollama
            
        Returns:
            Complete generated text
        """
        try:
            # Prepare request
            url = f"{self.base_url}/api/generate"
            data = {
                "model": self.model,
                "prompt": prompt,
                "temperature": kwargs.get("temperature", self.temperature),
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                "stream": True
            }
            
            # Add any additional parameters
            for key, value in kwargs.items():
                if key not in data:
                    data[key] = value
            
            # Make request
            response = requests.post(url, json=data, stream=True)
            
            if response.status_code == 200:
                full_response = ""
                
                for line in response.iter_lines():
                    if line:
                        chunk = json.loads(line)
                        text_chunk = chunk.get("response", "")
                        full_response += text_chunk
                        
                        if callback:
                            callback(text_chunk)
                
                return full_response
            else:
                logger.error(f"Ollama returned status code {response.status_code}: {response.text}")
                return f"Error: Ollama returned status code {response.status_code}"
        
        except Exception as e:
            logger.error(f"Error generating text with Ollama: {str(e)}")
            return f"Error: {str(e)}"
    
    def analyze_detections(self, detections: List[Dict], transcription: str) -> str:
        """
        Analyze YOLO detections and transcription for inconsistencies.
        
        Args:
            detections: List of YOLO detections
            transcription: Transcribed text
            
        Returns:
            Analysis text
        """
        # Count objects by class
        object_counts = {}
        for det in detections:
            class_name = det.get("class_name", "unknown")
            if class_name not in object_counts:
                object_counts[class_name] = 0
            object_counts[class_name] += 1
        
        # Create a summary of detected objects
        objects_summary = ", ".join([f"{count} {obj}" for obj, count in object_counts.items()])
        
        # Create prompt
        prompt = f"""
        I need to analyze a video for factual accuracy. Here's what I know:
        
        Detected objects: {objects_summary}
        
        Transcription: "{transcription}"
        
        Please analyze if there are any inconsistencies between what was said and what was detected in the video.
        Focus on factual claims about objects, people, or scenes that might be contradicted by the visual evidence.
        
        Provide your analysis in a clear, concise format.
        """
        
        return self.generate(prompt)
    
    def fact_check(self, claim: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Fact check a claim using Ollama.
        
        Args:
            claim: Claim to check
            context: Optional context information
            
        Returns:
            Fact check result
        """
        # Create prompt
        if context:
            prompt = f"""
            I need to fact-check the following claim:
            
            Claim: "{claim}"
            
            Context information: {context}
            
            Please analyze the claim for factual accuracy based on the provided context.
            Return your analysis in the following format:
            
            Verified: [true/false/uncertain]
            Confidence: [0-1 scale]
            Explanation: [your explanation]
            """
        else:
            prompt = f"""
            I need to fact-check the following claim:
            
            Claim: "{claim}"
            
            Please analyze the claim for factual accuracy based on your knowledge.
            Return your analysis in the following format:
            
            Verified: [true/false/uncertain]
            Confidence: [0-1 scale]
            Explanation: [your explanation]
            """
        
        response = self.generate(prompt)
        
        # Parse response
        result = {
            "claim": claim,
            "response": response,
            "verified": None,
            "confidence": 0.0,
            "explanation": ""
        }
        
        # Try to extract structured information
        try:
            lines = response.strip().split("\n")
            for line in lines:
                line = line.strip()
                if line.startswith("Verified:"):
                    value = line.replace("Verified:", "").strip().lower()
                    if value in ["true", "yes"]:
                        result["verified"] = True
                    elif value in ["false", "no"]:
                        result["verified"] = False
                    else:
                        result["verified"] = None
                
                elif line.startswith("Confidence:"):
                    try:
                        value = line.replace("Confidence:", "").strip()
                        result["confidence"] = float(value)
                    except:
                        pass
                
                elif line.startswith("Explanation:"):
                    result["explanation"] = line.replace("Explanation:", "").strip()
        except Exception as e:
            logger.error(f"Error parsing fact check response: {str(e)}")
        
        return result
