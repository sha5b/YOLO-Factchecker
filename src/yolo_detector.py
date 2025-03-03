import os
import json
import numpy as np
import cv2
from pathlib import Path
import logging
from typing import List, Dict, Tuple, Union, Optional

logger = logging.getLogger("yolo_detector")

class YOLODetector:
    """
    YOLO-based object detection for video analysis.
    This implementation uses Ultralytics YOLOv8 running locally.
    """
    
    def __init__(
        self,
        model_path: str = "yolov8s.pt",
        confidence_threshold: float = 0.5,
        device: str = "auto",
        img_size: int = 640,
        custom_objects: bool = False
    ):
        """
        Initialize YOLO detector.
        
        Args:
            model_path: Path to YOLO model weights or model name
            confidence_threshold: Minimum confidence for detections
            device: Device to run on ('cpu', 'cuda', 'auto')
            img_size: Image size for detection
            custom_objects: Whether to use a custom-trained model with specific classes
        """
        self.confidence_threshold = confidence_threshold
        self.img_size = img_size
        self.device = self._get_device(device)
        self.custom_objects = custom_objects
        
        logger.info(f"Initializing YOLO detector on {self.device}")
        self._load_model(model_path)
    
    def _get_device(self, device: str) -> str:
        """Determine the device to run on."""
        if device.lower() == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return device
    
    def _load_model(self, model_path: str) -> None:
        """Load YOLO model."""
        try:
            # Import here to avoid dependencies if YOLODetector is not used
            from ultralytics import YOLO
            
            # Check if model_path is a path or a model name
            if os.path.exists(model_path):
                logger.info(f"Loading YOLO model from: {model_path}")
                self.model = YOLO(model_path)
            else:
                logger.info(f"Loading YOLO model: {model_path}")
                self.model = YOLO(model_path)
            
            # Move model to device
            self.model.to(self.device)
            
            logger.info(f"YOLO model loaded successfully: {self.model.names}")
            self.class_names = self.model.names
            
        except Exception as e:
            logger.error(f"Error loading YOLO model: {str(e)}")
            raise
    
    def detect(
        self,
        image: np.ndarray,
        augment: bool = False
    ) -> List[Dict]:
        """
        Detect objects in a single image.
        
        Args:
            image: Input image as numpy array (BGR format)
            augment: Whether to use augmentation
            
        Returns:
            List of detection dictionaries
        """
        try:
            # Ensure image is in correct format (RGB for YOLO)
            if image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image  # Assume already RGB
            
            # Run inference
            results = self.model(
                image_rgb,
                imgsz=self.img_size,  # Changed from 'size' to 'imgsz'
                augment=augment,
                verbose=False,
                conf=self.confidence_threshold
            )
            
            # Process results
            detections = self._process_results(results[0], image.shape)
            return detections
            
        except Exception as e:
            logger.error(f"Error during detection: {str(e)}")
            return []
    
    def detect_batch(
        self,
        images: List[np.ndarray],
        batch_size: int = 8,
        augment: bool = False
    ) -> List[List[Dict]]:
        """
        Detect objects in batch of images.
        
        Args:
            images: List of input images
            batch_size: Batch size for inference
            augment: Whether to use augmentation
            
        Returns:
            List of detection lists (one per image)
        """
        if not images:
            return []
        
        all_detections = []
        
        # Process in batches
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            
            # Convert BGR to RGB for each image
            batch_rgb = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.shape[2] == 3 else img 
                          for img in batch]
            
            try:
                # Run inference on batch
                results = self.model(
                    batch_rgb,
                    imgsz=self.img_size,  # Changed from 'size' to 'imgsz'
                    augment=augment,
                    verbose=False,
                    conf=self.confidence_threshold
                )
                
                # Process each result
                batch_detections = []
                for j, result in enumerate(results):
                    img_shape = batch[j].shape
                    detections = self._process_results(result, img_shape)
                    batch_detections.append(detections)
                
                all_detections.extend(batch_detections)
                
            except Exception as e:
                logger.error(f"Error during batch detection: {str(e)}")
                # Return empty detections for failed batch
                all_detections.extend([[] for _ in range(len(batch))])
        
        return all_detections
    
    def _process_results(self, result, img_shape: Tuple[int, int, int]) -> List[Dict]:
        """
        Process YOLO results into standardized format.
        
        Args:
            result: YOLO result object
            img_shape: Original image shape
            
        Returns:
            List of detection dictionaries
        """
        height, width = img_shape[:2]
        detections = []
        
        # Convert detections to standard format
        if hasattr(result, 'boxes') and len(result.boxes) > 0:
            boxes = result.boxes
            
            for i in range(len(boxes)):
                # Get box coordinates
                box = boxes[i].xyxy[0].cpu().numpy()  # x1, y1, x2, y2
                
                # Get confidence and class
                conf = float(boxes[i].conf[0].cpu().numpy())
                cls_id = int(boxes[i].cls[0].cpu().numpy())
                cls_name = self.class_names[cls_id]
                
                # Skip if below threshold
                if conf < self.confidence_threshold:
                    continue
                
                # Convert to standard format
                x1, y1, x2, y2 = map(float, box)
                
                # Calculate relative coordinates for portability
                rel_bbox = [x1/width, y1/height, x2/width, y2/height]
                
                detection = {
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "rel_bbox": rel_bbox,
                    "confidence": float(conf),
                    "class_id": cls_id,
                    "class_name": cls_name
                }
                
                detections.append(detection)
        
        return detections
    
    def _draw_detections(self, frame: np.ndarray, detections: List[Dict], enhanced: bool = False) -> np.ndarray:
        """
        Draw detection bounding boxes on frame.
        
        Args:
            frame: Input frame
            detections: List of detections
            enhanced: Whether to use enhanced visibility for bounding boxes
            
        Returns:
            Frame with drawn detections
        """
        drawn_frame = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = map(int, det["bbox"])
            class_name = det["class_name"]
            confidence = det["confidence"]
            
            # Generate color based on class id
            color = self._get_color(det["class_id"])
            
            # Determine line thickness and font scale based on enhanced mode
            line_thickness = 4 if enhanced else 2
            font_scale = 0.7 if enhanced else 0.5
            
            # Draw bounding box with enhanced visibility if requested
            if enhanced:
                # Draw outer box with black color for better visibility
                cv2.rectangle(drawn_frame, (x1-1, y1-1), (x2+1, y2+1), (0, 0, 0), line_thickness+2)
            
            # Draw main bounding box
            cv2.rectangle(drawn_frame, (x1, y1), (x2, y2), color, line_thickness)
            
            # Draw label with enhanced visibility
            label = f"{class_name}: {confidence:.2f}"
            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
            
            # Draw label background
            cv2.rectangle(drawn_frame, (x1, y1), (x1 + text_size[0], y1 - text_size[1] - 5), color, -1)
            
            # Draw label text
            cv2.putText(drawn_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                        font_scale, (255, 255, 255), 2)
            
            # Add highlight for person detections (potential faces with expressions)
            if class_name == 'person' and enhanced:
                # Calculate more precise face area based on typical human proportions
                person_width = x2 - x1
                person_height = y2 - y1
                
                # Face is typically in the upper portion of the person bounding box
                # For a standing/sitting person, face is about 1/7 to 1/8 of total height
                # and positioned at the top ~1/8 of the body
                face_width = int(person_width * 0.5)  # Face is about 50% of body width
                face_height = int(person_height * 0.15)  # Face is about 15% of body height
                
                # Center the face horizontally
                face_x1 = x1 + int((person_width - face_width) / 2)
                face_x2 = face_x1 + face_width
                
                # Position face at appropriate height (not at the very top)
                # For most people in videos, face starts about 5-10% down from the top of the bounding box
                face_y1 = y1 + int(person_height * 0.05)  # Start 5% down from top
                face_y2 = face_y1 + face_height
                
                # Draw face highlight with a more visible style
                cv2.rectangle(drawn_frame, (face_x1, face_y1), (face_x2, face_y2), 
                             (0, 255, 255), 2)  # Yellow highlight for face area
                
                # Add "Face" label
                face_label = "Face"
                cv2.putText(drawn_frame, face_label, (face_x1, face_y1 - 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), 2)
        
        return drawn_frame
    
    def _get_color(self, class_id: int) -> Tuple[int, int, int]:
        """Generate unique color for class."""
        # Generate deterministic color based on class id
        np.random.seed(class_id)
        color = tuple(map(int, np.random.randint(0, 255, 3)))
        return color
    
    def save_detections(self, detections: List[Dict], output_path: str) -> None:
        """
        Save detections to JSON file.
        
        Args:
            detections: List of detection dictionaries
            output_path: Path to save JSON file
        """
        try:
            with open(output_path, 'w') as f:
                json.dump(detections, f, indent=2)
            logger.info(f"Detections saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving detections: {str(e)}")
    
    def load_detections(self, input_path: str) -> List[Dict]:
        """
        Load detections from JSON file.
        
        Args:
            input_path: Path to JSON file
            
        Returns:
            List of detection dictionaries
        """
        try:
            with open(input_path, 'r') as f:
                detections = json.load(f)
            logger.info(f"Loaded {len(detections)} detections from {input_path}")
            return detections
        except Exception as e:
            logger.error(f"Error loading detections: {str(e)}")
            return []
