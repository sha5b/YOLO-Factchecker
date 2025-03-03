import os
import json
import numpy as np
import torch
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
        model_path: str = "yolov8n.pt",
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
            return "cuda" if torch.cuda.is_available() else "cpu"
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
                size=self.img_size,
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
                    size=self.img_size,
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
    
    def detect_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        sample_rate: int = 1,  # Process every Nth frame
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        draw_on_frames: bool = False
    ) -> List[Dict]:
        """
        Run detection on video file.
        
        Args:
            video_path: Path to video file
            output_path: Path to save detection results (JSON)
            sample_rate: Process every Nth frame
            start_frame: Starting frame
            end_frame: Ending frame (None for all frames)
            draw_on_frames: Whether to draw bounding boxes on frames
            
        Returns:
            List of detections with frame information
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error(f"Error opening video file: {video_path}")
            return []
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if end_frame is None:
            end_frame = total_frames
        
        # Seek to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Initialize output video if needed
        output_video = None
        if draw_on_frames and output_path:
            video_output = output_path.replace('.json', '.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_video = cv2.VideoWriter(video_output, fourcc, fps, (width, height))
        
        all_detections = []
        frame_count = start_frame
        processed_count = 0
        
        try:
            while cap.isOpened() and frame_count < end_frame:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Process every Nth frame
                if (frame_count - start_frame) % sample_rate == 0:
                    # Run detection
                    detections = self.detect(frame)
                    
                    # Add frame information
                    for det in detections:
                        det["frame"] = frame_count
                        det["timestamp"] = frame_count / fps
                    
                    all_detections.extend(detections)
                    processed_count += 1
                    
                    # Draw on frame if requested
                    if draw_on_frames:
                        drawn_frame = self._draw_detections(frame, detections)
                        
                        # Add frame number
                        cv2.putText(
                            drawn_frame, f"Frame: {frame_count}", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (0, 255, 0), 2
                        )
                        
                        if output_video:
                            output_video.write(drawn_frame)
                
                frame_count += 1
                
                # Log progress periodically
                if frame_count % 100 == 0:
                    logger.info(f"Processed {frame_count}/{end_frame} frames")
        
        finally:
            cap.release()
            if output_video:
                output_video.release()
        
        logger.info(f"Video processing complete. Processed {processed_count} frames with {len(all_detections)} detections")
        
        # Save detections if output path provided
        if output_path:
            self.save_detections(all_detections, output_path)
        
        return all_detections
    
    def _draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Draw detection bounding boxes on frame.
        
        Args:
            frame: Input frame
            detections: List of detections
            
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
            
            # Draw bounding box
            cv2.rectangle(drawn_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(drawn_frame, (x1, y1), (x1 + text_size[0], y1 - text_size[1] - 5), color, -1)
            cv2.putText(drawn_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
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


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="YOLO Detector")
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    parser.add_argument("--output", type=str, required=True, help="Output path for detections")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLO model path")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu, cuda, auto)")
    parser.add_argument("--draw", action="store_true", help="Draw detections on video")
    parser.add_argument("--sample", type=int, default=1, help="Process every Nth frame")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Initialize detector
    detector = YOLODetector(
        model_path=args.model,
        confidence_threshold=args.conf,
        device=args.device
    )
    
    # Process video
    detector.detect_video(
        video_path=args.video,
        output_path=args.output,
        sample_rate=args.sample,
        draw_on_frames=args.draw
    )
