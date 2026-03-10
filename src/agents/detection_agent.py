"""
Detection Agent for the Agentic Object Detection System.
Handles object detection with intelligent model selection.
Integrates with existing models, data loaders, and tools.
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import os

from .base_agent import BaseAgent
from .policies import Policy, RandomPolicy, EntropyPolicy, ConfidencePolicy, AdaptivePolicy

# Import existing models
from ..models.lightweight_detector import LightweightDetector
from ..models.yolo_model import YOLODetector
from ..models.faster_rcnn_model import FasterRCNNDetector
from ..models.base_model import BaseDetector

# Import data utilities
from ..data.data_loader import load_image as load_image_raw

# Import tools
from ..tools.vision_tools import (
    ImageProcessor, 
    DetectionVisualizer, 
    MetricsCalculator,
    FeatureExtractor
)


class DetectionAgent(BaseDetector, BaseAgent):
    """
    Agent that handles object detection with intelligent model selection.
    Supports multiple detection strategies and can switch between models.
    Inherits from BaseDetector for compatibility with existing inference pipeline.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the detection agent."""
        self.config = config
        self.models: Dict[str, BaseDetector] = {}
        self.current_model: str = "lightweight"
        self.detection_history: List[Dict] = []
        self.policy: Optional[Policy] = None
        
        # Setup agent components first (independent of bases)
        self._setup_agent()
        
        # Set required attributes before initializing bases
        self.name: str = "DetectionAgent"
        self.total_steps: int = 0
        
        # Initialize models
        self._init_models()
        
        # Initialize policy
        self._init_policy()
        
        # Logger (must be after _setup_agent for logging config)
        self.logger = self._get_logger()
    
    def _setup_agent(self):
        """Setup detection-specific components."""
        # Initialize vision tools
        self.image_processor = ImageProcessor()
        self.visualizer = DetectionVisualizer()
        self.metrics_calc = MetricsCalculator()
        self.feature_extractor = FeatureExtractor()
        
        # Data paths
        paths = self.config.get("paths", {})
        self.data_dir = paths.get("data_dir", "data")
        self.results_dir = paths.get("results_dir", "results")
        
        # Create results directory
        os.makedirs(self.results_dir, exist_ok=True)
    
    def _init_models(self):
        """Initialize available detection models."""
        model_cfg = self.config.get("model", {})
        
        # Initialize models based on config
        available_models = model_cfg.get("available", ["lightweight", "yolo", "faster_rcnn"])
        
        for model_name in available_models:
            try:
                if model_name == "yolo":
                    weights = model_cfg.get("yolo_weights", "yolov8n.pt")
                    self.models[model_name] = YOLODetector(
                        weights,
                        model_cfg.get("conf_threshold", 0.35),
                        model_cfg.get("iou_threshold", 0.5)
                    )
                    print(f"Loaded model: {model_name}")
                elif model_name == "faster_rcnn":
                    self.models[model_name] = FasterRCNNDetector(
                        model_cfg.get("conf_threshold", 0.35)
                    )
                    print(f"Loaded model: {model_name}")
                elif model_name == "lightweight":
                    self.models[model_name] = LightweightDetector(
                        model_cfg.get("conf_threshold", 0.35)
                    )
                    print(f"Loaded model: {model_name}")
            except Exception as e:
                print(f"Warning: Failed to load {model_name}: {e}")
        
        # Fallback to lightweight if no models loaded
        if not self.models:
            self.models["lightweight"] = LightweightDetector()
            print("Using fallback lightweight detector")
        
        # Set primary model
        primary = model_cfg.get("primary", "lightweight")
        self.current_model = primary if primary in self.models else "lightweight"
    
    def _init_policy(self):
        """Initialize the policy for model selection."""
        agent_cfg = self.config.get("agent", {})
        strategy_cfg = agent_cfg.get("strategy", {})
        policy_type = agent_cfg.get("policy", "entropy")
        
        if policy_type == "random":
            self.policy = RandomPolicy(list(self.models.keys()))
        elif policy_type == "confidence":
            self.policy = ConfidencePolicy(
                confidence_threshold=agent_cfg.get("confidence_threshold", 0.7)
            )
        elif policy_type == "adaptive":
            self.policy = AdaptivePolicy(
                entropy_threshold=agent_cfg.get("entropy_threshold", 1.2),
                confidence_threshold=agent_cfg.get("confidence_threshold", 0.5)
            )
        else:  # entropy (default)
            low_entropy_model = (
                agent_cfg.get("low_entropy_strategy")
                or strategy_cfg.get("low_entropy")
                or "lightweight"
            )
            high_entropy_model = (
                agent_cfg.get("high_entropy_strategy")
                or strategy_cfg.get("high_entropy")
                or "yolo"
            )
            self.policy = EntropyPolicy(
                entropy_threshold=agent_cfg.get("entropy_threshold", 1.2),
                low_entropy_model=low_entropy_model,
                high_entropy_model=high_entropy_model
            )
        
        print(f"Initialized policy: {policy_type}")
    
    # ============== BaseDetector Interface ==============
    
    def predict(self, image) -> List[Dict[str, Any]]:
        """
        Perform detection on image using current model.
        This makes DetectionAgent compatible with existing inference pipeline.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of detections with bbox, score, class info
        """
        return self.detect(image)
    
    # ============== Agent Methods ==============
    
    def observe(self, state: Dict[str, Any]) -> np.ndarray:
        """
        Process detection state into observation vector.
        
        Args:
            state: State containing image and optional detections
            
        Returns:
            Observation vector [num_detections, avg_confidence, entropy, model_used]
        """
        detections = state.get("detections", [])
        
        if not detections:
            return np.array([0, 0.0, 10.0, 0.0], dtype=np.float32)
        
        scores = np.array([d["score"] for d in detections], dtype=np.float32)
        
        # Calculate observation features
        num_detections = len(detections)
        avg_confidence = float(np.mean(scores))
        entropy = self._calculate_entropy(scores)
        
        # Encode current model as float
        model_encoding = self._encode_model(self.current_model)
        
        observation = np.array([
            num_detections,
            avg_confidence,
            entropy,
            model_encoding
        ], dtype=np.float32)
        
        return observation
    
    def _calculate_entropy(self, scores: np.ndarray) -> float:
        """Calculate entropy from detection scores."""
        if len(scores) == 0:
            return 10.0
        
        scores = np.clip(scores, 1e-6, 1.0)
        p = scores / (np.sum(scores) + 1e-12)
        ent = -np.sum(p * np.log(p + 1e-12))
        return float(ent)
    
    def _encode_model(self, model_name: str) -> float:
        """Encode model name as float for observation."""
        model_map = {"lightweight": 0.0, "yolo": 1.0, "faster_rcnn": 2.0}
        return model_map.get(model_name, 0.0)
    
    def act(self, observation: np.ndarray) -> Dict[str, Any]:
        """
        Select detection action based on observation.
        
        Args:
            observation: Processed observation vector
            
        Returns:
            Action dictionary with model selection and parameters
        """
        # Get detections from current model
        state = {"detections": self.detection_history[-1] if self.detection_history else []}
        
        # Use policy to decide model
        if self.policy:
            selected_model = self.policy.select_model(
                observation, 
                list(self.models.keys()),
                state.get("detections", [])
            )
        else:
            selected_model = self.current_model
        
        # Switch model if different
        action = {
            "action": "detect",
            "model": selected_model,
            "switch_model": selected_model != self.current_model
        }
        
        if selected_model != self.current_model:
            self.current_model = selected_model
            print(f"Switching to model: {selected_model}")
        
        return action
    
    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Perform detection on image using current model.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of detections with bbox, score, class info
        """
        model = self.models.get(self.current_model)
        if model is None:
            print(f"Error: Model {self.current_model} not available")
            return []
        
        detections = model.predict(image)
        
        # Store detection result
        self.detection_history.append(detections)
        
        return detections
    
    def detect_with_model(self, image: np.ndarray, model_name: str) -> List[Dict[str, Any]]:
        """
        Perform detection with a specific model.
        
        Args:
            image: Input image
            model_name: Name of model to use
            
        Returns:
            List of detections
        """
        model = self.models.get(model_name)
        if model is None:
            print(f"Warning: Model {model_name} not available")
            return []
        
        return model.predict(image)
    
    # ============== Data Integration ==============
    
    def load_image(self, image_path: str) -> np.ndarray:
        """Load image using the data loader."""
        return load_image_raw(image_path)
    
    def process_image(self, image: np.ndarray, target_size: Tuple[int, int] = None) -> np.ndarray:
        """Process image using ImageProcessor."""
        if target_size:
            return self.image_processor.resize(image, target_size[0], target_size[1])
        return image
    
    def detect_from_file(self, image_path: str, visualize: bool = True, 
                        output_path: str = None) -> Dict[str, Any]:
        """
        Complete detection pipeline from file.
        
        Args:
            image_path: Path to input image
            visualize: Whether to draw detection boxes
            output_path: Path to save output (optional)
            
        Returns:
            Dictionary with detections and metadata
        """
        # Load image
        image = self.load_image(image_path)
        
        # Detect
        detections = self.detect(image)
        
        # Visualize if requested
        result_image = image
        if visualize:
            result_image = self.visualizer.draw_detections(image, detections)
        
        # Save if output path provided
        if output_path:
            import cv2
            cv2.imwrite(output_path, result_image)
        
        return {
            "image_path": image_path,
            "detections": detections,
            "num_detections": len(detections),
            "model_used": self.current_model,
            "visualization": result_image if visualize else None
        }
    
    def detect_batch(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Run detection on a batch of images.
        
        Args:
            image_paths: List of image paths
            
        Returns:
            List of detection results
        """
        results = []
        for path in image_paths:
            result = self.detect_from_file(path, visualize=False)
            results.append(result)
        return results
    
    def detect_directory(self, directory: str, extensions: List[str] = None,
                        recursive: bool = False) -> List[Dict[str, Any]]:
        """
        Run detection on all images in a directory.
        
        Args:
            directory: Path to directory
            extensions: List of image extensions to process
            recursive: Whether to search recursively
            
        Returns:
            List of detection results
        """
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        dir_path = Path(directory)
        
        # Find all images
        if recursive:
            image_files = []
            for ext in extensions:
                image_files.extend(dir_path.rglob(f"*{ext}"))
        else:
            image_files = []
            for ext in extensions:
                image_files.extend(dir_path.glob(f"*{ext}"))
        
        # Process each image
        results = []
        for img_path in image_files:
            result = self.detect_from_file(str(img_path), visualize=False)
            results.append(result)
        
        return results
    
    # ============== Evaluation Integration ==============
    
    def evaluate_detections(self, predictions: List[Dict], 
                          ground_truth: List[Dict],
                          iou_threshold: float = 0.5) -> Dict[str, float]:
        """
        Evaluate detections against ground truth.
        
        Args:
            predictions: List of predicted detections
            ground_truth: List of ground truth annotations
            iou_threshold: IoU threshold for matching
            
        Returns:
            Dictionary with precision, recall, F1
        """
        precision, recall = self.metrics_calc.compute_precision_recall(
            predictions, ground_truth, iou_threshold
        )
        f1 = self.metrics_calc.compute_f1(precision, recall)
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    
    # ============== Learning Methods ==============
    
    def learn(self, observation: np.ndarray, action: Dict, reward: float,
              next_obs: np.ndarray, done: bool):
        """
        Update agent based on experience.
        
        Args:
            observation: Current observation
            action: Action taken
            reward: Reward received
            next_obs: Next observation
            done: Whether episode is done
        """
        self.store_experience(observation, action, reward, next_obs, done)
    
    def get_detection_stats(self) -> Dict[str, Any]:
        """Get detection statistics."""
        if not self.detection_history:
            return {"total_detections": 0}
        
        all_detections = []
        for dets in self.detection_history:
            all_detections.extend(dets)
        
        if not all_detections:
            return {"total_detections": 0}
        
        scores = [d["score"] for d in all_detections]
        
        return {
            "total_detections": len(all_detections),
            "avg_confidence": float(np.mean(scores)),
            "max_confidence": float(np.max(scores)),
            "min_confidence": float(np.min(scores)),
            "current_model": self.current_model,
            "available_models": list(self.models.keys())
        }
    
    def reset(self):
        """Reset agent state."""
        self.detection_history = []
        self.obs_history = []
        self.action_history = []
        self.reward_history = []
    
    def __repr__(self) -> str:
        return f"DetectionAgent(model={self.current_model}, steps={self.total_steps})"


# ============== Factory Function ==============

def create_detection_agent(config_path: str = "config.yaml") -> DetectionAgent:
    """
    Factory function to create a DetectionAgent from config.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Initialized DetectionAgent
    """
    from ..utils.config_loader import load_config
    
    config = load_config(config_path)
    return DetectionAgent(config)


def create_detection_agent_from_dict(config: Dict[str, Any]) -> DetectionAgent:
    """
    Factory function to create a DetectionAgent from config dict.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Initialized DetectionAgent
    """
    return DetectionAgent(config)
