"""
Policy implementations for the Agentic Object Detection System.
Defines different strategies for model selection.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class Policy(ABC):
    """Abstract base class for policies."""
    
    @abstractmethod
    def select_model(self, observation: np.ndarray, available_models: List[str], 
                    detections: List[Dict]) -> str:
        """
        Select a model based on observation.
        
        Args:
            observation: Current observation vector
            available_models: List of available model names
            detections: Current detections
            
        Returns:
            Selected model name
        """
        pass


class RandomPolicy(Policy):
    """Random model selection policy."""
    
    def __init__(self, available_models: List[str]):
        """
        Initialize random policy.
        
        Args:
            available_models: List of available model names
        """
        self.available_models = available_models
        self.rng = np.random.default_rng()
    
    def select_model(self, observation: np.ndarray, available_models: List[str],
                    detections: List[Dict]) -> str:
        """Select a random model."""
        models = available_models if available_models else self.available_models
        return self.rng.choice(models)


class EntropyPolicy(Policy):
    """Entropy-based model selection policy."""
    
    def __init__(self, entropy_threshold: float = 1.2,
                 low_entropy_model: str = "lightweight",
                 high_entropy_model: str = "yolo"):
        """
        Initialize entropy policy.
        
        Args:
            entropy_threshold: Threshold for entropy-based decision
            low_entropy_model: Model to use when entropy is low
            high_entropy_model: Model to use when entropy is high
        """
        self.entropy_threshold = entropy_threshold
        self.low_entropy_model = low_entropy_model
        self.high_entropy_model = high_entropy_model
    
    def select_model(self, observation: np.ndarray, available_models: List[str],
                    detections: List[Dict]) -> str:
        """
        Select model based on entropy of detection scores.
        
        Low entropy (confident predictions) → lightweight model
        High entropy (uncertain predictions) → heavy model (YOLO)
        """
        # Extract entropy from observation (index 2)
        if len(observation) >= 3:
            entropy = float(observation[2])
        else:
            # Calculate from detections
            entropy = self._calculate_entropy(detections)
        
        if entropy <= self.entropy_threshold:
            selected = self.low_entropy_model
        else:
            selected = self.high_entropy_model
        
        # Verify model is available
        if selected not in available_models:
            selected = available_models[0] if available_models else "lightweight"
        
        return selected
    
    def _calculate_entropy(self, detections: List[Dict]) -> float:
        """Calculate entropy from detection scores."""
        if not detections:
            return 10.0  # High entropy for no detections
        
        scores = np.array([d["score"] for d in detections], dtype=np.float32)
        scores = np.clip(scores, 1e-6, 1.0)
        p = scores / (np.sum(scores) + 1e-12)
        ent = -np.sum(p * np.log(p + 1e-12))
        return float(ent)


class ConfidencePolicy(Policy):
    """Confidence-based model selection policy."""
    
    def __init__(self, confidence_threshold: float = 0.7,
                 low_conf_model: str = "lightweight",
                 high_conf_model: str = "yolo"):
        """
        Initialize confidence policy.
        
        Args:
            confidence_threshold: Threshold for confidence-based decision
            low_conf_model: Model to use when confidence is low
            high_conf_model: Model to use when confidence is high
        """
        self.confidence_threshold = confidence_threshold
        self.low_conf_model = low_conf_model
        self.high_conf_model = high_conf_model
    
    def select_model(self, observation: np.ndarray, available_models: List[str],
                    detections: List[Dict]) -> str:
        """
        Select model based on average confidence of detections.
        
        High confidence → lightweight model
        Low confidence → heavy model (YOLO)
        """
        # Extract confidence from observation (index 1)
        if len(observation) >= 2:
            avg_confidence = float(observation[1])
        else:
            # Calculate from detections
            avg_confidence = self._calculate_avg_confidence(detections)
        
        if avg_confidence >= self.confidence_threshold:
            selected = self.high_conf_model  # High confidence - use accurate model
        else:
            selected = self.low_conf_model  # Low confidence - use fast model
        
        # Verify model is available
        if selected not in available_models:
            selected = available_models[0] if available_models else "lightweight"
        
        return selected
    
    def _calculate_avg_confidence(self, detections: List[Dict]) -> float:
        """Calculate average confidence from detections."""
        if not detections:
            return 0.0
        
        scores = [d["score"] for d in detections]
        return float(np.mean(scores))


class AdaptivePolicy(Policy):
    """Adaptive policy that considers both entropy and confidence."""
    
    def __init__(self, entropy_threshold: float = 1.2,
                 confidence_threshold: float = 0.5,
                 models: Dict[str, str] = None):
        """
        Initialize adaptive policy.
        
        Args:
            entropy_threshold: Threshold for entropy
            confidence_threshold: Threshold for confidence
            models: Dict mapping (entropy_level, conf_level) to model
        """
        self.entropy_threshold = entropy_threshold
        self.confidence_threshold = confidence_threshold
        
        # Default model mapping
        self.models = models or {
            ("low", "high"): "lightweight",   # Low entropy, high confidence
            ("low", "low"): "yolo",           # Low entropy, low confidence
            ("high", "high"): "yolo",          # High entropy, high confidence
            ("high", "low"): "faster_rcnn",    # High entropy, low confidence
        }
    
    def select_model(self, observation: np.ndarray, available_models: List[str],
                    detections: List[Dict]) -> str:
        """Select model based on both entropy and confidence."""
        # Extract features from observation
        if len(observation) >= 3:
            avg_confidence = float(observation[1])
            entropy = float(observation[2])
        else:
            entropy = self._calculate_entropy(detections)
            avg_confidence = self._calculate_avg_confidence(detections)
        
        # Determine levels
        entropy_level = "low" if entropy <= self.entropy_threshold else "high"
        conf_level = "high" if avg_confidence >= self.confidence_threshold else "low"
        
        # Get model
        key = (entropy_level, conf_level)
        selected = self.models.get(key, "lightweight")
        
        # Verify model is available
        if selected not in available_models:
            # Fallback logic
            if avg_confidence >= self.confidence_threshold:
                selected = "yolo"
            else:
                selected = "lightweight"
            
            if selected not in available_models:
                selected = available_models[0] if available_models else "lightweight"
        
        return selected
    
    def _calculate_entropy(self, detections: List[Dict]) -> float:
        if not detections:
            return 10.0
        scores = np.array([d["score"] for d in detections], dtype=np.float32)
        scores = np.clip(scores, 1e-6, 1.0)
        p = scores / (np.sum(scores) + 1e-12)
        return float(-np.sum(p * np.log(p + 1e-12)))
    
    def _calculate_avg_confidence(self, detections: List[Dict]) -> float:
        if not detections:
            return 0.0
        return float(np.mean([d["score"] for d in detections]))


class ThresholdCascadePolicy(Policy):
    """Cascade policy that uses multiple thresholds."""
    
    def __init__(self, thresholds: List[float] = None,
                 models: List[str] = None):
        """
        Initialize cascade policy.
        
        Args:
            thresholds: List of confidence thresholds [0.3, 0.5, 0.7, 0.9]
            models: List of models in order of complexity
        """
        self.thresholds = thresholds or [0.3, 0.5, 0.7, 0.9]
        self.models = models or ["lightweight", "yolo", "faster_rcnn", "yolo"]
    
    def select_model(self, observation: np.ndarray, available_models: List[str],
                    detections: List[Dict]) -> str:
        """
        Select model based on confidence thresholds.
        Starts with fastest model, escalates if confidence is low.
        """
        # Get max confidence from detections
        if len(observation) >= 2:
            max_confidence = float(observation[1])  # Could be avg, adjust as needed
        else:
            max_confidence = self._get_max_confidence(detections)
        
        # Find appropriate model based on confidence
        for i, threshold in enumerate(self.thresholds):
            if max_confidence >= threshold and i < len(self.models):
                selected = self.models[i]
                if selected in available_models:
                    return selected
        
        # Default to last model
        selected = self.models[-1] if self.models else "lightweight"
        if selected not in available_models:
            selected = available_models[0] if available_models else "lightweight"
        
        return selected
    
    def _get_max_confidence(self, detections: List[Dict]) -> float:
        if not detections:
            return 0.0
        return float(max([d["score"] for d in detections]))
