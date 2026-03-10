import numpy as np
from .base_model import BaseDetector

class LightweightDetector(BaseDetector):
    """Light fallback model producing a fake prediction for testing."""

    def __init__(self, conf_threshold: float = 0.35):
        self.conf_threshold = conf_threshold

    def predict(self, image):
        h, w = image.shape[:2]
        x1, y1 = int(w*0.35), int(h*0.35)
        x2, y2 = int(w*0.65), int(h*0.65)
        score = float(np.clip(np.random.normal(0.7, 0.1), 0.0, 1.0))
        if score < self.conf_threshold:
            return []
        return [{
            "bbox": [x1, y1, x2, y2],
            "score": score,
            "class_id": 0,
            "class_name": "object"
        }]
