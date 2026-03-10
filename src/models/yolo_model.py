from typing import List, Dict, Any
from .base_model import BaseDetector

class YOLODetector(BaseDetector):
    """Uses Ultralytics YOLO if installed."""

    def __init__(self, weights: str = "yolov8n.pt", conf: float = 0.35, iou: float = 0.5):
        try:
            from ultralytics import YOLO
        except Exception as e:
            raise ImportError("Ultralytics is not installed. Run: pip install ultralytics") from e

        self.model = YOLO(weights)
        self.conf = conf
        self.iou = iou

    def predict(self, image) -> List[Dict[str, Any]]:
        results = self.model.predict(source=image, conf=self.conf, iou=self.iou, verbose=False)
        out: List[Dict[str, Any]] = []
        r = results[0]
        names = r.names
        if r.boxes is None:
            return out
        for b in r.boxes:
            xyxy = b.xyxy[0].tolist()
            score = float(b.conf[0].item())
            cls = int(b.cls[0].item())
            out.append({
                "bbox": [int(x) for x in xyxy],
                "score": score,
                "class_id": cls,
                "class_name": str(names.get(cls, str(cls)))
            })
        return out
