from typing import List, Dict, Any
from .base_model import BaseDetector

class FasterRCNNDetector(BaseDetector):
    """Optional PyTorch FasterRCNN detector (heavy)."""

    def __init__(self, conf_threshold: float = 0.35):
        self.conf_threshold = conf_threshold
        try:
            import torch
            import torchvision
            from torchvision.transforms import functional as F
        except Exception as e:
            raise ImportError("torch/torchvision not installed.") from e

        self.torch = torch
        self.F = F
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
        self.model.eval()

    def predict(self, image) -> List[Dict[str, Any]]:
        torch = self.torch
        img = self.F.to_tensor(image)
        with torch.no_grad():
            pred = self.model([img])[0]

        out = []
        boxes = pred["boxes"].cpu().numpy()
        scores = pred["scores"].cpu().numpy()
        labels = pred["labels"].cpu().numpy()

        for box, score, label in zip(boxes, scores, labels):
            if float(score) < self.conf_threshold:
                continue
            x1, y1, x2, y2 = box.astype(int).tolist()
            out.append({
                "bbox": [x1, y1, x2, y2],
                "score": float(score),
                "class_id": int(label),
                "class_name": f"class_{int(label)}"
            })
        return out
