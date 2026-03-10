"""
Vision tools for the Agentic Object Detection System.
Provides image processing and analysis utilities.
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path


class ImageProcessor:
    """Image preprocessing and enhancement tools."""
    
    @staticmethod
    def load_image(path: str) -> np.ndarray:
        """Load image from file."""
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Failed to load image: {path}")
        return img
    
    @staticmethod
    def save_image(image: np.ndarray, path: str) -> bool:
        """Save image to file."""
        return cv2.imwrite(path, image)
    
    @staticmethod
    def resize(image: np.ndarray, width: int, height: int, 
              keep_aspect: bool = True) -> np.ndarray:
        """Resize image to specified dimensions."""
        if keep_aspect:
            h, w = image.shape[:2]
            scale = min(width / w, height / h)
            new_w, new_h = int(w * scale), int(h * scale)
            return cv2.resize(image, (new_w, new_h))
        return cv2.resize(image, (width, height))
    
    @staticmethod
    def normalize(image: np.ndarray, mean: List[float] = None, 
                std: List[float] = None) -> np.ndarray:
        """Normalize image with mean and std."""
        img = image.astype(np.float32) / 255.0
        
        if mean is not None:
            mean = np.array(mean, dtype=np.float32)
            img = img - mean
        
        if std is not None:
            std = np.array(std, dtype=np.float32)
            img = img / std
        
        return img
    
    @staticmethod
    def denormalize(image: np.ndarray, mean: List[float] = None,
                   std: List[float] = None) -> np.ndarray:
        """Denormalize image."""
        img = image.copy()
        
        if std is not None:
            std = np.array(std, dtype=np.float32)
            img = img * std
        
        if mean is not None:
            mean = np.array(mean, dtype=np.float32)
            img = img + mean
        
        return np.clip(img * 255.0, 0, 255).astype(np.uint8)
    
    @staticmethod
    def crop(image: np.ndarray, x: int, y: int, 
            width: int, height: int) -> np.ndarray:
        """Crop image to specified region."""
        h, w = image.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(w, x + width), min(h, y + height)
        return image[y1:y2, x1:x2]
    
    @staticmethod
    def pad(image: np.ndarray, padding: int, color: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
        """Pad image with border."""
        return cv2.copyMakeBorder(
            image, padding, padding, padding, padding,
            cv2.BORDER_CONSTANT, value=color
        )


class DetectionVisualizer:
    """Visualization tools for detections."""
    
    COLORS = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (128, 0, 0), (0, 128, 0), (0, 0, 128)
    ]
    
    @staticmethod
    def draw_detections(image: np.ndarray, detections: List[Dict],
                       show_labels: bool = True, show_scores: bool = True,
                       thickness: int = 2) -> np.ndarray:
        """Draw detection boxes on image."""
        img = image.copy()
        
        for i, det in enumerate(detections):
            bbox = det["bbox"]
            x1, y1, x2, y2 = [int(v) for v in bbox]
            
            # Get color
            color = DetectionVisualizer.COLORS[i % len(DetectionVisualizer.COLORS)]
            
            # Draw rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
            
            # Prepare label
            if show_labels or show_scores:
                label_parts = []
                if show_labels and "class_name" in det:
                    label_parts.append(det["class_name"])
                if show_scores and "score" in det:
                    label_parts.append(f"{det['score']:.2f}")
                
                if label_parts:
                    label = " ".join(label_parts)
                    
                    # Draw label background
                    (label_w, label_h), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                    )
                    cv2.rectangle(
                        img, 
                        (x1, y1 - label_h - 10),
                        (x1 + label_w, y1),
                        color, -1
                    )
                    
                    # Draw label text
                    cv2.putText(
                        img, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
                    )
        
        return img
    
    @staticmethod
    def draw_heatmap(image: np.ndarray, detections: List[Dict],
                    sigma: float = 10.0) -> np.ndarray:
        """Draw attention heatmap based on detection scores."""
        h, w = image.shape[:2]
        heatmap = np.zeros((h, w), dtype=np.float32)
        
        for det in detections:
            bbox = det["bbox"]
            x1, y1, x2, y2 = [int(v) for v in bbox]
            score = det.get("score", 1.0)
            
            # Create Gaussian blob at box center
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            for y in range(max(0, cy - 50), min(h, cy + 50)):
                for x in range(max(0, cx - 50), min(w, cx + 50)):
                    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                    heatmap[y, x] += score * np.exp(-dist**2 / (2 * sigma**2))
        
        # Normalize and apply colormap
        heatmap = np.clip(heatmap, 0, 1)
        heatmap = (heatmap * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Blend with original
        return cv2.addWeighted(image, 0.6, heatmap_colored, 0.4, 0)


class FeatureExtractor:
    """Extract features from images and detections."""
    
    @staticmethod
    def extract_region_features(image: np.ndarray, bbox: List[int]) -> np.ndarray:
        """Extract features from a region (ROI)."""
        x1, y1, x2, y2 = [int(v) for v in bbox]
        roi = image[y1:y2, x1:x2]
        
        if roi.size == 0:
            return np.array([])
        
        # Flatten and normalize
        features = roi.flatten().astype(np.float32)
        features = features / 255.0
        
        return features
    
    @staticmethod
    def compute_histogram(image: np.ndarray, bins: int = 256) -> np.ndarray:
        """Compute color histogram."""
        hist = []
        for i in range(3):  # BGR channels
            h = cv2.calcHist([image], [i], None, [bins], [0, 256])
            hist.append(h.flatten())
        return np.concatenate(hist)
    
    @staticmethod
    def extract_sift_features(image: np.ndarray, max_features: int = 100) -> Tuple:
        """Extract SIFT features from image."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        sift = cv2.SIFT_create(nfeatures=max_features)
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        
        return keypoints, descriptors
    
    @staticmethod
    def compute_similarity(det1: Dict, det2: Dict) -> float:
        """Compute similarity between two detections."""
        # IoU similarity
        bbox1 = det1["bbox"]
        bbox2 = det2["bbox"]
        
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        iou = intersection / union if union > 0 else 0.0
        
        # Class similarity
        class_match = int(det1.get("class_id") == det2.get("class_id"))
        
        # Combined score
        return 0.7 * iou + 0.3 * class_match


class MetricsCalculator:
    """Calculate detection metrics."""
    
    @staticmethod
    def compute_iou(bbox1: List[float], bbox2: List[float]) -> float:
        """Compute IoU between two bounding boxes."""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def compute_precision_recall(predictions: List[Dict], 
                                ground_truth: List[Dict],
                                iou_threshold: float = 0.5) -> Tuple[float, float]:
        """Compute precision and recall."""
        if not predictions:
            return 0.0, 0.0
        if not ground_truth:
            return 0.0, 0.0
        
        # Match predictions to ground truth
        matched_gt = set()
        true_positives = 0
        
        for pred in predictions:
            best_iou = 0.0
            best_idx = -1
            
            for idx, gt in enumerate(ground_truth):
                if idx in matched_gt:
                    continue
                
                iou = MetricsCalculator.compute_iou(pred["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            
            if best_iou >= iou_threshold and best_idx >= 0:
                true_positives += 1
                matched_gt.add(best_idx)
        
        precision = true_positives / len(predictions) if predictions else 0.0
        recall = true_positives / len(ground_truth) if ground_truth else 0.0
        
        return precision, recall
    
    @staticmethod
    def compute_f1(precision: float, recall: float) -> float:
        """Compute F1 score."""
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
