#!/usr/bin/env python3
"""
Script to generate sample test data for the Agentic Object Detection System.
Creates synthetic images and annotations for testing the pipeline.
"""

import os
import numpy as np
import cv2
import json
from pathlib import Path

def create_sample_image(width=640, height=480, filename=None):
    """Create a synthetic test image with random shapes."""
    # Create random background
    bg_color = np.random.randint(50, 200, 3, dtype=np.uint8)
    image = np.full((height, width, 3), bg_color, dtype=np.uint8)
    
    # Add random rectangles (simulating objects)
    num_objects = np.random.randint(1, 4)
    bboxes = []
    
    for _ in range(num_objects):
        # Random position and size
        x1 = np.random.randint(50, width - 150)
        y1 = np.random.randint(50, height - 150)
        w = np.random.randint(80, 150)
        h = np.random.randint(80, 150)
        x2 = min(x1 + w, width - 50)
        y2 = min(y1 + h, height - 50)
        
        # Random color
        color = tuple(np.random.randint(0, 255, 3).tolist())
        
        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), color, -1)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 2)
        
        bboxes.append({
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "class_id": np.random.randint(0, 3),
            "class_name": np.random.choice(["person", "car", "dog"])
        })
    
    if filename:
        cv2.imwrite(filename, image)
    
    return image, bboxes


def generate_test_data(base_dir="data", num_images=5):
    """Generate test dataset."""
    test_dir = Path(base_dir) / "test"
    test_dir.mkdir(parents=True, exist_ok=True)
    
    annotations = {"images": [], "annotations": [], "categories": [
        {"id": 0, "name": "person"},
        {"id": 1, "name": "car"},
        {"id": 2, "name": "dog"}
    ]}
    
    ann_id = 1
    for i in range(num_images):
        img_path = test_dir / f"test_image_{i:03d}.jpg"
        img, bboxes = create_sample_image(filename=str(img_path))
        
        img_id = i + 1
        annotations["images"].append({
            "id": img_id,
            "file_name": f"test_image_{i:03d}.jpg",
            "width": img.shape[1],
            "height": img.shape[0]
        })
        
        for bbox in bboxes:
            annotations["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": bbox["class_id"],
                "bbox": bbox["bbox"],
                "area": (bbox["bbox"][2] - bbox["bbox"][0]) * (bbox["bbox"][3] - bbox["bbox"][1]),
                "iscrowd": 0
            })
            ann_id += 1
        
        print(f"Created: {img_path}")
    
    # Save annotations
    ann_path = test_dir / "annotations.json"
    with open(ann_path, 'w') as f:
        json.dump(annotations, f, indent=2)
    print(f"Created: {ann_path}")
    
    return annotations


def generate_train_data(base_dir="data", num_images=10):
    """Generate training dataset."""
    train_dir = Path(base_dir) / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    
    annotations = {"images": [], "annotations": [], "categories": [
        {"id": 0, "name": "person"},
        {"id": 1, "name": "car"},
        {"id": 2, "name": "dog"}
    ]}
    
    ann_id = 1
    for i in range(num_images):
        img_path = train_dir / f"train_image_{i:03d}.jpg"
        img, bboxes = create_sample_image(filename=str(img_path))
        
        img_id = i + 1
        annotations["images"].append({
            "id": img_id,
            "file_name": f"train_image_{i:03d}.jpg",
            "width": img.shape[1],
            "height": img.shape[0]
        })
        
        for bbox in bboxes:
            annotations["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": bbox["class_id"],
                "bbox": bbox["bbox"],
                "area": (bbox["bbox"][2] - bbox["bbox"][0]) * (bbox["bbox"][3] - bbox["bbox"][1]),
                "iscrowd": 0
            })
            ann_id += 1
        
        print(f"Created: {img_path}")
    
    # Save annotations
    ann_path = train_dir / "annotations.json"
    with open(ann_path, 'w') as f:
        json.dump(annotations, f, indent=2)
    print(f"Created: {ann_path}")
    
    return annotations


def generate_val_data(base_dir="data", num_images=3):
    """Generate validation dataset."""
    val_dir = Path(base_dir) / "val"
    val_dir.mkdir(parents=True, exist_ok=True)
    
    annotations = {"images": [], "annotations": [], "categories": [
        {"id": 0, "name": "person"},
        {"id": 1, "name": "car"},
        {"id": 2, "name": "dog"}
    ]}
    
    ann_id = 1
    for i in range(num_images):
        img_path = val_dir / f"val_image_{i:03d}.jpg"
        img, bboxes = create_sample_image(filename=str(img_path))
        
        img_id = i + 1
        annotations["images"].append({
            "id": img_id,
            "file_name": f"val_image_{i:03d}.jpg",
            "width": img.shape[1],
            "height": img.shape[0]
        })
        
        for bbox in bboxes:
            annotations["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": bbox["class_id"],
                "bbox": bbox["bbox"],
                "area": (bbox["bbox"][2] - bbox["bbox"][0]) * (bbox["bbox"][3] - bbox["bbox"][1]),
                "iscrowd": 0
            })
            ann_id += 1
        
        print(f"Created: {img_path}")
    
    # Save annotations
    ann_path = val_dir / "annotations.json"
    with open(ann_path, 'w') as f:
        json.dump(annotations, f, indent=2)
    print(f"Created: {ann_path}")
    
    return annotations


def main():
    # Set seed for reproducibility
    np.random.seed(42)
    
    print("=" * 50)
    print("Generating Sample Data for Object Detection")
    print("=" * 50)
    
    # Change to project directory
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    print("\n[1/3] Generating test data...")
    generate_test_data("data", num_images=5)
    
    print("\n[2/3] Generating training data...")
    generate_train_data("data", num_images=10)
    
    print("\n[3/3] Generating validation data...")
    generate_val_data("data", num_images=3)
    
    print("\n" + "=" * 50)
    print("Sample data generation complete!")
    print("=" * 50)
    print("\nYou can now run inference:")
    print("  python -m src.inference.detect_image --image data/test/test_image_000.jpg")


if __name__ == "__main__":
    main()
