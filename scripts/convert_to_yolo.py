#!/usr/bin/env python3
"""
Convert COCO format annotations to YOLO format for training.
YOLO format: <class_id> <x_center> <y_center> <width> <height>
All values are normalized (0-1)
"""

import json
import os
from pathlib import Path
from PIL import Image


def convert_coco_to_yolo(coco_annotation_path, images_dir, output_dir, categories):
    """
    Convert COCO JSON annotations to YOLO txt format.
    
    Args:
        coco_annotation_path: Path to COCO format JSON
        images_dir: Directory containing images
        output_dir: Directory to save YOLO format txt files
        categories: List of category names in order
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load COCO annotations
    with open(coco_annotation_path, 'r') as f:
        coco = json.load(f)
    
    # Create category mapping
    cat_id_to_name = {cat['id']: cat['name'] for cat in coco['categories']}
    name_to_id = {name: idx for idx, name in enumerate(categories)}
    
    # Group annotations by image_id
    annotations_by_image = {}
    for ann in coco['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)
    
    # Process each image
    processed = 0
    for img_info in coco['images']:
        img_id = img_info['id']
        img_filename = img_info['file_name']
        img_width = img_info['width']
        img_height = img_info['height']
        
        # Get annotations for this image
        img_annotations = annotations_by_image.get(img_id, [])
        
        # Create YOLO format txt file
        txt_filename = Path(img_filename).stem + '.txt'
        txt_path = output_dir / txt_filename
        
        with open(txt_path, 'w') as f:
            for ann in img_annotations:
                # Get class ID in YOLO format
                coco_cat_id = ann['category_id']
                class_name = cat_id_to_name[coco_cat_id]
                yolo_class_id = name_to_id[class_name]
                
                # Get bounding box (COCO format: x, y, width, height)
                x, y, w, h = ann['bbox']
                
                # Convert to YOLO format (center x, center y, width, height)
                # All normalized by image dimensions
                x_center = (x + w / 2) / img_width
                y_center = (y + h / 2) / img_height
                w_norm = w / img_width
                h_norm = h / img_height
                
                # Write in YOLO format
                f.write(f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
        
        processed += 1
    
    print(f"Converted {processed} images to YOLO format")
    print(f"Output saved to: {output_dir}")
    return processed


def main():
    """Main function to convert all datasets."""
    project_root = Path(__file__).parent.parent
    categories = ["person", "car", "dog"]
    
    print("=" * 50)
    print("Converting COCO annotations to YOLO format")
    print("=" * 50)
    
    # Convert training data
    print("\n[1/3] Converting training data...")
    convert_coco_to_yolo(
        coco_annotation_path=project_root / "data" / "train" / "annotations.json",
        images_dir=project_root / "data" / "train",
        output_dir=project_root / "data" / "train" / "labels",
        categories=categories
    )
    
    # Convert validation data
    print("\n[2/3] Converting validation data...")
    convert_coco_to_yolo(
        coco_annotation_path=project_root / "data" / "val" / "annotations.json",
        images_dir=project_root / "data" / "val",
        output_dir=project_root / "data" / "val" / "labels",
        categories=categories
    )
    
    # Convert test data (for reference, though we don't train on it)
    print("\n[3/3] Converting test data...")
    convert_coco_to_yolo(
        coco_annotation_path=project_root / "data" / "test" / "annotations.json",
        images_dir=project_root / "data" / "test",
        output_dir=project_root / "data" / "test" / "labels",
        categories=categories
    )
    
    print("\n" + "=" * 50)
    print("Conversion complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
