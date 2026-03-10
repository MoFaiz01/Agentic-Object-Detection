#!/usr/bin/env python3
"""
Complete training pipeline for YOLO on custom dataset.
This script:
1. Organizes data into YOLO format
2. Converts COCO annotations to YOLO format
3. Creates data.yaml configuration
4. Trains the YOLO model
"""

import os
import sys
import json
import yaml
import shutil
import argparse
from pathlib import Path


def create_yolo_labels(coco_annotation_path, images_dir, output_labels_dir, categories):
    """Convert COCO JSON annotations to YOLO txt format."""
    output_labels_dir = Path(output_labels_dir)
    output_labels_dir.mkdir(parents=True, exist_ok=True)
    
    with open(coco_annotation_path, 'r') as f:
        coco = json.load(f)
    
    cat_id_to_name = {cat['id']: cat['name'] for cat in coco['categories']}
    name_to_id = {name: idx for idx, name in enumerate(categories)}
    
    annotations_by_image = {}
    for ann in coco['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)
    
    processed = 0
    for img_info in coco['images']:
        img_id = img_info['id']
        img_filename = img_info['file_name']
        img_width = img_info['width']
        img_height = img_info['height']
        
        img_annotations = annotations_by_image.get(img_id, [])
        
        txt_filename = Path(img_filename).stem + '.txt'
        txt_path = output_labels_dir / txt_filename
        
        with open(txt_path, 'w') as f:
            for ann in img_annotations:
                coco_cat_id = ann['category_id']
                class_name = cat_id_to_name[coco_cat_id]
                yolo_class_id = name_to_id[class_name]
                
                x, y, w, h = ann['bbox']
                x_center = (x + w / 2) / img_width
                y_center = (y + h / 2) / img_height
                w_norm = w / img_width
                h_norm = h / img_height
                
                f.write(f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
        
        processed += 1
    
    return processed


def organize_and_convert(project_root, categories):
    """Organize data and convert annotations to YOLO format."""
    data_dir = project_root / "data"
    
    print("\n[Step 1] Organizing data structure...")
    
    for split in ['train', 'val', 'test']:
        split_dir = data_dir / split
        images_dir = split_dir / 'images'
        labels_dir = split_dir / 'labels'
        
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy images to images folder
        for img_file in split_dir.glob('*.jpg'):
            dest = images_dir / img_file.name
            if not dest.exists():
                shutil.copy(img_file, dest)
        
        for img_file in split_dir.glob('*.png'):
            dest = images_dir / img_file.name
            if not dest.exists():
                shutil.copy(img_file, dest)
    
    print("[Step 2] Converting COCO annotations to YOLO format...")
    
    for split in ['train', 'val', 'test']:
        ann_file = data_dir / split / 'annotations.json'
        if ann_file.exists():
            images_dir = data_dir / split / 'images'
            labels_dir = data_dir / split / 'labels'
            
            count = create_yolo_labels(
                coco_annotation_path=ann_file,
                images_dir=images_dir,
                output_labels_dir=labels_dir,
                categories=categories
            )
            print(f"  {split}: Converted {count} images")


def create_data_config(project_root, categories):
    """Create data.yaml for YOLO training."""
    data_config = {
        'path': str(project_root / 'data'),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': len(categories),
        'names': {idx: name for idx, name in enumerate(categories)}
    }
    
    yaml_path = project_root / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False)
    
    print(f"\n[Step 3] Created data.yaml at: {yaml_path}")
    return yaml_path


def train_model(data_yaml_path, model_name, epochs, project_root):
    """Train YOLO model."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("\nERROR: Ultralytics not installed!")
        print("Please install with: pip install ultralytics")
        return None
    
    print(f"\n[Step 4] Loading pretrained model: {model_name}")
    model = YOLO(model_name)
    
    results_dir = project_root / 'models' / 'trained_weights'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[Step 5] Training YOLO model...")
    print(f"  - Data config: {data_yaml_path}")
    print(f"  - Epochs: {epochs}")
    print(f"  - Device: CPU (use '0' for GPU)")
    
    results = model.train(
        data=str(data_yaml_path),
        epochs=epochs,
        imgsz=640,
        project=str(results_dir),
        name='yolo_train',
        verbose=True,
        device='cpu',
        patience=15,
        save=True,
        plots=True,
    )
    
    print("\n" + "=" * 50)
    print("Training completed!")
    print("=" * 50)
    
    save_dir = Path(results.save_dir)
    best_weights = save_dir / 'weights' / 'best.pt'
    last_weights = save_dir / 'weights' / 'last.pt'
    
    if best_weights.exists():
        print(f"Best weights: {best_weights}")
    if last_weights.exists():
        print(f"Last weights: {last_weights}")
    
    return results, best_weights if best_weights.exists() else None


def update_config(project_root, trained_weights_path):
    """Update config.yaml to use trained weights."""
    config_path = project_root / 'config.yaml'
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if trained_weights_path and trained_weights_path.exists():
        config['model']['yolo_weights'] = str(trained_weights_path)
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"\n[Step 6] Updated config.yaml to use trained weights!")
        print(f"  New weights path: {trained_weights_path}")


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description="YOLO training pipeline")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Base YOLO model")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    # Configuration
    categories = ["person", "car", "dog"]
    model_name = args.model  # Use yolov8s.pt, yolov8m.pt for better accuracy
    epochs = args.epochs
    
    print("=" * 60)
    print("YOLO Training Pipeline")
    print("=" * 60)
    
    # Step 1-2: Organize data and convert annotations
    organize_and_convert(project_root, categories)
    
    # Step 3: Create data configuration
    data_yaml = create_data_config(project_root, categories)
    
    # Step 4-5: Train model
    results, best_weights = train_model(data_yaml, model_name, epochs, project_root)
    
    # Step 6: Update config
    update_config(project_root, best_weights)
    
    print("\n" + "=" * 60)
    print("Training pipeline completed successfully!")
    print("=" * 60)
    print("\nTo run inference with trained model:")
    print("  python -m src.inference.detect_image --image path/to/image.jpg")


if __name__ == "__main__":
    main()
