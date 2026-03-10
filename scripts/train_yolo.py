#!/usr/bin/env python3
"""
Train YOLO model on the custom dataset.
Uses Ultralytics YOLO API for training.
"""

import os
import yaml
from pathlib import Path


def create_data_yaml(project_root):
    """Create data.yaml for YOLO training."""
    data_config = {
        'path': str(project_root / 'data'),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': 3,  # number of classes
        'names': {
            0: 'person',
            1: 'car',
            2: 'dog'
        }
    }
    
    yaml_path = project_root / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False)
    
    print(f"Created data.yaml at: {yaml_path}")
    return yaml_path


def train_yolo(data_yaml_path, model_name='yolov8n.pt', epochs=50, imgsz=640, project_root=None):
    """
    Train YOLO model.
    
    Args:
        data_yaml_path: Path to data.yaml
        model_name: Pretrained model to use (yolov8n.pt, yolov8s.pt, etc.)
        epochs: Number of training epochs
        imgsz: Image size
        project_root: Project root directory
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: Ultralytics not installed!")
        print("Install with: pip install ultralytics")
        return None
    
    # Create model from pretrained weights
    print(f"\nLoading pretrained model: {model_name}")
    model = YOLO(model_name)
    
    # Training results directory
    results_dir = project_root / 'models' / 'trained_weights'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Train the model
    print(f"\nStarting training...")
    print(f"  Data config: {data_yaml_path}")
    print(f"  Epochs: {epochs}")
    print(f"  Image size: {imgsz}")
    print(f"  Results will be saved to: {results_dir}")
    
    results = model.train(
        data=str(data_yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        project=str(results_dir),
        name='yolo_train',
        exist_ok=True,
        verbose=True,
        device='cpu',  # Use CPU by default (change to '0' for GPU)
        patience=10,   # Early stopping patience
        save=True,     # Save trained weights
        plots=True,    # Generate training plots
    )
    
    # Find the best trained weights
    best_weights = results_dir / 'yolo_train' / 'weights' / 'best.pt'
    last_weights = results_dir / 'yolo_train' / 'weights' / 'last.pt'
    
    print("\n" + "=" * 50)
    print("Training completed!")
    print("=" * 50)
    
    if best_weights.exists():
        print(f"Best weights: {best_weights}")
    if last_weights.exists():
        print(f"Last weights: {last_weights}")
    
    return results


def main():
    """Main training function."""
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    print("=" * 50)
    print("YOLO Training Script")
    print("=" * 50)
    
    # Create data.yaml
    print("\n[1/2] Creating data configuration...")
    data_yaml = create_data_yaml(project_root)
    
    # Check if ultralytics is installed
    try:
        import ultralytics
        print(f"Ultralytics version: {ultralytics.__version__}")
    except ImportError:
        print("ERROR: Ultralytics is not installed!")
        print("Please install with: pip install ultralytics")
        return
    
    # Train YOLO
    print("\n[2/2] Starting YOLO training...")
    results = train_yolo(
        data_yaml_path=data_yaml,
        model_name='yolov8n.pt',  # Use nano for fast training
        epochs=50,
        imgsz=640,
        project_root=project_root
    )
    
    print("\n" + "=" * 50)
    print("Training complete!")
    print("=" * 50)
    
    # Update config.yaml to use trained weights
    config_path = project_root / 'config.yaml'
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update to use trained weights
    trained_weights_path = project_root / 'models' / 'trained_weights' / 'yolo_train' / 'weights' / 'best.pt'
    if trained_weights_path.exists():
        config['model']['yolo_weights'] = str(trained_weights_path)
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"\nUpdated config.yaml to use trained weights: {trained_weights_path}")


if __name__ == "__main__":
    main()
