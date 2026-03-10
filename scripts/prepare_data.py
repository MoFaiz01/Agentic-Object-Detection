#!/usr/bin/env python3
"""
Prepare data for YOLO training.
Creates proper directory structure and converts annotations.
"""

import os
import shutil
from pathlib import Path


def organize_data_for_yolo(project_root):
    """Organize data into YOLO format: images/ and labels/ subdirectories."""
    
    data_dir = project_root / "data"
    
    # Create label directories
    for split in ['train', 'val', 'test']:
        labels_dir = data_dir / split / 'labels'
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if images already exist in correct location
        images_dir = data_dir / split
        existing_images = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
        
        print(f"{split}: Found {len(existing_images)} images")
    
    print("\nData organization complete!")
    print("YOLO expects:")
    print("  - data/train/images/ and data/train/labels/")
    print("  - data/val/images/ and data/val/labels/")
    print("  - data/test/images/ and data/test/labels/")
    
    # Create symlinks for images if needed
    for split in ['train', 'val', 'test']:
        src_dir = data_dir / split
        images_dir = data_dir / split / 'images'
        labels_dir = data_dir / split / 'labels'
        
        # Create images directory
        images_dir.mkdir(parents=True, exist_ok=True)
        
        # Move images to images folder if they're in the root split folder
        for img_file in src_dir.glob('*.jpg'):
            dest = images_dir / img_file.name
            if not dest.exists():
                shutil.copy(img_file, dest)
                print(f"Copied {img_file.name} to {images_dir}")
        
        for img_file in src_dir.glob('*.png'):
            dest = images_dir / img_file.name
            if not dest.exists():
                shutil.copy(img_file, dest)
                print(f"Copied {img_file.name} to {images_dir}")
    
    print("\nAll data organized successfully!")


def main():
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    print("=" * 50)
    print("Preparing data for YOLO training")
    print("=" * 50)
    
    organize_data_for_yolo(project_root)
    
    print("\n" + "=" * 50)
    print("Data preparation complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
