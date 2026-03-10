#!/usr/bin/env python3
"""
Download and prepare real-world datasets for this project.

Supported sources:
1) coco128 (small, fast)
2) coco2017-val (larger, ~5k images in source split)

Output is written to data/{train,val,test} with COCO-style annotations.json
and labels can then be generated via scripts/convert_to_yolo.py.
"""

import os
import json
import urllib.request
import zipfile
import tempfile
import shutil
import random
import argparse
from pathlib import Path

# COCO category ids in annotations JSON
COCO_TARGET_CATEGORY_IDS = {
    1: "person",
    3: "car",
    18: "dog"
}

# Map to pipeline class ids
PIPELINE_IDS = {
    "person": 0,
    "car": 1,
    "dog": 2
}

def download_and_extract(url, extract_to):
    """Download a zip file and extract it."""
    print(f"Downloading {url}...")
    temp_zip = Path(tempfile.gettempdir()) / "coco128_temp.zip"
    urllib.request.urlretrieve(url, temp_zip)
    
    print(f"Extracting to {extract_to}...")
    with zipfile.ZipFile(temp_zip, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    
    temp_zip.unlink()

def create_split_data_from_yolo_labels(images_dir, labels_dir, output_dir, split_name, img_files):
    """Filter and copy images and convert their annotations."""
    out_images = output_dir / split_name / "images"
    out_labels = output_dir / split_name / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)
    
    annotations = {
        "images": [], 
        "annotations": [], 
        "categories": [
            {"id": PIPELINE_IDS["person"], "name": "person"},
            {"id": PIPELINE_IDS["car"], "name": "car"},
            {"id": PIPELINE_IDS["dog"], "name": "dog"}
        ]
    }
    
    ann_id = 1
    processed_count = 0
    
    for i, img_file in enumerate(img_files):
        img_id = i + 1
        img_path = images_dir / img_file
        label_file = img_file.replace('.jpg', '.txt')
        label_path = labels_dir / label_file
        
        valid_bboxes = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    class_id = int(parts[0])
                    
                    # YOLO COCO class ids in labels: person=0, car=2, dog=16
                    if class_id in {0, 2, 16}:
                        class_name = {0: "person", 2: "car", 16: "dog"}[class_id]
                        new_id = PIPELINE_IDS[class_name]
                        
                        # YOLO format: class x_center y_center width height
                        x_c, y_c, w, h = map(float, parts[1:])
                        
                        # Approximation for coco128 labels -> pseudo COCO pixels
                        img_w, img_h = 640, 640 
                        
                        x_min = (x_c - w / 2) * img_w
                        y_min = (y_c - h / 2) * img_h
                        w_px = w * img_w
                        h_px = h * img_h
                        valid_bboxes.append({
                            "category_id": new_id,
                            # COCO format is [x_min, y_min, width, height]
                            "bbox": [x_min, y_min, w_px, h_px]
                        })
        
        # Only copy images that have our target classes (or at least some images)
        if valid_bboxes or i < 5: # Keep some empty ones
            dest_img = out_images / img_file
            if not dest_img.exists():
                shutil.copy(img_path, dest_img)
            
            annotations["images"].append({
                "id": img_id,
                "file_name": img_file,
                "width": 640,
                "height": 640
            })
            
            for bbox in valid_bboxes:
                box = bbox["bbox"]
                annotations["annotations"].append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": bbox["category_id"],
                    "bbox": box,
                    "area": box[2] * box[3],
                    "iscrowd": 0
                })
                ann_id += 1
                
            processed_count += 1

    ann_path = output_dir / split_name / "annotations.json"
    with open(ann_path, 'w') as f:
        json.dump(annotations, f, indent=2)
        
    print(f"Created real data for {split_name}: {processed_count} images.")
    return processed_count

def create_split_data_from_coco_json(images_dir, coco_json_path, output_dir, split_name, image_names):
    """Create split data from official COCO JSON annotations."""
    out_images = output_dir / split_name / "images"
    out_labels = output_dir / split_name / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    with open(coco_json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    image_by_name = {im["file_name"]: im for im in coco["images"]}
    anns_by_image = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    annotations = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": PIPELINE_IDS["person"], "name": "person"},
            {"id": PIPELINE_IDS["car"], "name": "car"},
            {"id": PIPELINE_IDS["dog"], "name": "dog"},
        ],
    }

    ann_id = 1
    img_id_out = 1
    kept = 0
    for file_name in image_names:
        im = image_by_name.get(file_name)
        if not im:
            continue
        src = images_dir / file_name
        if not src.exists():
            continue

        mapped_anns = []
        for ann in anns_by_image.get(im["id"], []):
            cid = ann["category_id"]
            if cid not in COCO_TARGET_CATEGORY_IDS:
                continue
            cname = COCO_TARGET_CATEGORY_IDS[cid]
            mapped_anns.append({
                "id": ann_id,
                "image_id": img_id_out,
                "category_id": PIPELINE_IDS[cname],
                "bbox": ann["bbox"],
                "area": ann.get("area", ann["bbox"][2] * ann["bbox"][3]),
                "iscrowd": int(ann.get("iscrowd", 0)),
            })
            ann_id += 1

        # Keep only images with at least one target annotation
        if not mapped_anns:
            continue

        shutil.copy(src, out_images / file_name)
        annotations["images"].append({
            "id": img_id_out,
            "file_name": file_name,
            "width": im["width"],
            "height": im["height"],
        })
        annotations["annotations"].extend(mapped_anns)
        img_id_out += 1
        kept += 1

    ann_path = output_dir / split_name / "annotations.json"
    with open(ann_path, "w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=2)

    print(f"Created real data for {split_name}: {kept} images.")
    return kept


def prepare_coco128(project_root):
    url = "https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip"
    temp_dir = Path(tempfile.gettempdir()) / "coco128_extract"
    data_dir = project_root / "data"

    print("=" * 60)
    print("Downloading Real World Sample Data (COCO128)")
    print("=" * 60)

    download_and_extract(url, temp_dir)
    coco_images = temp_dir / "coco128" / "images" / "train2017"
    coco_labels = temp_dir / "coco128" / "labels" / "train2017"
    all_images = sorted([f for f in os.listdir(coco_images) if f.endswith(".jpg")])

    train_imgs = all_images[:80]
    val_imgs = all_images[80:108]
    test_imgs = all_images[108:]

    print("\n[1/3] Processing training data...")
    create_split_data_from_yolo_labels(coco_images, coco_labels, data_dir, "train", train_imgs)
    print("\n[2/3] Processing validation data...")
    create_split_data_from_yolo_labels(coco_images, coco_labels, data_dir, "val", val_imgs)
    print("\n[3/3] Processing test data...")
    create_split_data_from_yolo_labels(coco_images, coco_labels, data_dir, "test", test_imgs)

    shutil.rmtree(temp_dir, ignore_errors=True)


def prepare_coco2017_val(project_root, max_images=0, seed=42):
    """
    Download COCO val2017 + annotations and create filtered splits.
    max_images=0 keeps all matched images.
    """
    images_url = "http://images.cocodataset.org/zips/val2017.zip"
    ann_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    temp_dir = Path(tempfile.gettempdir()) / "coco2017_val_extract"
    data_dir = project_root / "data"

    print("=" * 60)
    print("Downloading Larger Real Dataset (COCO 2017 val)")
    print("=" * 60)

    download_and_extract(images_url, temp_dir)
    download_and_extract(ann_url, temp_dir)

    images_dir = temp_dir / "val2017"
    coco_json_path = temp_dir / "annotations" / "instances_val2017.json"
    with open(coco_json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    anns_by_image = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    matched = []
    for im in coco["images"]:
        anns = anns_by_image.get(im["id"], [])
        if any(a["category_id"] in COCO_TARGET_CATEGORY_IDS for a in anns):
            matched.append(im["file_name"])

    rnd = random.Random(seed)
    rnd.shuffle(matched)
    if max_images and max_images > 0:
        matched = matched[:max_images]

    n = len(matched)
    n_train = int(0.7 * n)
    n_val = int(0.2 * n)
    train_imgs = matched[:n_train]
    val_imgs = matched[n_train:n_train + n_val]
    test_imgs = matched[n_train + n_val:]

    print(f"Matched images with target classes: {n}")
    print(f"Split sizes -> train={len(train_imgs)}, val={len(val_imgs)}, test={len(test_imgs)}")

    print("\n[1/3] Processing training data...")
    create_split_data_from_coco_json(images_dir, coco_json_path, data_dir, "train", train_imgs)
    print("\n[2/3] Processing validation data...")
    create_split_data_from_coco_json(images_dir, coco_json_path, data_dir, "val", val_imgs)
    print("\n[3/3] Processing test data...")
    create_split_data_from_coco_json(images_dir, coco_json_path, data_dir, "test", test_imgs)

    shutil.rmtree(temp_dir, ignore_errors=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Download and prepare real-world datasets")
    parser.add_argument(
        "--dataset",
        default="coco2017-val",
        choices=["coco128", "coco2017-val"],
        help="Dataset source to prepare",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Optional cap on matched images for coco2017-val (0 = all)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed")
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    if args.dataset == "coco128":
        prepare_coco128(project_root)
    else:
        prepare_coco2017_val(project_root, max_images=args.max_images, seed=args.seed)

    print("\n" + "=" * 60)
    print("Real-world data downloaded and organized!")
    print("=" * 60)
    print("\nNext steps:")
    print("  python scripts/convert_to_yolo.py")
    print("  python scripts/full_train.py")

if __name__ == "__main__":
    main()
