from pathlib import Path
import cv2

def load_image(image_path: str):
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")
    return img
