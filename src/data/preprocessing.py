import cv2

def resize(image, width: int = 640, height: int = 640):
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)

def normalize(image):
    return image.astype("float32") / 255.0
