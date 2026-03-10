import cv2
import numpy as np

def horizontal_flip(image, p: float = 0.5):
    if np.random.rand() < p:
        return cv2.flip(image, 1)
    return image
