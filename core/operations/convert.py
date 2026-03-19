import cv2
import numpy as np


def to_grayscale(img: np.ndarray, params: dict) -> np.ndarray:
    """Convert image to grayscale and return as 3-channel RGB-like array"""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    return gray_rgb


def to_sepia(img: np.ndarray, params: dict) -> np.ndarray:
    """Apply a sepia by params['intensity'] (0.0-1.0)"""
    intensity = float(params.get("intensity", 1.0))
    intensity = max(0.0, min(1.0, intensity))

    # standard sepia matrix
    sepia_filter = np.array(
        [[0.393, 0.769, 0.189], [0.349, 0.686, 0.168], [0.272, 0.534, 0.131]],
        dtype=np.float32,
    )
    img_float = img.astype(np.float32)
    sepia_img = img_float @ sepia_filter.T

    blended = (1 - intensity) * img_float + intensity * sepia_img
    sepia_img = np.clip(blended, 0, 255).astype(np.uint8)

    return sepia_img
