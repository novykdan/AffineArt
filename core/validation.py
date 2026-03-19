import numpy as np


def validate_image(img: np.ndarray) -> None:
    """Check that image is a uint8 H*W*3 NumPy array"""
    if not isinstance(img, np.ndarray):
        raise ValueError("Input is not a numpy ndarray")
    if img.dtype != np.uint8:
        raise ValueError(f"Image dtype must be uint8, got {img.dtype}")
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Image shape must be H*W*3, got {img.shape}")


def to_uint8_img(img: np.ndarray) -> np.ndarray:
    """Clip values to [0, 255] and convert array to uint8"""
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)
