import cv2
import numpy as np


"""Image adjustment operations"""


def gamma_correction(img: np.ndarray, params: dict) -> np.ndarray:
    """Gamma-correct the image. Use "params['gamma']" (>0) to control mid-tones"""
    gamma = float(params.get("gamma", 1.0))
    if gamma <= 0:
        raise ValueError("Gamma value must be greater than 0")

    inv_gamma = 1.0 / gamma
    # precompute gamma-corrected values for all possible pixels
    lookup_table = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in range(256)],
        dtype=np.uint8,
    )
    return cv2.LUT(img, lookup_table)


def brightness_contrast(img: np.ndarray, params: dict) -> np.ndarray:
    """Adjust image brightness (beta) and contrast (alpha) from params"""
    brightness = int(params.get("brightness", 0))
    contrast = float(params.get("contrast", 1.0))

    if contrast < 0:
        raise ValueError("Contrast must be >= 0")

    adjusted = cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)
    return adjusted
