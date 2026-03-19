import cv2
import numpy as np


def gaussian_blur(img: np.ndarray, params: dict) -> np.ndarray:
    """Gaussian blur with odd kernel_size and optional sigma (standard deviation)"""
    kernel_size = int(params.get("kernel_size", 7))
    sigma = float(params.get("sigma", 0.0))

    if kernel_size < 1:
        raise ValueError("Kernel size must be >= 1")
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be odd")

    if sigma < 0:
        raise ValueError("Sigma must be >= 0")

    blurred = cv2.GaussianBlur(img, ksize=(kernel_size, kernel_size), sigmaX=sigma)
    return blurred


def sharpen(img: np.ndarray, params: dict) -> np.ndarray:
    """Sharpen image using unsharp-mask style using alpha"""
    alpha = float(params.get("alpha", 1.0))
    if alpha < 0:
        raise ValueError("Alpha must be >= 0")

    blurred = cv2.GaussianBlur(img, (5, 5), sigmaX=0)
    sharpened = cv2.addWeighted(img, 1 + alpha, blurred, -alpha, 0)
    return sharpened
