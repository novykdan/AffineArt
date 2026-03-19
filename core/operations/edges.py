import cv2
import numpy as np


def detect_edges(img: np.ndarray, params: dict) -> np.ndarray:
    """Detect edges using Canny with threshold1 and threshold2"""
    threshold1 = float(params.get("threshold1", 100.0))
    threshold2 = float(params.get("threshold2", 200.0))

    if threshold1 < 0 or threshold2 < 0:
        raise ValueError("Thresholds must be >= 0")

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, threshold1, threshold2)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

    return edges_rgb
