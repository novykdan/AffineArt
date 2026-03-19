import cv2
import numpy as np


def rotate(img: np.ndarray, params: dict) -> np.ndarray:
    """Rotate image around centr using params['angle'] and params['scale']"""

    angle = float(params.get("angle", 0.0))
    scale = float(params.get("scale", 1.0))

    height, width = img.shape[:2]
    center = (width // 2, height // 2)

    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)

    rotated_img = cv2.warpAffine(
        img,
        rotation_matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )

    return rotated_img


def translate(img: np.ndarray, params: dict) -> np.ndarray:
    """Translate image by integer shift_x and shift_y from params"""
    shift_x = int(params.get("shift_x", 0))
    shift_y = int(params.get("shift_y", 0))

    height, width = img.shape[:2]

    translation_matrix = np.array([[1, 0, shift_x], [0, 1, shift_y]], dtype=np.float32)

    shifted_img = cv2.warpAffine(
        img,
        translation_matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )

    return shifted_img


def scale(img: np.ndarray, params: dict) -> np.ndarray:
    """Resize image by scale_x and scale_y (both must be > 0)"""
    scale_x = float(params.get("scale_x", 1.0))
    scale_y = float(params.get("scale_y", 1.0))

    if scale_x <= 0 or scale_y <= 0:
        raise ValueError("Scale must be > 0")

    height, width = img.shape[:2]
    new_width = int(width * scale_x)
    new_height = int(height * scale_y)

    scaled_img = cv2.resize(
        img, (new_width, new_height), interpolation=cv2.INTER_LINEAR
    )

    return scaled_img
