import cv2
import numpy as np

MAX_STRENGTH = 0.5


def perspective_transform(img: np.ndarray, params: dict) -> np.ndarray:
    """Apply horizontal perspective skew using params['strength'] (-0.5 - 0.5)"""
    strength = float(params.get("strength", 0.0))
    strength = max(-MAX_STRENGTH, min(MAX_STRENGTH, strength))

    height, width = img.shape[:2]

    current_corners = np.array(
        [
            [0, 0],  # top-left
            [width - 1, 0],  # top-right
            [width - 1, height - 1],  # bottom-right
            [0, height - 1],  # bottom-left
        ],
        dtype=np.float32,
    )

    offset = strength * width

    if strength > 0:
        target_corners = np.array(
            [
                [offset, 0],  # top-left
                [width - offset - 1, 0],  # top-right
                [width - 1, height - 1],  # bottom-right
                [0, height - 1],  # bottom-left
            ],
            dtype=np.float32,
        )
    else:
        offset = abs(offset)
        target_corners = np.array(
            [
                [0, 0],  # top-left
                [width - 1, 0],  # top-right
                [width - offset - 1, height - 1],  # bottom-right
                [offset, height - 1],  # bottom-left
            ],
            dtype=np.float32,
        )

    transform_matrix = cv2.getPerspectiveTransform(current_corners, target_corners)
    result = cv2.warpPerspective(
        img,
        transform_matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    return result
