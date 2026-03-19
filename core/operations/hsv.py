import cv2
import numpy as np


def color_tune_hsv(img: np.ndarray, params: dict) -> np.ndarray:
    """Shift hue and scale saturation/value using HSV params"""
    hue_shift = float(params.get("hue_shift", 0.0))
    saturation_scale = float(params.get("saturation_scale", 1.0))
    value_scale = float(params.get("value_scale", 1.0))

    saturation_scale = max(0.0, saturation_scale)
    value_scale = max(0.0, value_scale)

    if hue_shift == 0.0 and saturation_scale == 1.0 and value_scale == 1.0:
        return img

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)

    h = (h + hue_shift) % 180
    s = np.clip(s * saturation_scale, 0, 255)
    v = np.clip(v * value_scale, 0, 255)

    hsv_tuned = cv2.merge([h, s, v]).astype(np.uint8)
    rgb_tuned = cv2.cvtColor(hsv_tuned, cv2.COLOR_HSV2RGB)

    return rgb_tuned
