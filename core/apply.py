from typing import Any
import numpy as np

from .registry import get_operation
from .validation import validate_image, to_uint8_img


def apply_step(img: Any, operation_id: str, params: dict | None = None) -> Any:
    """Apply single operation to image using merged defaults and params"""
    validate_image(img)

    operation = get_operation(operation_id)

    merged = dict(operation.defaults)
    if params:
        merged.update(params)

    out = operation.apply(img, merged)

    if isinstance(out, np.ndarray):
        out = to_uint8_img(out)
        validate_image(out)

    return out


def apply_steps(img: Any, steps: list[dict]) -> Any:
    """Apply a sequence of operations (pipeline) to the image"""
    result = img
    for step in steps:
        operation_id = step["id"]
        params = step.get("params", {})
        result = apply_step(result, operation_id, params)
    return result
