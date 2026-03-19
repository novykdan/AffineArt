from .registry import register_operation
from .schemes import Operation
from .operations.adjust import gamma_correction, brightness_contrast
from .operations.convert import to_grayscale, to_sepia
from .operations.filters import gaussian_blur
from .operations.filters import sharpen
from .operations.edges import detect_edges
from .operations.hsv import color_tune_hsv
from .operations.affine import rotate, translate, scale
from .operations.perspective import perspective_transform


def _identity(img, params: dict):
    """Returns input image unchanged"""
    return img


def register_default_operations() -> None:
    """Register default operations in the global registry"""
    register_operation(
        Operation(
            id="core.identity",
            label="Identity",
            apply=_identity,
            description="Returns the input image unchanged",
        )
    )

    register_operation(
        Operation(
            id="adjust.gamma",
            label="Gamma",
            apply=gamma_correction,
            defaults={"gamma": 1.0},
            description="Gamma correction using LUT",
        )
    )

    register_operation(
        Operation(
            id="adjust.brightness_contrast",
            label="Brightness/Contrast",
            apply=brightness_contrast,
            defaults={"brightness": 0, "contrast": 1.0},
            description="Brightness/Contrast",
        )
    )

    register_operation(
        Operation(
            id="convert.grayscale",
            label="Grayscale",
            apply=to_grayscale,
            description="Grayscale image (returns RGB)",
        )
    )

    register_operation(
        Operation(
            id="filter.gaussian_blur",
            label="Gaussian blur",
            apply=gaussian_blur,
            defaults={"kernel_size": 7, "sigma": 0.0},
            description="Gaussian blur (kernel_size must be odd)",
        )
    )

    register_operation(
        Operation(
            id="filter.sharpen",
            label="Sharpen",
            apply=sharpen,
            defaults={"alpha": 1.0},
            description="Sharpen image using unsharp mask (Gaussian blur + addWeighted)",
        )
    )

    register_operation(
        Operation(
            id="edges.detect",
            label="Edge detection",
            apply=detect_edges,
            defaults={"threshold1": 100.0, "threshold2": 200.0},
            description="Detect edges (Canny algorithm)",
        )
    )

    register_operation(
        Operation(
            id="convert.sepia",
            label="Sepia",
            apply=to_sepia,
            defaults={"intensity": 1.0},
            description="Apply sepia tone (intensity 0 to 1)",
        )
    )

    register_operation(
        Operation(
            id="color.hsv_tune",
            label="HSV Tune",
            apply=color_tune_hsv,
            defaults={"hue_shift": 0.0, "saturation_scale": 1.0, "value_scale": 1.0},
            description="Tunes image's hue, saturation and value",
        )
    )

    register_operation(
        Operation(
            id="affine.rotate",
            label="Rotate",
            apply=rotate,
            defaults={"angle": 0.0, "scale": 1.0},
            description="Rotate image around center",
        )
    )

    register_operation(
        Operation(
            id="affine.translate",
            label="Translate",
            apply=translate,
            defaults={"shift_x": 0.0, "shift_y": 0.0},
            description="Shift image in x/y",
        )
    )

    register_operation(
        Operation(
            id="affine.scale",
            label="Scale",
            apply=scale,
            defaults={"scale_x": 1.0, "scale_y": 1.0},
            description="Scale image in x/y",
        )
    )

    register_operation(
        Operation(
            id="perspective.vertical",
            label="Vertical perspective",
            apply=perspective_transform,
            defaults={"strength": 0.0},
            description="Simple vertical perspective transform",
        )
    )
