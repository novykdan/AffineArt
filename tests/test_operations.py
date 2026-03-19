import pytest
import numpy as np
import cv2

from core.operations.adjust import gamma_correction, brightness_contrast
from core.operations.affine import rotate, translate, scale
from core.operations.convert import to_grayscale, to_sepia
from core.operations.edges import detect_edges
from core.operations.filters import gaussian_blur, sharpen
from core.operations.hsv import color_tune_hsv
from core.operations.perspective import perspective_transform


@pytest.fixture
def sample_img():
    """Create a sample image for testing"""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(img, (25, 25), (75, 75), (255, 128, 64), -1)
    return img


def test_gamma(sample_img):
    res = gamma_correction(sample_img, {"gamma": 2.0})
    assert res.shape == sample_img.shape
    with pytest.raises(ValueError):
        gamma_correction(sample_img, {"gamma": 0})


def test_brightness_contrast(sample_img):
    res = brightness_contrast(sample_img, {"contrast": 2.0, "brightness": 50})
    assert res.dtype == np.uint8
    with pytest.raises(ValueError):
        brightness_contrast(sample_img, {"contrast": -0.5})


def test_rotate(sample_img):
    res = rotate(sample_img, {"angle": 90, "scale": 1.0})
    assert res.shape == sample_img.shape
    res_zero = rotate(sample_img, {"angle": 0, "scale": 0})
    assert np.max(res_zero) == 0


def test_translate(sample_img):
    res = translate(sample_img, {"shift_x": 10, "shift_y": 10})
    assert res.shape == sample_img.shape
    res_big = translate(sample_img, {"shift_x": 500, "shift_y": 500})
    assert res_big.shape == sample_img.shape


def test_scale(sample_img):
    res = scale(sample_img, {"scale_x": 1.5, "scale_y": 1.5})
    assert res.shape == (150, 150, 3)
    with pytest.raises(ValueError):
        scale(sample_img, {"scale_x": 0, "scale_y": 1.0})


def test_to_grayscale(sample_img):
    res = to_grayscale(sample_img, {})
    assert res.shape == (100, 100, 3)
    assert res[50, 50, 0] == res[50, 50, 1] == res[50, 50, 2]


def test_to_sepia(sample_img):
    res = to_sepia(sample_img, {"intensity": 1.0})
    assert res.shape == sample_img.shape
    res_clamped = to_sepia(sample_img, {"intensity": 5.0})
    assert res_clamped.dtype == np.uint8


def test_detect_edges(sample_img):
    res = detect_edges(sample_img, {"threshold1": 100, "threshold2": 200})
    assert len(res.shape) == 3
    with pytest.raises(ValueError):
        detect_edges(sample_img, {"threshold1": -1, "threshold2": 100})


def test_gaussian_blur(sample_img):
    res = gaussian_blur(sample_img, {"kernel_size": 5, "sigma": 1.0})
    assert res.shape == sample_img.shape
    with pytest.raises(ValueError):
        gaussian_blur(sample_img, {"kernel_size": 4})


def test_sharpen(sample_img):
    res = sharpen(sample_img, {"alpha": 1.5})
    assert res.shape == sample_img.shape
    res_zero = sharpen(sample_img, {"alpha": 0.0})
    assert res_zero.shape == sample_img.shape


def test_color_tune_hsv(sample_img):
    res = color_tune_hsv(
        sample_img, {"hue_shift": 10, "saturation_scale": 1.2, "value_scale": 0.9}
    )
    assert res.shape == sample_img.shape
    res_orig = color_tune_hsv(
        sample_img, {"hue_shift": 0, "saturation_scale": 1.0, "value_scale": 1.0}
    )
    assert np.array_equal(res_orig, sample_img)


def test_perspective(sample_img):
    res = perspective_transform(sample_img, {"strength": 0.2})
    assert res.shape == sample_img.shape
    res_max = perspective_transform(sample_img, {"strength": 0.9})
    assert res_max.shape == sample_img.shape
