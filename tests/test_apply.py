import pytest
import numpy as np
from core.apply import apply_step, apply_steps
from core.registry import clear_registry
from core.catalog import register_default_operations


@pytest.fixture(autouse=True)
def setup_registry():
    clear_registry()
    register_default_operations()


@pytest.fixture
def black_img():
    return np.zeros((10, 10, 3), dtype=np.uint8)


def test_apply_single_step_defaults(black_img):
    res = apply_step(black_img, "adjust.gamma")
    assert np.array_equal(res, black_img)


def test_apply_single_step_with_params(black_img):
    gray_img = black_img + 100
    res = apply_step(gray_img, "adjust.brightness_contrast", {"brightness": 50})
    assert res[0, 0, 0] == 150


def test_apply_pipeline(black_img):
    steps = [
        {"id": "adjust.brightness_contrast", "params": {"brightness": 10}},
        {"id": "adjust.brightness_contrast", "params": {"brightness": 20}},
    ]
    res = apply_steps(black_img, steps)
    assert res[0, 0, 0] == 30


def test_unknown_operation(black_img):
    with pytest.raises(KeyError) as err:
        apply_step(black_img, "non_existent_filter")
    assert "Unknown operation id" in str(err.value)


def test_validation_trigger(black_img):
    with pytest.raises(ValueError) as err:
        apply_step(black_img.astype(np.float32), "core.identity")
    assert "Image dtype must be uint8" in str(err.value)


def test_identity_operation(black_img):
    res = apply_step(black_img, "core.identity")
    assert np.array_equal(res, black_img)
