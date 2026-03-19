from typing import Any

import streamlit as st
import numpy as np
import cv2

from core.catalog import register_default_operations
from core.registry import clear_registry, list_operations, get_operation
from core.apply import apply_step


def upload_image(uploaded) -> np.ndarray | None:
    """Read uploaded file and decode to RGB numpy array"""
    if uploaded is None:
        return None
    data = uploaded.read()
    arr = np.frombuffer(data, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


def encode_png(img: np.ndarray) -> bytes:
    """Encode numpy array to png bytes for download"""
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    is_success, buf = cv2.imencode(".png", bgr)
    return buf.tobytes() if is_success else b""


def clamp_kernel_size(kernel_size: int) -> int:
    """Ensure kernel size is positive and odd (OpenCV requirement)"""
    if kernel_size <= 0:
        return 1
    return kernel_size if kernel_size % 2 == 1 else kernel_size + 1


def validate_kernel_size(value):
    """Validate that kernel size is a positive odd int"""
    try:
        kernel_size = int(value)
    except Exception:
        return False, "kernel_size must be an integer"

    if kernel_size < 1:
        return False, "kernel_size must be >= 1"
    if kernel_size % 2 == 0:
        return False, "kernel_size must be odd"

    return True, None


def validate_non_negative(value, name):
    """Ensure value is a >= 0 float"""
    if float(value) < 0:
        return False, f"{name} must be >= 0"
    return True, None


def validate_gamma(value):
    """Ensure gamma is > 0 (avoid division by zero)"""
    if float(value) <= 0:
        return False, "gamma must be > 0"
    return True, None


def validate_unit_interval(value, name):
    """Ensure value is between 0.0 and 1.0"""
    value = float(value)
    if not 0.0 <= value <= 1.0:
        return False, f"{name} must be between 0 and 1"
    return True, None


PARAM_RULES = {
    "kernel_size": validate_kernel_size,
    "sigma": lambda value: validate_non_negative(value, "sigma"),
    "alpha": lambda value: validate_non_negative(value, "alpha"),
    "contrast": lambda value: validate_non_negative(value, "contrast"),
    "gamma": validate_gamma,
    "intensity": lambda value: validate_unit_interval(value, "intensity"),
}

FLOAT_PARAM_RANGES: dict[str, tuple[float, float, float]] = {
    "gamma": (0.1, 3.0, 0.01),
    "intensity": (0.0, 1.0, 0.01),
    "sigma": (0.0, 50.0, 0.1),
    "alpha": (0.0, 5.0, 0.01),
    "contrast": (0.0, 3.0, 0.01),
    "angle": (-180.0, 180.0, 1.0),
}

DEFAULT_FLOAT_RANGE: tuple[float, float, float] = (-1.0, 3.0, 0.01)


def validate_params(params: dict) -> tuple[bool, str]:
    """Validate all parametrs according to rules in PARAM_RULES"""
    for param_name, param_value in params.items():
        validator = PARAM_RULES.get(param_name)
        if validator is None:
            continue

        is_valid, error_message = validator(param_value)

        if not is_valid:
            return False, error_message or f"Invalid value for {param_name}"

    return True, ""


def apply_pipeline(original_image: np.ndarray, steps: list[dict]) -> np.ndarray:
    """Apply all pipeline steps on the image copy (to show original image also)"""
    current_image = original_image.copy()

    for step in steps:
        current_image = apply_step(current_image, step["id"], step.get("params", {}))

    return current_image


def describe_uploaded_file(uploaded_file):
    """Extract file metadata and reset stream position (seek 0)"""
    if uploaded_file is None:
        return None

    name = getattr(uploaded_file, "name", None)
    size = getattr(uploaded_file, "size", None)

    if size is None:
        try:
            data = uploaded_file.read()
            size = len(data)
            # reset file pointer, otherwise openCV will read empty file
            uploaded_file.seek(0)
        except Exception:
            size = None

    return (name, size)


def reset_pipeline_state() -> None:
    """Reset session variables when a new image is loaded"""
    st.session_state.orig = None
    st.session_state["steps"] = []
    st.session_state.pop("_preview", None)
    st.session_state["next_step_uid"] = 0


def find_step_index(step_uid: int) -> int:
    """Find index of the step by its uid"""
    for index, step in enumerate(st.session_state.get("steps", [])):
        if step.get("uid") == step_uid:
            return index
    return -1


def main() -> None:
    """Main Streamlit entry point and UI logic"""
    st.set_page_config(
        page_title="AffineArt",
        page_icon="assets/affineart_logo.png",
        layout="wide",
    )

    clear_registry()
    register_default_operations()

    st.session_state.setdefault("orig", None)
    st.session_state.setdefault("steps", [])
    st.session_state.setdefault("next_step_uid", 0)

    st.markdown("## 🎨 AffineArt - Image Editor")

    uploaded_file = st.file_uploader(
        "Upload image",
        type=["png", "jpg", "jpeg"],
        key="uploader",
    )

    current_signature = describe_uploaded_file(uploaded_file)
    previous_signature = st.session_state.get("_uploaded_sig")

    # reset pipeline if uploaded new file
    if current_signature != previous_signature:
        if current_signature is None:
            reset_pipeline_state()
        else:
            image = upload_image(uploaded_file)
            if image is None:
                st.error("Could not read image")
                return

            reset_pipeline_state()
            st.session_state.orig = image

        st.session_state["_uploaded_sig"] = current_signature

    original_image = st.session_state.orig
    if original_image is None:
        st.info("Upload an image to start editing")
        return

    left_column, center_column, right_column = st.columns([0.9, 1, 1])

    with left_column:
        st.header("Effects")

        undo_column, _, clear_column = st.columns([1, 0.3, 1])

        with undo_column:
            undo_clicked = st.button("Undo", use_container_width=True)

        with clear_column:
            clear_clicked = st.button("Clear", use_container_width=True)

        if undo_clicked and st.session_state["steps"]:
            st.session_state["steps"].pop()
            st.session_state.pop("_preview", None)

        if clear_clicked:
            st.session_state["steps"] = []
            st.session_state.pop("_preview", None)

        st.divider()
        st.markdown(
            "🤓Tip: use **Preview** to try an effect without adding it permanently"
        )

        available_operations = list_operations()
        preview_steps = list(st.session_state["steps"])

        for operation in available_operations:
            with st.expander(operation.label, expanded=False):
                st.write(operation.description or "")

                default_params = dict(operation.defaults)
                param_values: dict[str, Any] = {}

                for param_name, default_value in default_params.items():
                    # unique key (streamlit requirement)
                    widget_key = f"ctrl_{operation.id}_{param_name}"

                    if isinstance(default_value, float):
                        min_value, max_value, step = FLOAT_PARAM_RANGES.get(
                            param_name,
                            DEFAULT_FLOAT_RANGE,
                        )
                        param_values[param_name] = st.slider(
                            label=param_name,
                            min_value=min_value,
                            max_value=max_value,
                            value=float(default_value),
                            step=step,
                            key=widget_key,
                            help=f"Parameter {param_name}",
                        )

                    elif isinstance(default_value, int):
                        if param_name == "kernel_size":
                            raw_value = st.number_input(
                                param_name,
                                min_value=1,
                                max_value=999,
                                value=int(default_value),
                                step=2,  # force odd kernel size
                                key=widget_key,
                            )
                            param_values[param_name] = clamp_kernel_size(int(raw_value))
                        else:
                            param_values[param_name] = st.number_input(
                                param_name,
                                min_value=0,
                                max_value=999,
                                value=int(default_value),
                                step=1,
                                key=widget_key,
                            )
                    else:
                        param_values[param_name] = st.text_input(
                            param_name,
                            value=str(default_value),
                            key=widget_key,
                        )

                add_column, preview_column = st.columns([0.6, 0.4])

                if add_column.button("Add", key=f"add_{operation.id}"):
                    is_valid, error_message = validate_params(param_values)
                    if not is_valid:
                        st.error(error_message)
                    else:
                        uid = st.session_state["next_step_uid"]
                        st.session_state["next_step_uid"] = uid + 1
                        st.session_state["steps"].append(
                            {
                                "uid": uid,
                                "id": operation.id,
                                "params": param_values,
                            }
                        )
                        st.session_state.pop("_preview", None)

                if preview_column.button("Preview", key=f"preview_{operation.id}"):

                    preview_steps.append({"id": operation.id, "params": param_values})
                    st.session_state["_preview"] = apply_pipeline(
                        original_image,
                        preview_steps,
                    )

    with center_column:
        st.header("Original")
        st.image(original_image, width=420)

    with right_column:
        st.header("Result")

        steps = st.session_state["steps"]
        preview_image = st.session_state.get("_preview")

        if steps:
            final_image = apply_pipeline(original_image, steps)
        else:
            final_image = original_image

        displayed_image = preview_image if preview_image is not None else final_image

        st.image(displayed_image, width=420)

        if preview_image is not None:
            st.info(
                "🟢Preview is active. "
                "Discard preview to get back to the last applied result"
            )

            if st.button(
                "Discard preview", key="clear_preview", use_container_width=True
            ):
                st.session_state.pop("_preview", None)
                st.rerun()

        st.download_button(
            "Download PNG",
            data=encode_png(final_image),
            file_name="result.png",
            mime="image/png",
            use_container_width=True,
        )

        st.divider()
        st.subheader("Applied effects")

        if not steps:
            st.write("No effects applied yet.")
        else:

            for step in list(steps):
                step_uid = step.get("uid")
                operation = get_operation(step["id"])

                name_column, actions_column = st.columns([0.75, 0.25])

                with name_column:
                    st.markdown(f"**{operation.label}**")
                    params = step.get("params", {})
                    if params:
                        for param_name, param_value in params.items():
                            st.markdown(f"- **{param_name}**: {param_value}")
                    else:
                        st.markdown("_no parameters_")

                step_index = find_step_index(step_uid)

                with actions_column:
                    move_up_clicked = st.button("⬆️", key=f"up_{step_uid}")
                    remove_clicked = st.button("🗑️", key=f"remove_{step_uid}")

                    if move_up_clicked and step_index > 0:
                        steps_copy = list(st.session_state["steps"])
                        steps_copy[step_index - 1], steps_copy[step_index] = (
                            steps_copy[step_index],
                            steps_copy[step_index - 1],
                        )
                        st.session_state["steps"] = steps_copy
                        st.session_state.pop("_preview", None)
                        st.rerun()

                    if remove_clicked and step_index >= 0:
                        steps_copy = list(st.session_state["steps"])
                        steps_copy.pop(step_index)
                        st.session_state["steps"] = steps_copy
                        st.session_state.pop("_preview", None)
                        st.rerun()
                        break

                st.divider()


if __name__ == "__main__":
    main()
