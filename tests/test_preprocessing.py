import cv2
import numpy as np
import pytest

from plateflow.preprocessing.pipeline import (
    PreprocessingConfig,
    add_padding,
    apply_adaptive_threshold,
    apply_clahe,
    build_ocr_pipeline,
    decode_bytes,
    deskew,
    encode_to_bytes,
    resize_to_target,
    run_pipeline,
    to_grayscale,
    upscale_if_small,
)


@pytest.fixture
def config() -> PreprocessingConfig:
    return PreprocessingConfig()


@pytest.fixture
def color_image_bytes() -> bytes:
    image = np.ones((80, 200, 3), dtype=np.uint8) * 200
    cv2.putText(image, "34 ABC 123", (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    _, encoded = cv2.imencode(".jpg", image)
    return encoded.tobytes()


@pytest.fixture
def small_image_bytes() -> bytes:
    image = np.ones((20, 100, 3), dtype=np.uint8) * 200
    _, encoded = cv2.imencode(".jpg", image)
    return encoded.tobytes()


@pytest.fixture
def grayscale_image_bytes() -> bytes:
    image = np.ones((60, 200), dtype=np.uint8) * 200
    _, encoded = cv2.imencode(".jpg", image)
    return encoded.tobytes()


def test_decode_bytes_from_valid_image(color_image_bytes: bytes) -> None:
    image = decode_bytes(color_image_bytes)
    assert image is not None
    assert len(image.shape) == 3


def test_decode_bytes_raises_on_invalid_data() -> None:
    with pytest.raises(ValueError):
        decode_bytes(b"not an image")


def test_encode_to_bytes(color_image_bytes: bytes) -> None:
    image = decode_bytes(color_image_bytes)
    result = encode_to_bytes(image)
    assert isinstance(result, bytes)
    assert len(result) > 0


def test_add_padding_increases_size(config: PreprocessingConfig, color_image_bytes: bytes) -> None:
    image = decode_bytes(color_image_bytes)
    original_h, original_w = image.shape[:2]
    step = add_padding(config)
    padded = step(image)
    assert padded.shape[0] == original_h + 2 * config.padding
    assert padded.shape[1] == original_w + 2 * config.padding


def test_to_grayscale_converts_color(color_image_bytes: bytes) -> None:
    image = decode_bytes(color_image_bytes)
    step = to_grayscale()
    result = step(image)
    assert len(result.shape) == 2


def test_to_grayscale_passes_through_gray(grayscale_image_bytes: bytes) -> None:
    image = decode_bytes(grayscale_image_bytes)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    step = to_grayscale()
    result = step(gray)
    assert len(result.shape) == 2


def test_apply_clahe_produces_grayscale_output(config: PreprocessingConfig, color_image_bytes: bytes) -> None:
    image = decode_bytes(color_image_bytes)
    step = apply_clahe(config)
    result = step(image)
    assert len(result.shape) == 2


def test_apply_adaptive_threshold_produces_binary(color_image_bytes: bytes) -> None:
    image = decode_bytes(color_image_bytes)
    step = apply_adaptive_threshold()
    result = step(image)
    assert len(result.shape) == 2
    unique_values = np.unique(result)
    assert set(unique_values).issubset({0, 255})


def test_upscale_if_small_scales_small_image(config: PreprocessingConfig, small_image_bytes: bytes) -> None:
    image = decode_bytes(small_image_bytes)
    assert image.shape[0] < config.upscale_min_height
    step = upscale_if_small(config)
    result = step(image)
    assert result.shape[0] >= config.upscale_min_height


def test_upscale_if_small_keeps_large_image(config: PreprocessingConfig, color_image_bytes: bytes) -> None:
    image = decode_bytes(color_image_bytes)
    original_h = image.shape[0]
    assert original_h >= config.upscale_min_height
    step = upscale_if_small(config)
    result = step(image)
    assert result.shape[0] == original_h


def test_deskew_runs_without_error(config: PreprocessingConfig, color_image_bytes: bytes) -> None:
    image = decode_bytes(color_image_bytes)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    step = deskew(config)
    result = step(gray)
    assert result.shape == gray.shape


def test_resize_to_target_produces_correct_shape(config: PreprocessingConfig, color_image_bytes: bytes) -> None:
    image = decode_bytes(color_image_bytes)
    step = resize_to_target(config)
    result = step(image)
    assert result.shape[1] == config.target_width
    assert result.shape[0] == config.target_height


def test_run_pipeline_end_to_end(config: PreprocessingConfig, color_image_bytes: bytes) -> None:
    steps = build_ocr_pipeline(config)
    result = run_pipeline(color_image_bytes, steps)
    assert isinstance(result, bytes)
    assert len(result) > 0
    decoded = decode_bytes(result)
    assert decoded is not None
