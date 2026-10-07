from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np


@dataclass(frozen=True)
class PreprocessingConfig:
    target_height: int = 64
    target_width: int = 256
    padding: int = 4
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: int = 8
    deskew_max_angle: float = 15.0
    upscale_min_height: int = 32


PreprocessingStep = Callable[[np.ndarray], np.ndarray]


def decode_bytes(image_data: bytes) -> np.ndarray:
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image bytes")
    return image


def encode_to_bytes(image: np.ndarray) -> bytes:
    _, encoded = cv2.imencode(".jpg", image)
    return encoded.tobytes()


def add_padding(config: PreprocessingConfig) -> PreprocessingStep:
    def step(image: np.ndarray) -> np.ndarray:
        return cv2.copyMakeBorder(
            image,
            config.padding,
            config.padding,
            config.padding,
            config.padding,
            cv2.BORDER_CONSTANT,
            value=[255, 255, 255],
        )

    return step


def to_grayscale() -> PreprocessingStep:
    def step(image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    return step


def apply_clahe(config: PreprocessingConfig) -> PreprocessingStep:
    clahe = cv2.createCLAHE(
        clipLimit=config.clahe_clip_limit,
        tileGridSize=(config.clahe_tile_grid_size, config.clahe_tile_grid_size),
    )

    def step(image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return clahe.apply(image)

    return step


def apply_adaptive_threshold() -> PreprocessingStep:
    def step(image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2,
        )

    return step


def upscale_if_small(config: PreprocessingConfig) -> PreprocessingStep:
    def step(image: np.ndarray) -> np.ndarray:
        h = image.shape[0]
        if h < config.upscale_min_height:
            scale = config.upscale_min_height / h
            new_w = int(image.shape[1] * scale)
            image = cv2.resize(image, (new_w, config.upscale_min_height), interpolation=cv2.INTER_CUBIC)
        return image

    return step


def deskew(config: PreprocessingConfig) -> PreprocessingStep:
    def step(image: np.ndarray) -> np.ndarray:
        gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        coords = np.column_stack(np.where(gray < 128))
        if coords.shape[0] < 5:
            return image
        angle = cv2.minAreaRect(coords.astype(np.float32))[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        if abs(angle) > config.deskew_max_angle:
            return image
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(
            image,
            rotation_matrix,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

    return step


def resize_to_target(config: PreprocessingConfig) -> PreprocessingStep:
    def step(image: np.ndarray) -> np.ndarray:
        return cv2.resize(
            image,
            (config.target_width, config.target_height),
            interpolation=cv2.INTER_CUBIC,
        )

    return step


def build_ocr_pipeline(config: PreprocessingConfig) -> list[PreprocessingStep]:
    return [
        add_padding(config),
        upscale_if_small(config),
        apply_clahe(config),
        apply_adaptive_threshold(),
        deskew(config),
        resize_to_target(config),
    ]


def run_pipeline(image_data: bytes, steps: list[PreprocessingStep]) -> bytes:
    image = decode_bytes(image_data)
    for step in steps:
        image = step(image)
    return encode_to_bytes(image)
