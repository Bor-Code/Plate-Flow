import logging
import time

import cv2
import numpy as np

from plateflow.config.settings import Settings
from plateflow.domain.exceptions import EmptyDetectionError, InvalidImageError
from plateflow.domain.models import OcrResult, PlateDetection, PlateReading
from plateflow.domain.protocols import Detector, OcrEngine
from plateflow.preprocessing.pipeline import PreprocessingConfig, build_ocr_pipeline, run_pipeline
from plateflow.validation.validator import validate_plate

logger = logging.getLogger(__name__)


class ImagePipeline:
    def __init__(
        self,
        detector: Detector,
        ocr_engine: OcrEngine,
        settings: Settings,
    ) -> None:
        self._detector = detector
        self._ocr_engine = ocr_engine
        self._settings = settings
        self._preprocessing_config = PreprocessingConfig()
        self._preprocessing_steps = build_ocr_pipeline(self._preprocessing_config)

    def process(self, image_data: bytes) -> list[PlateReading]:
        timestamp_ms = int(time.time() * 1000)

        detections = self._detector.detect(image_data)

        if not detections:
            logger.debug("No plates detected in image")
            raise EmptyDetectionError("No plates detected in the provided image")

        readings: list[PlateReading] = []

        for detection in detections:
            reading = self._process_single_detection(detection, timestamp_ms)
            readings.append(reading)

        return readings

    def _process_single_detection(
        self,
        detection: PlateDetection,
        timestamp_ms: int,
    ) -> PlateReading:
        ocr_result: OcrResult | None = None
        is_valid = False

        try:
            preprocessed = run_pipeline(detection.image_crop, self._preprocessing_steps)
            ocr_result = self._ocr_engine.read_text(preprocessed)

            validation_result = validate_plate(ocr_result.text)
            is_valid = validation_result.is_valid

            if validation_result.confidence_penalty > 0:
                adjusted_conf = ocr_result.confidence - validation_result.confidence_penalty
                ocr_result = OcrResult(
                    text=validation_result.normalized_text,
                    confidence=max(0.0, adjusted_conf),
                    raw_text=ocr_result.raw_text,
                )
            elif is_valid:
                ocr_result = OcrResult(
                    text=validation_result.normalized_text,
                    confidence=ocr_result.confidence,
                    raw_text=ocr_result.raw_text,
                )

        except InvalidImageError as e:
            logger.warning(f"Invalid crop for detection: {e}")
        except Exception as e:
            logger.warning(f"OCR failed for detection: {e}")

        return PlateReading(
            detection=detection,
            ocr_result=ocr_result,
            is_valid=is_valid,
            timestamp_ms=timestamp_ms,
        )


def annotate_image(image_data: bytes, readings: list[PlateReading]) -> bytes:
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise InvalidImageError("Could not decode image for annotation")

    for reading in readings:
        box = reading.detection.box
        x1, y1, x2, y2 = int(box.x_min), int(box.y_min), int(box.x_max), int(box.y_max)

        color = (0, 200, 0) if reading.is_valid else (0, 0, 200)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        label = ""
        if reading.ocr_result:
            label = reading.ocr_result.text
            conf = reading.ocr_result.confidence
            label = f"{label} ({conf:.2f})"

        if label:
            cv2.putText(
                image,
                label,
                (x1, max(y1 - 8, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

    _, encoded = cv2.imencode(".jpg", image)
    return encoded.tobytes()
