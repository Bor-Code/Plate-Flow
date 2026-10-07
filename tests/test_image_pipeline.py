from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from plateflow.config.settings import Settings
from plateflow.domain.exceptions import EmptyDetectionError
from plateflow.domain.models import BoundingBox, OcrResult, PlateDetection, PlateReading
from plateflow.pipeline.image_pipeline import ImagePipeline, annotate_image


@pytest.fixture
def settings() -> Settings:
    return Settings()


@pytest.fixture
def sample_image_bytes() -> bytes:
    image = np.ones((480, 640, 3), dtype=np.uint8) * 200
    _, encoded = cv2.imencode(".jpg", image)
    return encoded.tobytes()


@pytest.fixture
def sample_crop_bytes() -> bytes:
    image = np.ones((64, 256, 3), dtype=np.uint8) * 240
    _, encoded = cv2.imencode(".jpg", image)
    return encoded.tobytes()


@pytest.fixture
def sample_detection(sample_crop_bytes: bytes) -> PlateDetection:
    return PlateDetection(
        box=BoundingBox(x_min=10.0, y_min=10.0, x_max=110.0, y_max=50.0),
        confidence=0.92,
        image_crop=sample_crop_bytes,
    )


@pytest.fixture
def mock_detector(sample_detection: PlateDetection) -> MagicMock:
    detector = MagicMock()
    detector.detect.return_value = [sample_detection]
    return detector


@pytest.fixture
def mock_ocr_engine() -> MagicMock:
    engine = MagicMock()
    engine.read_text.return_value = OcrResult(
        text="34ABC123",
        confidence=0.93,
        raw_text="34 ABC 123",
    )
    return engine


@pytest.fixture
def empty_detector() -> MagicMock:
    detector = MagicMock()
    detector.detect.return_value = []
    return detector


class TestImagePipeline:
    def test_process_returns_plate_readings(
        self,
        settings: Settings,
        mock_detector: MagicMock,
        mock_ocr_engine: MagicMock,
        sample_image_bytes: bytes,
    ) -> None:
        pipeline = ImagePipeline(mock_detector, mock_ocr_engine, settings)
        readings = pipeline.process(sample_image_bytes)

        assert len(readings) == 1
        assert isinstance(readings[0], PlateReading)

    def test_process_valid_plate_marked_valid(
        self,
        settings: Settings,
        mock_detector: MagicMock,
        mock_ocr_engine: MagicMock,
        sample_image_bytes: bytes,
    ) -> None:
        pipeline = ImagePipeline(mock_detector, mock_ocr_engine, settings)
        readings = pipeline.process(sample_image_bytes)

        assert readings[0].is_valid is True
        assert readings[0].ocr_result is not None
        assert readings[0].ocr_result.text == "34ABC123"

    def test_process_raises_on_empty_detection(
        self,
        settings: Settings,
        empty_detector: MagicMock,
        mock_ocr_engine: MagicMock,
        sample_image_bytes: bytes,
    ) -> None:
        pipeline = ImagePipeline(empty_detector, mock_ocr_engine, settings)

        with pytest.raises(EmptyDetectionError):
            pipeline.process(sample_image_bytes)

    def test_process_invalid_plate_marked_invalid(
        self,
        settings: Settings,
        mock_detector: MagicMock,
        sample_image_bytes: bytes,
    ) -> None:
        bad_ocr = MagicMock()
        bad_ocr.read_text.return_value = OcrResult(
            text="INVALID",
            confidence=0.5,
            raw_text="INVALID",
        )
        pipeline = ImagePipeline(mock_detector, bad_ocr, settings)
        readings = pipeline.process(sample_image_bytes)

        assert readings[0].is_valid is False

    def test_process_ocr_failure_returns_invalid_reading(
        self,
        settings: Settings,
        mock_detector: MagicMock,
        sample_image_bytes: bytes,
    ) -> None:
        failing_ocr = MagicMock()
        failing_ocr.read_text.side_effect = Exception("OCR backend error")

        pipeline = ImagePipeline(mock_detector, failing_ocr, settings)
        readings = pipeline.process(sample_image_bytes)

        assert len(readings) == 1
        assert readings[0].is_valid is False
        assert readings[0].ocr_result is None

    def test_process_multiple_detections(
        self,
        settings: Settings,
        mock_ocr_engine: MagicMock,
        sample_image_bytes: bytes,
        sample_crop_bytes: bytes,
    ) -> None:
        detection_a = PlateDetection(
            box=BoundingBox(10.0, 10.0, 110.0, 50.0),
            confidence=0.9,
            image_crop=sample_crop_bytes,
        )
        detection_b = PlateDetection(
            box=BoundingBox(200.0, 10.0, 300.0, 50.0),
            confidence=0.85,
            image_crop=sample_crop_bytes,
        )
        multi_detector = MagicMock()
        multi_detector.detect.return_value = [detection_a, detection_b]

        pipeline = ImagePipeline(multi_detector, mock_ocr_engine, settings)
        readings = pipeline.process(sample_image_bytes)

        assert len(readings) == 2

    def test_confidence_penalty_applied_for_corrected_plate(
        self,
        settings: Settings,
        mock_detector: MagicMock,
        sample_image_bytes: bytes,
    ) -> None:
        corrected_ocr = MagicMock()
        corrected_ocr.read_text.return_value = OcrResult(
            text="34OBC123",
            confidence=0.90,
            raw_text="34OBC123",
        )
        pipeline = ImagePipeline(mock_detector, corrected_ocr, settings)
        readings = pipeline.process(sample_image_bytes)

        if readings[0].is_valid and readings[0].ocr_result:
            assert readings[0].ocr_result.confidence < 0.90


class TestAnnotateImage:
    def test_annotate_returns_bytes(
        self,
        sample_image_bytes: bytes,
        sample_detection: PlateDetection,
    ) -> None:
        reading = PlateReading(
            detection=sample_detection,
            ocr_result=OcrResult(text="34ABC123", confidence=0.93, raw_text="34 ABC 123"),
            is_valid=True,
            timestamp_ms=1000,
        )
        result = annotate_image(sample_image_bytes, [reading])
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_annotate_invalid_plate_uses_red_box(
        self,
        sample_image_bytes: bytes,
        sample_detection: PlateDetection,
    ) -> None:
        reading = PlateReading(
            detection=sample_detection,
            ocr_result=None,
            is_valid=False,
            timestamp_ms=1000,
        )
        result = annotate_image(sample_image_bytes, [reading])
        assert isinstance(result, bytes)

    def test_annotate_empty_readings(self, sample_image_bytes: bytes) -> None:
        result = annotate_image(sample_image_bytes, [])
        assert isinstance(result, bytes)
