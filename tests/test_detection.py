from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from plateflow.config.settings import get_settings
from plateflow.detection.yolo import YoloDetector
from plateflow.domain.exceptions import InvalidImageError


@pytest.fixture
def mock_settings() -> MagicMock:
    settings = get_settings()
    settings.detector_model_path = "dummy.pt"
    return settings


@pytest.fixture
def sample_image_bytes() -> bytes:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    _, encoded = cv2.imencode(".jpg", image)
    return encoded.tobytes()


@patch("plateflow.detection.yolo.YOLO")
def test_yolo_detector_initialization(mock_yolo_class: MagicMock, mock_settings: MagicMock) -> None:
    YoloDetector(settings=mock_settings)
    mock_yolo_class.assert_called_once_with("dummy.pt")


@patch("plateflow.detection.yolo.YOLO")
def test_detect_returns_detections(mock_yolo_class: MagicMock, mock_settings: MagicMock, sample_image_bytes: bytes) -> None:
    mock_model_instance = MagicMock()
    mock_yolo_class.return_value = mock_model_instance
    
    mock_result = MagicMock()
    mock_box = MagicMock()
    mock_box.xyxy = [MagicMock()]
    mock_box.xyxy[0].cpu().numpy.return_value = np.array([10.0, 10.0, 50.0, 50.0])
    mock_box.conf = [MagicMock()]
    mock_box.conf[0].cpu().numpy.return_value = np.array([0.9])
    
    mock_result.boxes = [mock_box]
    mock_model_instance.predict.return_value = [mock_result]
    
    detector = YoloDetector(settings=mock_settings)
    detections = detector.detect(sample_image_bytes)
    
    assert len(detections) == 1
    assert detections[0].confidence == 0.9
    assert detections[0].box.width == 40.0
    assert detections[0].box.height == 40.0
    assert isinstance(detections[0].image_crop, bytes)


@patch("plateflow.detection.yolo.YOLO")
def test_detect_with_invalid_image(mock_yolo_class: MagicMock, mock_settings: MagicMock) -> None:
    detector = YoloDetector(settings=mock_settings)
    
    with pytest.raises(InvalidImageError):
        detector.detect(b"invalid data")


@patch("plateflow.detection.yolo.YOLO")
def test_detect_no_boxes(mock_yolo_class: MagicMock, mock_settings: MagicMock, sample_image_bytes: bytes) -> None:
    mock_model_instance = MagicMock()
    mock_yolo_class.return_value = mock_model_instance
    
    mock_result = MagicMock()
    mock_result.boxes = []
    mock_model_instance.predict.return_value = [mock_result]
    
    detector = YoloDetector(settings=mock_settings)
    detections = detector.detect(sample_image_bytes)
    
    assert len(detections) == 0
