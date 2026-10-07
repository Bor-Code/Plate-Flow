from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from plateflow.config.settings import Settings
from plateflow.domain.exceptions import OcrError
from plateflow.ocr.easyocr_engine import EasyOcrEngine
from plateflow.ocr.factory import create_ocr_engine
from plateflow.ocr.tesseract_engine import TesseractEngine


@pytest.fixture
def sample_image_bytes() -> bytes:
    image = np.ones((64, 256), dtype=np.uint8) * 255
    cv2.putText(image, "34ABC123", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0, 2)
    _, encoded = cv2.imencode(".jpg", image)
    return encoded.tobytes()


@pytest.fixture
def invalid_image_bytes() -> bytes:
    return b"not an image"


class TestEasyOcrEngine:
    @patch("plateflow.ocr.easyocr_engine.easyocr")
    def test_read_text_returns_ocr_result(self, mock_easyocr: MagicMock, sample_image_bytes: bytes) -> None:
        mock_reader = MagicMock()
        mock_reader.readtext.return_value = [
            (None, "34 ABC 123", 0.95),
        ]
        mock_easyocr.Reader.return_value = mock_reader

        engine = EasyOcrEngine()
        engine._reader = mock_reader

        result = engine.read_text(sample_image_bytes)

        assert result.text == "34ABC123"
        assert result.confidence == pytest.approx(0.95)
        assert result.raw_text == "34 ABC 123"

    @patch("plateflow.ocr.easyocr_engine.easyocr")
    def test_read_text_empty_result(self, mock_easyocr: MagicMock, sample_image_bytes: bytes) -> None:
        mock_reader = MagicMock()
        mock_reader.readtext.return_value = []
        engine = EasyOcrEngine()
        engine._reader = mock_reader

        result = engine.read_text(sample_image_bytes)

        assert result.text == ""
        assert result.confidence == 0.0

    def test_read_text_raises_on_invalid_image(self, invalid_image_bytes: bytes) -> None:
        engine = EasyOcrEngine()
        engine._reader = MagicMock()

        with pytest.raises(OcrError):
            engine.read_text(invalid_image_bytes)

    @patch("plateflow.ocr.easyocr_engine.easyocr")
    def test_read_text_multiple_detections(self, mock_easyocr: MagicMock, sample_image_bytes: bytes) -> None:
        mock_reader = MagicMock()
        mock_reader.readtext.return_value = [
            (None, "34", 0.90),
            (None, "ABC 123", 0.80),
        ]
        engine = EasyOcrEngine()
        engine._reader = mock_reader

        result = engine.read_text(sample_image_bytes)

        assert result.text == "34ABC123"
        assert result.confidence == pytest.approx(0.85)


class TestTesseractEngine:
    @patch("plateflow.ocr.tesseract_engine.pytesseract")
    def test_read_text_returns_ocr_result(self, mock_pytesseract: MagicMock, sample_image_bytes: bytes) -> None:
        mock_pytesseract.image_to_string.return_value = "34 ABC 123"
        mock_pytesseract.image_to_data.return_value = {"conf": [90, 85, 95]}
        mock_pytesseract.Output.DICT = "dict"

        engine = TesseractEngine()
        result = engine.read_text(sample_image_bytes)

        assert result.text == "34ABC123"
        assert result.confidence == pytest.approx(0.9)
        assert result.raw_text == "34 ABC 123"

    @patch("plateflow.ocr.tesseract_engine.pytesseract")
    def test_read_text_filters_negative_confidence(self, mock_pytesseract: MagicMock, sample_image_bytes: bytes) -> None:
        mock_pytesseract.image_to_string.return_value = "34ABC"
        mock_pytesseract.image_to_data.return_value = {"conf": [90, -1, 80]}
        mock_pytesseract.Output.DICT = "dict"

        engine = TesseractEngine()
        result = engine.read_text(sample_image_bytes)

        assert result.confidence == pytest.approx(0.85)

    def test_read_text_raises_on_invalid_image(self, invalid_image_bytes: bytes) -> None:
        engine = TesseractEngine()
        with pytest.raises(OcrError):
            engine.read_text(invalid_image_bytes)


class TestOcrFactory:
    def test_factory_returns_easyocr(self) -> None:
        settings = Settings(ocr_engine="easyocr")
        engine = create_ocr_engine(settings)
        assert isinstance(engine, EasyOcrEngine)

    def test_factory_returns_tesseract(self) -> None:
        settings = Settings(ocr_engine="tesseract")
        engine = create_ocr_engine(settings)
        assert isinstance(engine, TesseractEngine)

    def test_factory_raises_on_unknown_engine(self) -> None:
        settings = Settings(ocr_engine="unknown")
        with pytest.raises(OcrError):
            create_ocr_engine(settings)
