import os
from plateflow.config.settings import get_settings
from plateflow.config.logging import get_logger, setup_logging


def test_get_settings() -> None:
    os.environ["PLATEFLOW_OCR_ENGINE"] = "tesseract"
    settings = get_settings()
    assert settings.ocr_engine == "tesseract"
    assert settings.detector_confidence_threshold == 0.5
    del os.environ["PLATEFLOW_OCR_ENGINE"]


def test_logging_setup() -> None:
    setup_logging(level="DEBUG")
    logger = get_logger("test_logger")
    assert logger.name == "test_logger"
