from plateflow.config.settings import Settings
from plateflow.domain.exceptions import OcrError
from plateflow.domain.protocols import OcrEngine
from plateflow.ocr.easyocr_engine import EasyOcrEngine
from plateflow.ocr.tesseract_engine import TesseractEngine


def create_ocr_engine(settings: Settings) -> OcrEngine:
    engine_name = settings.ocr_engine.lower()

    if engine_name == "easyocr":
        return EasyOcrEngine()
    if engine_name == "tesseract":
        return TesseractEngine()

    raise OcrError(f"Unknown OCR engine: '{engine_name}'. Valid options: easyocr, tesseract")
