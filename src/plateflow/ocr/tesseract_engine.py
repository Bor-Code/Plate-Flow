import logging

import cv2
import numpy as np

from plateflow.domain.exceptions import OcrError
from plateflow.domain.models import OcrResult

logger = logging.getLogger(__name__)


class TesseractEngine:
    def __init__(self, lang: str = "eng", config: str = "--psm 8 --oem 3") -> None:
        self._lang = lang
        self._config = config

    def read_text(self, image_data: bytes) -> OcrResult:
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise OcrError("Could not decode image bytes for OCR")

        try:
            import pytesseract

            raw_text = pytesseract.image_to_string(image, lang=self._lang, config=self._config).strip()
            data = pytesseract.image_to_data(
                image,
                lang=self._lang,
                config=self._config,
                output_type=pytesseract.Output.DICT,
            )
        except ImportError as e:
            raise OcrError("pytesseract is not installed") from e
        except Exception as e:
            logger.error(f"Tesseract read failed: {e}")
            raise OcrError(f"Tesseract read failed: {e}") from e

        confidences = [
            float(c) / 100.0
            for c in data["conf"]
            if isinstance(c, int) and c >= 0
        ]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        return OcrResult(
            text=raw_text.upper().replace(" ", ""),
            confidence=avg_confidence,
            raw_text=raw_text,
        )
