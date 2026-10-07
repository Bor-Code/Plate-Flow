import logging
from typing import Any

import cv2
import numpy as np

from plateflow.domain.exceptions import OcrError
from plateflow.domain.models import OcrResult

logger = logging.getLogger(__name__)


class EasyOcrEngine:
    def __init__(self, languages: list[str] | None = None) -> None:
        self._languages = languages or ["en"]
        self._reader: Any = None

    def _get_reader(self) -> Any:
        if self._reader is None:
            try:
                import easyocr

                self._reader = easyocr.Reader(self._languages, gpu=False)
            except ImportError as e:
                raise OcrError("easyocr is not installed") from e
            except Exception as e:
                logger.error(f"Failed to initialize EasyOCR reader: {e}")
                raise OcrError(f"EasyOCR initialization failed: {e}") from e
        return self._reader

    def read_text(self, image_data: bytes) -> OcrResult:
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise OcrError("Could not decode image bytes for OCR")

        try:
            reader = self._get_reader()
            results = reader.readtext(image)
        except OcrError:
            raise
        except Exception as e:
            logger.error(f"EasyOCR read failed: {e}")
            raise OcrError(f"EasyOCR read failed: {e}") from e

        if not results:
            return OcrResult(text="", confidence=0.0, raw_text="")

        texts = []
        confidences = []
        for _, text, conf in results:
            texts.append(str(text))
            confidences.append(float(conf))

        raw_text = " ".join(texts)
        avg_confidence = sum(confidences) / len(confidences)

        return OcrResult(
            text=raw_text.upper().replace(" ", ""),
            confidence=avg_confidence,
            raw_text=raw_text,
        )
