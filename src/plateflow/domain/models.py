from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class BoundingBox:
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass(frozen=True)
class PlateDetection:
    box: BoundingBox
    confidence: float
    image_crop: bytes


@dataclass(frozen=True)
class OcrResult:
    text: str
    confidence: float
    raw_text: str


@dataclass(frozen=True)
class PlateReading:
    detection: PlateDetection
    ocr_result: Optional[OcrResult]
    is_valid: bool
    timestamp_ms: int
    track_id: Optional[int] = None
