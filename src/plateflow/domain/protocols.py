from typing import Protocol, List, Optional
from plateflow.domain.models import PlateDetection, OcrResult, PlateReading


class Detector(Protocol):
    def detect(self, image_data: bytes) -> List[PlateDetection]: ...


class OcrEngine(Protocol):
    def read_text(self, image_data: bytes) -> OcrResult: ...


class Tracker(Protocol):
    def update(
        self, detections: List[PlateDetection], timestamp_ms: int
    ) -> List[PlateDetection]: ...


class PlateRepository(Protocol):
    def save(self, reading: PlateReading) -> None: ...

    def is_recently_saved(self, text: str, window_ms: int) -> bool: ...
