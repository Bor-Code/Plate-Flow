from typing import Protocol

from plateflow.domain.models import OcrResult, PlateDetection, PlateReading


class Detector(Protocol):
    def detect(self, image_data: bytes) -> list[PlateDetection]: ...


class OcrEngine(Protocol):
    def read_text(self, image_data: bytes) -> OcrResult: ...


class Tracker(Protocol):
    def update(
        self, detections: list[PlateDetection], timestamp_ms: int
    ) -> list[PlateDetection]: ...


class PlateRepository(Protocol):
    def save(self, reading: PlateReading) -> None: ...

    def is_recently_saved(self, text: str, window_ms: int) -> bool: ...
