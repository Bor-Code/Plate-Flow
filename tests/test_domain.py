from plateflow.domain.models import BoundingBox, PlateDetection, OcrResult, PlateReading


def test_bounding_box_properties() -> None:
    box = BoundingBox(x_min=10.0, y_min=20.0, x_max=110.0, y_max=70.0)
    assert box.width == 100.0
    assert box.height == 50.0
    assert box.area == 5000.0


def test_plate_reading_creation() -> None:
    box = BoundingBox(x_min=0.0, y_min=0.0, x_max=10.0, y_max=10.0)
    detection = PlateDetection(box=box, confidence=0.9, image_crop=b"dummy")
    ocr = OcrResult(text="34ABC123", confidence=0.95, raw_text="34 ABC 123")
    
    reading = PlateReading(
        detection=detection,
        ocr_result=ocr,
        is_valid=True,
        timestamp_ms=1000,
        track_id=1
    )
    
    assert reading.is_valid
    assert reading.ocr_result is not None
    assert reading.ocr_result.text == "34ABC123"
    assert reading.track_id == 1
