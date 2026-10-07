import cv2
import numpy as np
from ultralytics import YOLO

from plateflow.config.logging import get_logger
from plateflow.config.settings import Settings
from plateflow.domain.exceptions import InvalidImageError, ModelNotFoundError
from plateflow.domain.models import BoundingBox, PlateDetection

logger = get_logger(__name__)


class YoloDetector:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        try:
            self._model = YOLO(self._settings.detector_model_path)
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise ModelNotFoundError(f"Model not found at {self._settings.detector_model_path}") from e

    def detect(self, image_data: bytes) -> list[PlateDetection]:
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            logger.error("Failed to decode image data")
            raise InvalidImageError("Could not decode image from bytes")

        results = self._model.predict(
            source=image,
            conf=self._settings.detector_confidence_threshold,
            iou=self._settings.detector_nms_threshold,
            verbose=False,
        )

        detections: list[PlateDetection] = []
        
        if not results:
            return detections
            
        result = results[0]
        
        if result.boxes is None or len(result.boxes) == 0:
            return detections

        for box_data in result.boxes:
            xyxy = box_data.xyxy[0].cpu().numpy()
            conf = float(box_data.conf[0].cpu().numpy())
            
            x_min, y_min, x_max, y_max = xyxy
            
            x_min_int, y_min_int = max(0, int(x_min)), max(0, int(y_min))
            x_max_int, y_max_int = min(image.shape[1], int(x_max)), min(image.shape[0], int(y_max))
            
            crop = image[y_min_int:y_max_int, x_min_int:x_max_int]
            _, crop_bytes = cv2.imencode(".jpg", crop)
            
            bounding_box = BoundingBox(
                x_min=float(x_min),
                y_min=float(y_min),
                x_max=float(x_max),
                y_max=float(y_max),
            )
            
            detection = PlateDetection(
                box=bounding_box,
                confidence=conf,
                image_crop=crop_bytes.tobytes(),
            )
            detections.append(detection)

        return detections
