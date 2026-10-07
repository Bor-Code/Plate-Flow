from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    detector_model_path: str = "weights/detector.onnx"
    detector_confidence_threshold: float = 0.5
    ocr_engine: str = "easyocr"
    database_url: str = "sqlite:///plateflow.db"
    duplicate_window_ms: int = 5000
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_prefix="PLATEFLOW_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


def get_settings() -> Settings:
    return Settings()
