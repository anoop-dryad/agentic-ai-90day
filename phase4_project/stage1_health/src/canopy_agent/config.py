"""Central configuration. Reads from environment, with sane defaults."""

import os


class Settings:
    BACKEND_BASE_URL: str = os.getenv("BACKEND_BASE_URL", "http://localhost:8080/v1")
    DEVICE_PATH: str = os.getenv("DEVICE_PATH", "devices")
    BACKEND_TIMEOUT_SECONDS: float = float(os.getenv("BACKEND_TIMEOUT_SECONDS", "5"))
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    MODEL_NAME: str = os.getenv("MODEL_NAME", "gemini-3.1-flash-lite")
    STALE_THRESHOLD_MINUTES: float = float(os.getenv("STALE_THRESHOLD_MINUTES", "15"))
    LOW_BATTERY_PCT: int = int(os.getenv("LOW_BATTERY_PCT", "20"))


settings = Settings()
