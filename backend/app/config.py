from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[1]

class Settings(BaseSettings):
    app_name: str = "AutoPilot DS"
    database_url: str = "sqlite:///./autopilot.db"
    jwt_secret: str = "CHANGE_THIS_IN_PRODUCTION"
    access_token_minutes: int = 10080
    max_upload_mb: int = 50
    cors_origins: str = "http://localhost:3000"

    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", extra="ignore")

settings = Settings()
