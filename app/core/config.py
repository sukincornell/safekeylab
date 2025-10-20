"""
Configuration settings for Aegis API
"""

from pydantic_settings import BaseSettings
from typing import List, Optional
import secrets

class Settings(BaseSettings):
    # Application
    APP_NAME: str = "Aegis API"
    VERSION: str = "1.0.0"
    DEBUG: bool = False

    # API Settings
    API_PREFIX: str = "/v1"
    SECRET_KEY: str = secrets.token_urlsafe(32)
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # Security
    ALLOWED_ORIGINS: List[str] = ["https://aegis-shield.ai", "https://api.aegis-shield.ai"]
    ALLOWED_HOSTS: List[str] = ["*"]

    # Database
    DATABASE_URL: str = "postgresql://aegis:password@localhost/aegis"
    REDIS_URL: str = "redis://localhost:6379"

    # Rate Limiting
    RATE_LIMIT_REQUESTS: int = 1000
    RATE_LIMIT_PERIOD: int = 60  # seconds

    # ML Models
    MODEL_CACHE_DIR: str = "/tmp/aegis_models"
    PII_MODEL: str = "microsoft/deberta-v3-base"
    CONFIDENCE_THRESHOLD: float = 0.85

    # Stripe
    STRIPE_SECRET_KEY: Optional[str] = None
    STRIPE_WEBHOOK_SECRET: Optional[str] = None

    # Monitoring
    SENTRY_DSN: Optional[str] = None
    PROMETHEUS_ENABLED: bool = True

    # Email
    SMTP_HOST: str = "smtp.gmail.com"
    SMTP_PORT: int = 587
    SMTP_USER: Optional[str] = None
    SMTP_PASSWORD: Optional[str] = None

    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()