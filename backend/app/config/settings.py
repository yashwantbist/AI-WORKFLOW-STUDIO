"""Application configuration loaded from environment variables / .env file."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central settings object.  Add new fields here as services are implemented."""

    # ── Application ───────────────────────────────────────────────────────────
    app_name: str = "AI Workflow Studio API"
    app_version: str = "0.1.0"
    debug: bool = False

    # ── NVIDIA NIM (future use) ───────────────────────────────────────────────
    nvidia_api_key: str = ""
    nvidia_nim_base_url: str = "https://integrate.api.nvidia.com/v1"

    # ── Auth (future use) ────────────────────────────────────────────────────
    secret_key: str = "changeme-before-production"
    access_token_expire_minutes: int = 30

    # ── Database (future use) ────────────────────────────────────────────────
    database_url: str = "sqlite:///./dev.db"

    model_config = SettingsConfigDict(
        env_file="../.env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
