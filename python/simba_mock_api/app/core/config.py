"""
Configuration settings for the Simba Mock API.
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings

# Determine the base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """Settings for the application."""

    # API settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Simba Mock API for Agent Development"

    # Data directory
    DATA_DIR: str = os.path.join(BASE_DIR, "data")

    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # API Key settings
    API_KEY_LENGTH: int = 32

    # Cache for storing API keys (in a production app, this would be replaced with a database)
    API_KEYS: set = set()

    class Config:
        env_file = os.path.join(BASE_DIR, ".env")
        case_sensitive = True


# Create global settings object
settings = Settings()

# Ensure data directory exists
os.makedirs(settings.DATA_DIR, exist_ok=True)
