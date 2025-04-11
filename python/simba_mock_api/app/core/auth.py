"""
Authentication utilities for the Simba Mock API.
"""

import secrets
import string
from app.core.config import settings


def generate_api_key() -> str:
    """
    Generate a new random API key.

    Returns:
        str: A secure random API key.
    """
    # Generate a secure random string
    alphabet = string.ascii_letters + string.digits
    api_key = "".join(secrets.choice(alphabet) for _ in range(settings.API_KEY_LENGTH))

    # Store the key for future validation
    settings.API_KEYS.add(api_key)

    return api_key


def validate_api_key(api_key: str) -> bool:
    """
    Validate if the provided API key is valid.

    Args:
        api_key (str): The API key to validate.

    Returns:
        bool: True if the API key is valid, False otherwise.
    """
    # In a production app, you would check against a secure database
    return api_key in settings.API_KEYS
