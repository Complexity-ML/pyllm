"""UI Configuration."""

import os

UI_CONFIG = {
    "api_url": os.getenv("PYLLM_API_URL", "http://localhost:8000"),
}


def get_api_url() -> str:
    return UI_CONFIG["api_url"]
