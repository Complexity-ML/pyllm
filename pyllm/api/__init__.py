"""API module - FastAPI routes for LLM and image inference."""

from pyllm.api.routes import create_app
from pyllm.api.image_routes import create_image_router

__all__ = ["create_app", "create_image_router"]
