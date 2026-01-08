"""FastAPI routes for image generation."""

import logging
from datetime import datetime
from typing import Optional, List

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from pyllm.inference.diffusion import DiffusionEngine, ImageGenerationConfig

logger = logging.getLogger("pyllm.api.image")


# Request/Response models

class ImageGenerateRequest(BaseModel):
    """Image generation request."""
    prompt: str
    negative_prompt: str = ""
    width: int = 256
    height: int = 256
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    seed: Optional[int] = None
    num_images: int = 1


class ImageResponse(BaseModel):
    """Generated image response."""
    b64_json: str
    seed: int


class ImagesResponse(BaseModel):
    """Multiple images response."""
    created: int
    data: List[ImageResponse]


def create_image_router(
    vae_path: Optional[str] = None,
    dit_path: Optional[str] = None,
) -> APIRouter:
    """Create image generation router."""

    router = APIRouter(prefix="/v1/images", tags=["images"])

    # Diffusion engine (lazy loading)
    diffusion_engine: Optional[DiffusionEngine] = None

    def get_diffusion_engine() -> DiffusionEngine:
        nonlocal diffusion_engine
        if diffusion_engine is None:
            diffusion_engine = DiffusionEngine()
            if vae_path and dit_path:
                diffusion_engine.load(vae_path, dit_path)
        return diffusion_engine

    @router.post("/load")
    async def load_diffusion_model(vae_path: str, dit_path: str):
        """Load diffusion models."""
        try:
            engine = get_diffusion_engine()
            engine.load(vae_path, dit_path)
            return {"status": "loaded", "vae": vae_path, "dit": dit_path}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/status")
    async def diffusion_status():
        """Check diffusion engine status."""
        engine = diffusion_engine
        return {
            "loaded": engine is not None and engine.is_loaded,
            "device": str(engine.device) if engine else None,
        }

    @router.post("/generations")
    async def generate_image(request: ImageGenerateRequest) -> ImagesResponse:
        """
        Generate images from text prompt.

        OpenAI-compatible endpoint.
        """
        try:
            engine = get_diffusion_engine()

            if not engine.is_loaded:
                raise HTTPException(
                    status_code=400,
                    detail="Diffusion models not loaded. Call /v1/images/load first."
                )

            config = ImageGenerationConfig(
                width=request.width,
                height=request.height,
                num_inference_steps=request.num_inference_steps,
                guidance_scale=request.guidance_scale,
                seed=request.seed,
            )

            images_data = []
            for _ in range(request.num_images):
                result = engine.generate(
                    prompt=request.prompt,
                    config=config,
                    negative_prompt=request.negative_prompt,
                )

                images_data.append(ImageResponse(
                    b64_json=result.to_base64(),
                    seed=result.seed,
                ))

                # Update seed for next image
                if config.seed is not None:
                    config.seed += 1

            return ImagesResponse(
                created=int(datetime.now().timestamp()),
                data=images_data,
            )

        except Exception as e:
            logger.error(f"Image generation error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    return router
