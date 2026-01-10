"""Inference module - LLM and Diffusion engines."""

from pyllm.inference.engine import InferenceEngine
from pyllm.inference.generator import StreamingGenerator
from pyllm.inference.diffusion import DiffusionEngine, ImageGenerationConfig, GeneratedImage

# TPU-INL accelerated engine (optional)
try:
    from pyllm.inference.tpu_inl_engine import (
        TPUINLAcceleratedEngine,
        TPUINLConfig,
        create_accelerated_engine,
        HAS_TPU_INL,
    )
except ImportError:
    TPUINLAcceleratedEngine = None
    TPUINLConfig = None
    create_accelerated_engine = None
    HAS_TPU_INL = False

__all__ = [
    "InferenceEngine",
    "StreamingGenerator",
    "DiffusionEngine",
    "ImageGenerationConfig",
    "GeneratedImage",
    # TPU-INL (optional)
    "TPUINLAcceleratedEngine",
    "TPUINLConfig",
    "create_accelerated_engine",
    "HAS_TPU_INL",
]
