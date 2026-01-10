"""
TPU-INL Accelerated Inference Engine

Integrates TPU-INL for cross-platform hardware acceleration.
Automatically detects the best backend (CUDA, TPU, AMD, Intel, DirectML, CPU).
"""

import logging
from typing import Optional, Generator, List
from dataclasses import dataclass
import torch

logger = logging.getLogger("pyllm.inference.tpu_inl")

# Try to import TPU-INL
HAS_TPU_INL = False
try:
    from tpu_inl import get_backend, Backend
    from tpu_inl.inference import InferenceEngine as TPUINLEngine, EngineConfig
    from tpu_inl.inference import StreamingSession, StreamingConfig, StreamingMode
    from tpu_inl.inference import KVCache
    from tpu_inl.optimizers import (
        apply_kernel_fusion, FusionStrategy,
        quantize_for_inference, QuantizationConfig, QuantizationType,
        get_optimal_parallelism
    )
    HAS_TPU_INL = True
    logger.info("TPU-INL available for hardware acceleration")
except ImportError as e:
    logger.warning(f"TPU-INL not available: {e}. Install with: pip install -e ../tpu-inl")


@dataclass
class TPUINLConfig:
    """Configuration for TPU-INL accelerated inference."""
    # Backend selection
    backend: str = "auto"  # auto, cuda, tpu, amd, intel, directml, cpu

    # Precision
    dtype: str = "auto"  # auto, fp32, fp16, bf16, int8

    # Optimizations
    use_kernel_fusion: bool = True
    use_quantization: bool = False
    quantization_type: str = "bf16"  # fp16, bf16, int8

    # KV Cache
    use_kv_cache: bool = True
    max_batch_size: int = 32
    max_seq_len: int = 4096

    # Compilation
    use_torch_compile: bool = True

    # Streaming
    stream_mode: str = "token"  # token, word, sentence


class TPUINLAcceleratedEngine:
    """
    Inference engine with TPU-INL hardware acceleration.

    This wraps the base pyllm InferenceEngine and adds:
    - Automatic backend detection (CUDA, TPU, AMD, Intel, DirectML)
    - Triton kernel fusion on CUDA/AMD
    - XLA compilation on TPU
    - Optimized KV caching
    - Quantization support
    """

    def __init__(self, config: Optional[TPUINLConfig] = None):
        self.config = config or TPUINLConfig()
        self.model = None
        self.tokenizer = None
        self.tpu_inl_engine = None
        self.backend = None
        self.device = None
        self._loaded = False

        # Detect backend
        if HAS_TPU_INL:
            if self.config.backend == "auto":
                self.backend = get_backend()
            else:
                self.backend = Backend(self.config.backend)
            logger.info(f"TPU-INL Backend: {self.backend.value}")

    def _get_device(self) -> torch.device:
        """Get device based on detected backend."""
        if not HAS_TPU_INL:
            if torch.cuda.is_available():
                return torch.device("cuda")
            return torch.device("cpu")

        if self.backend == Backend.CUDA:
            return torch.device("cuda")
        elif self.backend == Backend.AMD:
            return torch.device("cuda")  # ROCm uses cuda device
        elif self.backend == Backend.INTEL:
            if hasattr(torch, 'xpu') and torch.xpu.is_available():
                return torch.device("xpu")
            return torch.device("cpu")
        elif self.backend == Backend.DIRECTML:
            try:
                import torch_directml
                return torch_directml.device()
            except ImportError:
                return torch.device("cpu")
        elif self.backend == Backend.TPU:
            return torch.device("cpu")  # JAX handles TPU
        else:
            return torch.device("cpu")

    def load(self, model, tokenizer) -> None:
        """
        Load and optimize model with TPU-INL.

        Args:
            model: PyTorch model (already loaded)
            tokenizer: Tokenizer
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = self._get_device()

        logger.info(f"Loading model with TPU-INL acceleration on {self.device}")

        # Move to device
        self.model = self.model.to(self.device)
        self.model.eval()

        if HAS_TPU_INL:
            # Apply optimizations
            self._apply_optimizations()

            # Create TPU-INL inference engine
            engine_config = EngineConfig(
                backend=self.config.backend,
                dtype=self.config.dtype,
                max_batch_size=self.config.max_batch_size,
                max_sequence_length=self.config.max_seq_len,
                use_kv_cache=self.config.use_kv_cache,
                use_torch_compile=self.config.use_torch_compile,
            )

            self.tpu_inl_engine = TPUINLEngine(
                self.model,
                self.tokenizer,
                engine_config
            )
            logger.info("TPU-INL engine initialized")
        else:
            # Fallback: just use torch.compile if available
            if hasattr(torch, 'compile') and self.device.type == "cuda":
                try:
                    self.model = torch.compile(self.model, mode="reduce-overhead")
                    logger.info("Using torch.compile fallback")
                except Exception as e:
                    logger.warning(f"torch.compile failed: {e}")

        self._loaded = True
        logger.info("Model loaded with TPU-INL acceleration")

    def _apply_optimizations(self):
        """Apply TPU-INL optimizations to the model."""
        if not HAS_TPU_INL:
            return

        # Kernel fusion
        if self.config.use_kernel_fusion:
            if self.backend in [Backend.CUDA, Backend.AMD]:
                self.model = apply_kernel_fusion(
                    self.model,
                    FusionStrategy.TRITON
                )
                logger.info("Applied Triton kernel fusion")
            elif self.backend == Backend.INTEL:
                self.model = apply_kernel_fusion(
                    self.model,
                    FusionStrategy.ONEDNN
                )
                logger.info("Applied oneDNN kernel fusion")
            else:
                self.model = apply_kernel_fusion(
                    self.model,
                    FusionStrategy.TORCH_COMPILE
                )
                logger.info("Applied torch.compile fusion")

        # Quantization
        if self.config.use_quantization:
            quant_type = {
                "fp16": QuantizationType.FP16,
                "bf16": QuantizationType.BF16,
                "int8": QuantizationType.INT8,
            }.get(self.config.quantization_type, QuantizationType.BF16)

            quant_config = QuantizationConfig(
                weight_type=quant_type,
                activation_type=QuantizationType.FP16,
                dynamic=True
            )

            self.model = quantize_for_inference(
                self.model,
                quant_config,
                self.backend.value
            )
            logger.info(f"Applied {self.config.quantization_type} quantization")

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
    ) -> Generator[str, None, None]:
        """
        Generate text with streaming.

        Uses TPU-INL engine if available, otherwise falls back to basic generation.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        if HAS_TPU_INL and self.tpu_inl_engine:
            # Use TPU-INL streaming
            stream_mode = {
                "token": StreamingMode.TOKEN,
                "word": StreamingMode.WORD,
                "sentence": StreamingMode.SENTENCE,
            }.get(self.config.stream_mode, StreamingMode.TOKEN)

            stream_config = StreamingConfig(mode=stream_mode)

            from tpu_inl.inference import GenerationConfig
            gen_config = GenerationConfig(
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
            )

            # Use TPU-INL streaming engine
            from tpu_inl.inference.streaming import StreamingEngine
            streaming_engine = StreamingEngine(
                self.model,
                self.tokenizer,
                self.device
            )

            yield from streaming_engine.stream_generate(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                config=stream_config
            )
        else:
            # Fallback: basic generation
            yield from self._basic_generate(
                prompt, max_tokens, temperature, top_p, top_k, repetition_penalty
            )

    def _basic_generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
    ) -> Generator[str, None, None]:
        """Basic generation without TPU-INL (fallback)."""
        # Tokenize
        input_ids = self.tokenizer.encode(prompt)
        if isinstance(input_ids, list):
            input_ids = torch.tensor([input_ids], device=self.device)
        else:
            input_ids = input_ids.to(self.device)

        generated = input_ids

        with torch.inference_mode():
            for _ in range(max_tokens):
                # Forward
                outputs = self.model(generated)
                if isinstance(outputs, tuple):
                    logits = outputs[0][:, -1, :]
                elif hasattr(outputs, 'logits'):
                    logits = outputs.logits[:, -1, :]
                else:
                    logits = outputs[:, -1, :]

                # Temperature
                logits = logits / max(temperature, 0.01)

                # Top-k
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float('-inf')

                # Top-p
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')

                # Sample
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Decode and yield
                token_text = self.tokenizer.decode([next_token.item()])
                yield token_text

                # Update
                generated = torch.cat([generated, next_token], dim=1)

                # Check EOS
                eos_id = getattr(self.tokenizer, 'eos_token_id', 2)
                if next_token.item() == eos_id:
                    break

    def get_stats(self) -> dict:
        """Get inference statistics."""
        stats = {
            "backend": self.backend.value if self.backend else "unknown",
            "device": str(self.device),
            "tpu_inl_available": HAS_TPU_INL,
        }

        if HAS_TPU_INL and self.tpu_inl_engine:
            stats.update(self.tpu_inl_engine.get_stats())

        return stats

    @property
    def is_loaded(self) -> bool:
        return self._loaded


def create_accelerated_engine(
    model,
    tokenizer,
    backend: str = "auto",
    use_quantization: bool = False,
    quantization_type: str = "bf16"
) -> TPUINLAcceleratedEngine:
    """
    Factory function to create an accelerated engine.

    Usage:
        from pyllm.inference.tpu_inl_engine import create_accelerated_engine

        engine = create_accelerated_engine(model, tokenizer)
        for token in engine.generate("Hello"):
            print(token, end="", flush=True)
    """
    config = TPUINLConfig(
        backend=backend,
        use_quantization=use_quantization,
        quantization_type=quantization_type,
    )

    engine = TPUINLAcceleratedEngine(config)
    engine.load(model, tokenizer)

    return engine
