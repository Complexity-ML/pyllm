"""LLM Inference Engine - supports multiple backends."""

import logging
from typing import Optional, Generator, Dict, Any, List
from dataclasses import dataclass
import torch

from pyllm.core.config import ModelConfig

logger = logging.getLogger("pyllm.inference")


@dataclass
class Message:
    """Chat message."""
    role: str  # "user", "assistant", "system"
    content: str


@dataclass
class GenerationConfig:
    """Generation parameters."""
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.2
    max_new_tokens: int = 256
    do_sample: bool = True
    stream: bool = True


class InferenceEngine:
    """
    LLM Inference Engine.

    Supports:
    - INL-LLM models
    - HuggingFace transformers
    - Streaming generation
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.model = None
        self.tokenizer = None
        self.device = None
        self._loaded = False

    def load(self, model_path: Optional[str] = None) -> None:
        """Load the model."""
        path = model_path or self.config.path

        if not path:
            raise ValueError("Model path required")

        logger.info(f"Loading model from {path}")

        # Determine device
        if self.config.device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif self.config.device == "mps" and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        logger.info(f"Using device: {self.device}")

        # Try loading as INL-LLM first
        if self._try_load_inl_llm(path):
            logger.info("Loaded as INL-LLM model")
        elif self._try_load_transformers(path):
            logger.info("Loaded as Transformers model")
        else:
            raise RuntimeError(f"Could not load model from {path}")

        self._loaded = True
        logger.info("Model loaded successfully")

    def _try_load_inl_llm(self, path: str) -> bool:
        """Try loading as INL-LLM model."""
        try:
            from transformers import AutoTokenizer

            # Check if it's a safetensors file
            if path.endswith(".safetensors") or path.endswith(".pt"):
                from safetensors.torch import load_file

                # Try v2 architecture first
                try:
                    from inl_llm import IntegratorLanguageModel

                    self.model = IntegratorLanguageModel(
                        vocab_size=50261,
                        d_model=1280,
                        num_layers=18,
                        num_heads=20,
                        num_iterations_per_layer=2,
                        feedforward_dim=5120,
                        max_seq_len=self.config.max_seq_len
                    )

                    if path.endswith(".safetensors"):
                        state_dict = load_file(path)
                    else:
                        state_dict = torch.load(path, map_location="cpu")

                    self.model.load_state_dict(state_dict)
                    self.model.to(self.device)
                    self.model.eval()

                    self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
                    self.tokenizer.pad_token = self.tokenizer.eos_token

                    return True

                except ImportError:
                    pass

            return False

        except Exception as e:
            logger.debug(f"INL-LLM load failed: {e}")
            return False

    def _try_load_transformers(self, path: str) -> bool:
        """Try loading as HuggingFace Transformers model."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(path)
            self.model = AutoModelForCausalLM.from_pretrained(
                path,
                torch_dtype=getattr(torch, self.config.dtype),
                device_map="auto" if self.device.type == "cuda" else None,
            )

            if self.device.type != "cuda":
                self.model.to(self.device)

            self.model.eval()

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            return True

        except Exception as e:
            logger.debug(f"Transformers load failed: {e}")
            return False

    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> Generator[str, None, None]:
        """
        Generate text from prompt with streaming.

        Yields tokens one at a time.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        config = config or GenerationConfig(
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            repetition_penalty=self.config.repetition_penalty,
            max_new_tokens=self.config.max_new_tokens,
        )

        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.device)

        # Generate with streaming
        with torch.no_grad():
            for token in self._generate_tokens(input_ids, config):
                yield token

    def _generate_tokens(
        self,
        input_ids: torch.Tensor,
        config: GenerationConfig,
    ) -> Generator[str, None, None]:
        """Generate tokens one at a time."""
        generated = input_ids.clone()
        past_tokens = set(input_ids[0].tolist())

        for _ in range(config.max_new_tokens):
            # Forward pass
            with torch.no_grad():
                outputs = self.model(generated)

                # Get logits for last token
                if hasattr(outputs, "logits"):
                    logits = outputs.logits[:, -1, :]
                else:
                    logits = outputs[:, -1, :]

            # Apply repetition penalty
            if config.repetition_penalty != 1.0:
                for token_id in past_tokens:
                    if logits[0, token_id] < 0:
                        logits[0, token_id] *= config.repetition_penalty
                    else:
                        logits[0, token_id] /= config.repetition_penalty

            # Apply temperature
            if config.temperature > 0:
                logits = logits / config.temperature

            # Apply top-k
            if config.top_k > 0:
                indices_to_remove = logits < torch.topk(logits, config.top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')

            # Apply top-p (nucleus sampling)
            if config.top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > config.top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(
                    dim=-1, index=sorted_indices, src=sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')

            # Sample
            if config.do_sample:
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            # Check for EOS
            if next_token.item() == self.tokenizer.eos_token_id:
                break

            # Decode and yield
            token_text = self.tokenizer.decode(next_token[0], skip_special_tokens=True)
            yield token_text

            # Update state
            generated = torch.cat([generated, next_token], dim=-1)
            past_tokens.add(next_token.item())

            # Truncate if needed
            if generated.shape[1] > self.config.max_seq_len:
                generated = generated[:, -self.config.max_seq_len:]

    def chat(
        self,
        messages: List[Message],
        config: Optional[GenerationConfig] = None,
    ) -> Generator[str, None, None]:
        """
        Chat with the model.

        Formats messages into a prompt and generates response.
        """
        prompt = self._format_chat(messages)
        yield from self.generate(prompt, config)

    def _format_chat(self, messages: List[Message]) -> str:
        """Format chat messages into a prompt."""
        lines = []

        for msg in messages:
            if msg.role == "system":
                lines.append(f"System: {msg.content}")
            elif msg.role == "user":
                lines.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                lines.append(f"Assistant: {msg.content}")

        lines.append("Assistant:")
        return "\n".join(lines)

    def complete(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> str:
        """
        Generate complete response (non-streaming).
        """
        tokens = list(self.generate(prompt, config))
        return "".join(tokens)

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded
