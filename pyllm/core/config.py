"""Configuration for PyLLM."""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import json


@dataclass
class ModelConfig:
    """Model configuration."""
    name: str = "inl-llm"
    path: Optional[str] = None
    device: str = "cuda"  # cuda, cpu, mps
    dtype: str = "float16"  # float16, float32, bfloat16
    max_seq_len: int = 1024

    # Generation defaults
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.2
    max_new_tokens: int = 256


@dataclass
class ServerConfig:
    """Server configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    reload: bool = False


@dataclass
class UIConfig:
    """UI configuration."""
    title: str = "PyLLM Chat"
    theme: str = "dark"
    max_history: int = 50


@dataclass
class Config:
    """Main configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    ui: UIConfig = field(default_factory=UIConfig)

    @classmethod
    def load(cls, path: Optional[str] = None) -> "Config":
        """Load configuration from file or environment."""
        config = cls()

        if path and os.path.exists(path):
            with open(path, "r") as f:
                data = json.load(f)
                config = cls._from_dict(data)

        config._load_env()
        return config

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Create config from dictionary."""
        config = cls()

        if "model" in data:
            config.model = ModelConfig(**data["model"])
        if "server" in data:
            config.server = ServerConfig(**data["server"])
        if "ui" in data:
            config.ui = UIConfig(**data["ui"])

        return config

    def _load_env(self) -> None:
        """Load from environment variables."""
        # Model
        if path := os.getenv("PYLLM_MODEL_PATH"):
            self.model.path = path
        if device := os.getenv("PYLLM_DEVICE"):
            self.model.device = device

        # Server
        if host := os.getenv("PYLLM_HOST"):
            self.server.host = host
        if port := os.getenv("PYLLM_PORT"):
            self.server.port = int(port)
