"""Chat templates for different model formats."""

from typing import List, Optional
from dataclasses import dataclass
from enum import Enum


class TemplateType(str, Enum):
    """Supported chat template formats."""
    SIMPLE = "simple"           # Basic "User: ... Assistant: ..."
    CHATML = "chatml"           # <|im_start|>role\ncontent<|im_end|>
    LLAMA = "llama"             # [INST] ... [/INST]
    ALPACA = "alpaca"           # ### Instruction: ... ### Response:
    VICUNA = "vicuna"           # USER: ... ASSISTANT:
    MISTRAL = "mistral"         # [INST] ... [/INST]
    PHI = "phi"                 # <|user|> ... <|assistant|>
    ZEPHYR = "zephyr"           # <|user|>\n ... <|assistant|>\n


@dataclass
class Message:
    """Chat message."""
    role: str  # "user", "assistant", "system"
    content: str


class ChatTemplate:
    """
    Chat template formatter.

    Converts messages to model-specific format.
    """

    def __init__(self, template_type: TemplateType = TemplateType.SIMPLE):
        self.template_type = template_type

    def format(self, messages: List[Message], add_generation_prompt: bool = True) -> str:
        """Format messages according to template type."""
        if self.template_type == TemplateType.SIMPLE:
            return self._format_simple(messages, add_generation_prompt)
        elif self.template_type == TemplateType.CHATML:
            return self._format_chatml(messages, add_generation_prompt)
        elif self.template_type == TemplateType.LLAMA:
            return self._format_llama(messages, add_generation_prompt)
        elif self.template_type == TemplateType.ALPACA:
            return self._format_alpaca(messages, add_generation_prompt)
        elif self.template_type == TemplateType.VICUNA:
            return self._format_vicuna(messages, add_generation_prompt)
        elif self.template_type == TemplateType.MISTRAL:
            return self._format_mistral(messages, add_generation_prompt)
        elif self.template_type == TemplateType.PHI:
            return self._format_phi(messages, add_generation_prompt)
        elif self.template_type == TemplateType.ZEPHYR:
            return self._format_zephyr(messages, add_generation_prompt)
        else:
            return self._format_simple(messages, add_generation_prompt)

    def _format_simple(self, messages: List[Message], add_gen: bool) -> str:
        """Simple format: User: ... Assistant: ..."""
        lines = []
        for msg in messages:
            if msg.role == "system":
                lines.append(f"System: {msg.content}\n")
            elif msg.role == "user":
                lines.append(f"User: {msg.content}\n")
            elif msg.role == "assistant":
                lines.append(f"Assistant: {msg.content}\n")

        if add_gen:
            lines.append("Assistant:")

        return "".join(lines)

    def _format_chatml(self, messages: List[Message], add_gen: bool) -> str:
        """ChatML format used by many models."""
        lines = []
        for msg in messages:
            lines.append(f"<|im_start|>{msg.role}\n{msg.content}<|im_end|>\n")

        if add_gen:
            lines.append("<|im_start|>assistant\n")

        return "".join(lines)

    def _format_llama(self, messages: List[Message], add_gen: bool) -> str:
        """Llama 2 chat format."""
        lines = []
        system_msg = None

        for msg in messages:
            if msg.role == "system":
                system_msg = msg.content
            elif msg.role == "user":
                if system_msg:
                    lines.append(f"[INST] <<SYS>>\n{system_msg}\n<</SYS>>\n\n{msg.content} [/INST]")
                    system_msg = None
                else:
                    lines.append(f"[INST] {msg.content} [/INST]")
            elif msg.role == "assistant":
                lines.append(f" {msg.content} </s>")

        return "".join(lines)

    def _format_alpaca(self, messages: List[Message], add_gen: bool) -> str:
        """Alpaca instruction format."""
        lines = []

        for msg in messages:
            if msg.role == "system":
                lines.append(f"### System:\n{msg.content}\n\n")
            elif msg.role == "user":
                lines.append(f"### Instruction:\n{msg.content}\n\n")
            elif msg.role == "assistant":
                lines.append(f"### Response:\n{msg.content}\n\n")

        if add_gen:
            lines.append("### Response:\n")

        return "".join(lines)

    def _format_vicuna(self, messages: List[Message], add_gen: bool) -> str:
        """Vicuna format."""
        lines = []

        for msg in messages:
            if msg.role == "system":
                lines.append(f"{msg.content}\n\n")
            elif msg.role == "user":
                lines.append(f"USER: {msg.content}\n")
            elif msg.role == "assistant":
                lines.append(f"ASSISTANT: {msg.content}</s>\n")

        if add_gen:
            lines.append("ASSISTANT:")

        return "".join(lines)

    def _format_mistral(self, messages: List[Message], add_gen: bool) -> str:
        """Mistral instruction format."""
        lines = ["<s>"]

        for msg in messages:
            if msg.role == "system":
                lines.append(f"[INST] {msg.content} [/INST]")
            elif msg.role == "user":
                lines.append(f"[INST] {msg.content} [/INST]")
            elif msg.role == "assistant":
                lines.append(f"{msg.content}</s>")

        return "".join(lines)

    def _format_phi(self, messages: List[Message], add_gen: bool) -> str:
        """Phi format."""
        lines = []

        for msg in messages:
            if msg.role == "system":
                lines.append(f"<|system|>\n{msg.content}<|end|>\n")
            elif msg.role == "user":
                lines.append(f"<|user|>\n{msg.content}<|end|>\n")
            elif msg.role == "assistant":
                lines.append(f"<|assistant|>\n{msg.content}<|end|>\n")

        if add_gen:
            lines.append("<|assistant|>\n")

        return "".join(lines)

    def _format_zephyr(self, messages: List[Message], add_gen: bool) -> str:
        """Zephyr format."""
        lines = []

        for msg in messages:
            if msg.role == "system":
                lines.append(f"<|system|>\n{msg.content}</s>\n")
            elif msg.role == "user":
                lines.append(f"<|user|>\n{msg.content}</s>\n")
            elif msg.role == "assistant":
                lines.append(f"<|assistant|>\n{msg.content}</s>\n")

        if add_gen:
            lines.append("<|assistant|>\n")

        return "".join(lines)

    @staticmethod
    def detect_template(model_name: str) -> TemplateType:
        """Auto-detect template from model name."""
        name = model_name.lower()

        if "llama" in name or "llama-2" in name:
            return TemplateType.LLAMA
        elif "mistral" in name or "mixtral" in name:
            return TemplateType.MISTRAL
        elif "vicuna" in name:
            return TemplateType.VICUNA
        elif "alpaca" in name:
            return TemplateType.ALPACA
        elif "phi" in name:
            return TemplateType.PHI
        elif "zephyr" in name:
            return TemplateType.ZEPHYR
        elif "chatml" in name or "qwen" in name or "yi" in name:
            return TemplateType.CHATML
        else:
            return TemplateType.SIMPLE
