"""Streaming generator utilities."""

import asyncio
from typing import Generator, AsyncGenerator, Optional
from dataclasses import dataclass

from pyllm.inference.engine import InferenceEngine, GenerationConfig, Message


@dataclass
class StreamChunk:
    """A chunk of streamed text."""
    text: str
    finished: bool = False


class StreamingGenerator:
    """
    Async streaming generator wrapper.

    Wraps the synchronous generator for async contexts (FastAPI, etc.)
    """

    def __init__(self, engine: InferenceEngine):
        self.engine = engine

    async def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Async generator for streaming tokens."""
        loop = asyncio.get_event_loop()

        # Run sync generator in executor
        gen = self.engine.generate(prompt, config)

        try:
            while True:
                # Get next token in thread pool
                try:
                    token = await loop.run_in_executor(None, next, gen)
                    yield StreamChunk(text=token, finished=False)
                except StopIteration:
                    yield StreamChunk(text="", finished=True)
                    break
        except Exception as e:
            yield StreamChunk(text=f"[Error: {e}]", finished=True)

    async def chat(
        self,
        messages: list[Message],
        config: Optional[GenerationConfig] = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Async chat generator."""
        loop = asyncio.get_event_loop()
        gen = self.engine.chat(messages, config)

        try:
            while True:
                try:
                    token = await loop.run_in_executor(None, next, gen)
                    yield StreamChunk(text=token, finished=False)
                except StopIteration:
                    yield StreamChunk(text="", finished=True)
                    break
        except Exception as e:
            yield StreamChunk(text=f"[Error: {e}]", finished=True)
