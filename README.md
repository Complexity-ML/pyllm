# PyLLM

**Python LLM Inference with Streaming Chat** - A complete LLM inference platform with streaming responses, OpenAI-compatible API, and chat interface.

## Features

- **Streaming Generation** - Token-by-token streaming output
- **OpenAI-Compatible API** - Drop-in replacement for OpenAI endpoints
- **Multiple Backends** - Support for INL-LLM, HuggingFace Transformers
- **Chat Templates** - Support for various formats (ChatML, Llama, Mistral, etc.)
- **Streamlit UI** - Beautiful chat interface with Shadcn components
- **CLI Tools** - Generate, chat, and serve from command line

## Installation

```bash
pip install pyllm-inference

# With Streamlit UI
pip install pyllm-inference[ui]

# With INL-LLM support
pip install pyllm-inference[inl]
```

## Quick Start

### Start the API Server

```bash
# Start server with model
pyllm serve --model path/to/model.safetensors

# Custom port
pyllm serve --model path/to/model --port 8000
```

### Start the Chat UI

```bash
# Start Streamlit UI
pyllm ui --api-url http://localhost:8000
```

### Command Line Generation

```bash
# Generate text
pyllm generate --model path/to/model --prompt "Hello, world!"

# Interactive chat
pyllm chat --model path/to/model
```

## API Endpoints

### OpenAI-Compatible

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/v1/models` | List available models |
| POST | `/v1/chat/completions` | Chat completion (streaming) |
| POST | `/v1/generate` | Text generation (streaming) |

### Example: Chat Completion

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "messages": [
            {"role": "user", "content": "Hello!"}
        ],
        "stream": True
    },
    stream=True
)

for line in response.iter_lines():
    if line:
        print(line.decode())
```

### Example: With OpenAI Client

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="pyllm",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True
)

for chunk in response:
    print(chunk.choices[0].delta.content, end="")
```

## Python Usage

```python
from pyllm import InferenceEngine, Config
from pyllm.inference import GenerationConfig, Message

# Load model
engine = InferenceEngine()
engine.load("path/to/model.safetensors")

# Generate with streaming
for token in engine.generate("Hello, world!"):
    print(token, end="", flush=True)

# Chat
messages = [
    Message(role="user", content="What is Python?")
]

for token in engine.chat(messages):
    print(token, end="", flush=True)
```

## Chat Templates

Supports multiple chat formats:

- **Simple** - `User: ... Assistant: ...`
- **ChatML** - `<|im_start|>role ... <|im_end|>`
- **Llama** - `[INST] ... [/INST]`
- **Mistral** - `[INST] ... [/INST]`
- **Alpaca** - `### Instruction: ... ### Response:`
- **Vicuna** - `USER: ... ASSISTANT:`
- **Phi** - `<|user|> ... <|assistant|>`
- **Zephyr** - `<|user|>\n ... <|assistant|>\n`

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PYLLM_MODEL_PATH` | Model path | None |
| `PYLLM_DEVICE` | Device (cuda/cpu/mps) | cuda |
| `PYLLM_HOST` | Server host | 0.0.0.0 |
| `PYLLM_PORT` | Server port | 8000 |

### Config File

```json
{
  "model": {
    "path": "path/to/model",
    "device": "cuda",
    "max_seq_len": 1024
  },
  "server": {
    "host": "0.0.0.0",
    "port": 8000
  }
}
```

## Architecture

```
pyllm/
├── api/            # FastAPI routes (OpenAI-compatible)
├── cli/            # Command line interface
├── core/           # Configuration
├── inference/      # LLM engine and generation
│   ├── engine.py   # Main inference engine
│   ├── generator.py # Async streaming wrapper
│   └── templates.py # Chat templates
└── ui/             # Streamlit chat interface
```

## License

MIT License

## Credits

- Built for [INL-LLM](https://github.com/anthropics/pacific-prime) models
- Compatible with [HuggingFace Transformers](https://huggingface.co/transformers)
- UI powered by [Streamlit](https://streamlit.io/) + [Shadcn](https://ui.shadcn.com/)
