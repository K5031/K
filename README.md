# Cybrex
A modular AI assistant.

## Prerequisites

- Python 3.11 (newer versions may have package compatibility issues)
- A GGUF model file (e.g. from [HuggingFace](https://huggingface.co/models?library=gguf))
- CUDA toolkit (optional, for NVIDIA GPU acceleration)

For speech input/output, a microphone and speakers are required.

## Quickstart

```bash
git clone https://github.com/cybrex-ai/Cybrex
cd Cybrex
cp config.example.yaml config.yaml
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> **Note:** For NVIDIA GPU acceleration, install llama-cpp-python manually:
> ```bash
> CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --no-cache-dir
> ```

Edit `config.yaml`'s core and memory_long sections with your model path and hardware settings, then:

```bash
python main.py
```

## Example Configuration

```yaml
input: terminal
output: terminal
core:
  module: llama-cpp
  model_path: models/your-model.gguf
  n_gpu_layers: 10
  n_ctx: 32768
memory_short: flashback
memory_long: mem0
```

## Built-in Modules

| Capability | Module | Description |
|---|---|---|
| input | `whisper` | Whisper-based speech to text |
| input | `terminal` | Text input via terminal |
| output | `kokoro` | Kokoro TTS |
| output | `terminal` | Terminal text output |
| core | `llama-cpp` | Local inference via llama.cpp |
| memory_short | `flashback` | In-session conversation buffer |
| memory_long | `mem0` | Persistent memory via Mem0 |

## Writing a Module

Every module is a Python file with a class named `Module` that implements the relevant interface:

```python
from interfaces import InputInterface

class Module(InputInterface):
    def get_input(self) -> str:
        return input("> ")
```

Place it under `modules/<capability>/your_module.py` and reference it in `config.yaml`. Details about each interface can be found in interfaces.py.

## Interfaces

- `InputInterface` — `get_input() -> str`
- `OutputInterface` — `send(token: str)`, `flush()`, `interrupt()`
- `CoreInterface` — `generate(user_input, context, memories) -> Iterator[str]`
- `ShortTermMemoryInterface` — `add(role, content)`, `get() -> list`, `clear()`
- `LongTermMemoryInterface` — `store(messages)`, `retrieve(query) -> str`

## Roadmap

- [ ] Model warmup
- [ ] Context overflow prevention  
- [x] Interrupt mid-generation
- [ ] Vision module
- [ ] Internet access
- [ ] Action module
- [ ] GUI
- [ ] 2D character overlay module

## License

Apache 2.0