# Cybrex
A desktop AI assistant.

## Installation

```bash

```

Edit `config.yaml`'s core and memory_long sections with your model path and hardware setting

## Example Configuration

```yaml

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

## Available Interfaces

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
- [ ] Action modules