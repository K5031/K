# K

A desktop AI assistant built from small, swappable modules — speech or text in, an LLM (local or cloud) in the middle, speech or text out — with short-term conversation context and long-term memory extraction.

## Installation

```bash
git clone git@github.com:K5031/K.git
cd K
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

If you'll use a local core or memory module, drop the GGUF models you want into `models/`.

Run it:

```bash
python src/main.py
```

On first run this creates `~/.config/k/config.toml` (copied from `config.example.toml`) and `~/.config/k/system_prompt.txt` (copied from `default_prompt.txt`). Edit `config.toml` to pick your modules and `system_prompt.txt` to customize the assistant's personality.

Some modules need credentials — e.g. the `deepseek` core module reads its key from the `DEEPSEEK_API_KEY` environment variable.

## Example Configuration

```toml
controller = "local"
input = "terminal"
output = "terminal"
core = "llama_cpp"
context = "local"
memory = "remis"
```

Each key is a capability, and its value is the module (filename, no `.py`) that fills it, loaded from `src/modules/<capability>/`. The `controller` module must be listed first — it declares which other capabilities it needs, and `Registry.from_config` will refuse to start if the config doesn't match exactly.

A capability can also take a list of module names, e.g. `output = ["terminal", "kokoro"]`, to run several modules for that capability side by side.

## Built-in Modules

| Capability | Module | Description |
|---|---|---|
| controller | `local` | Interactive loop; interrupts generation on new input or Ctrl+C |
| controller | `cloud` | Interactive loop; also feeds reported token usage back into context trimming |
| input | `terminal` | Text input via stdin |
| input | `whisper` | Microphone capture with local Whisper speech-to-text |
| output | `terminal` | Prints tokens to stdout |
| output | `kokoro` | Streams sentence-buffered text through Kokoro TTS |
| core | `llama_cpp` | Local inference via llama.cpp, using a GGUF model from `models/` |
| core | `deepseek` | Cloud inference via the DeepSeek API |
| context | `local` | In-memory conversation buffer, trimmed by counting tokens with a local tokenizer |
| context | `cloud` | In-memory conversation buffer, trimmed using usage stats reported by the core module |
| memory | `remis` | Persistent long-term memory — extracts durable facts with a local LLM, then embeds, dedupes, and conflict-resolves them against a Chroma store |

## Writing a Module

Every module is a Python file with a class named `Module` that implements the relevant interface:

```python
from interfaces import InputInterface

class Module(InputInterface):
    def get_input(self) -> str:
        return input("> ")

    def has_input(self) -> bool:
        return False
```

Place it under `src/modules/<capability>/your_module.py` and reference it by name in `config.toml`. Details about each interface can be found in `src/interfaces.py`.

## Available Interfaces

- `InputInterface` — `get_input() -> str`, `has_input() -> bool`
- `OutputInterface` — `send(token: str)`, `interrupt()`, `flush()`
- `CoreInterface` — `generate(user_input, context, memories) -> Iterator[str]`, `set_system_prompt(prompt)`, `interrupt()`
- `ContextInterface` — `add(role, content)`, `get() -> list[dict]`, `clear()`
- `MemoryInterface` — `store(messages)`, `retrieve(query) -> str`
- `ControllerInterface` — `run(api)`, plus a `required_modules: list[str]` class attribute naming the capabilities it needs

## Roadmap

- [ ] Model warmup
- [x] Context overflow prevention
- [x] Interrupt mid-generation
- [ ] Vision module
- [ ] Action modules
