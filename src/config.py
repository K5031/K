import shutil
import os
import sys

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

CONFIG_DIR = os.path.expanduser("~/.config/k")
CONFIG_PATH = os.path.join(CONFIG_DIR, "config.toml")
SYSTEM_PROMPT_PATH = os.path.join(CONFIG_DIR, "system_prompt.txt")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG = os.path.join(BASE_DIR, "config.example.toml")
DEFAULT_PROMPT = os.path.join(BASE_DIR, "default_prompt.txt")


def ensure_config():
    os.makedirs(CONFIG_DIR, exist_ok=True)
    if not os.path.exists(CONFIG_PATH):
        shutil.copy(DEFAULT_CONFIG, CONFIG_PATH)
    if not os.path.exists(SYSTEM_PROMPT_PATH):
        shutil.copy(DEFAULT_PROMPT, SYSTEM_PROMPT_PATH)


def load_config():
    with open(CONFIG_PATH, "rb") as f:
        return tomllib.load(f)


def load_system_prompt():
    with open(SYSTEM_PROMPT_PATH) as f:
        return f.read()


ensure_config()
config = load_config()
system_prompt = load_system_prompt()