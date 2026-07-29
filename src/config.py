import yaml
import os

CONFIG_DIR = os.path.expanduser("~/.config/k")
CONFIG_PATH = os.path.join(CONFIG_DIR, "config.yaml")
SYSTEM_PROMPT_PATH = os.path.join(CONFIG_DIR, "system_prompt.txt")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(BASE_DIR, "..", "config.example.yaml")
DEFAULT_PROMPT = os.path.join(BASE_DIR, "..", "default_prompt.txt")

def ensure_config():
    os.makedirs(CONFIG_DIR, exist_ok=True)

    global CONFIG_PATH, SYSTEM_PROMPT_PATH
    if not os.path.exists(CONFIG_PATH):
        CONFIG_PATH = DEFAULT_CONFIG
    if not os.path.exists(SYSTEM_PROMPT_PATH):
        SYSTEM_PROMPT_PATH = DEFAULT_PROMPT

def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)

def load_system_prompt():
    with open(SYSTEM_PROMPT_PATH) as f:
        return f.read()

ensure_config()
config = load_config()
system_prompt = load_system_prompt()