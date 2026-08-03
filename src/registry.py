import importlib
import yaml
import os
from config import config, BASE_DIR

class Registry:
    def __init__(self):
        self._registry = {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        for module in self._registry.values():
            if hasattr(module, "stop"):
                print("Stopping module:", module)
                module.stop()

    def register(self, capability: str, module):
        self._registry[capability] = module

    def get(self, capability: str):
        try:
            return self._registry[capability]
        except KeyError:
            raise KeyError(f"No module registered for capability '{capability}'")

    @classmethod
    def from_config(cls) -> "Registry":
        api = cls()
        resolved = resolve_paths(config)
        for capability, spec in resolved.items():
            module_name = spec.pop("module")
            mod = importlib.import_module(f"modules.{capability}.{module_name}")
            instance = mod.Module(**spec)
            api.register(capability, instance)
        return api

def resolve_paths(config):
    for section in config.values():
        if "model_path" in section:
            section["model_path"] = os.path.join(BASE_DIR, "models", section["model_path"])
    return config