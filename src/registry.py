import importlib
from config import config
from klogger import get_logger

log = get_logger("Registry")


class Registry:
    def __init__(self):
        self._registry = {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        for capability, module in self._registry.items():
            if hasattr(module, "stop"):
                log.info("Stopping module: %s (%s)", capability, module)
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
        for capability, module_name in config.items():
            log.info("Loading %s: %s", capability, module_name)
            mod = importlib.import_module(f"modules.{capability}.{module_name}")
            instance = mod.Module()
            api.register(capability, instance)
        return api