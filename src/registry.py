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
        keys = list(config.keys())
        if not keys or keys[0] != "controller":
            raise ValueError(
                f"config must have 'controller' as its first key, got: {keys}"
            )

        controller_name = config["controller"]
        controller_mod = importlib.import_module(f"modules.controller.{controller_name}")
        required = set(controller_mod.Module.required_modules)
        provided = set(config.keys()) - {"controller"}

        missing = required - provided
        extra = provided - required
        if missing or extra:
            problems = []
            if missing:
                problems.append(f"missing: {sorted(missing)}")
            if extra:
                problems.append(f"unexpected: {sorted(extra)}")
            raise ValueError(
                f"config does not match controller '{controller_name}' requirements — "
                + ", ".join(problems)
            )

        api = cls()
        for capability, module_name in config.items():
            log.info("Loading %s: %s", capability, module_name)
            mod = importlib.import_module(f"modules.{capability}.{module_name}")
            instance = mod.Module()
            api.register(capability, instance)
        return api