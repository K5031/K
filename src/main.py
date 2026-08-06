from registry import Registry
import os
import warnings

warnings.filterwarnings("ignore")
os.environ["HF_HUB_OFFLINE"] = "1"


def main():
    with Registry.from_config() as api:
        api.get("controller").run(api)

if __name__ == "__main__":
    main()
