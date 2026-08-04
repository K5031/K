from controller import run
import os
import warnings

warnings.filterwarnings("ignore")
os.environ["HF_HUB_OFFLINE"] = "1"

if __name__ == "__main__":
    run()
