import argparse
from pathlib import Path

from src.training.prithvi.train import PrithviTrainer

CONFIG_PATH = Path(__file__).parent / "config.yaml"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=CONFIG_PATH,
        help="Path to Prithvi config YAML (defaults to config.yaml in this folder).",
    )
    args = parser.parse_args()
    PrithviTrainer(args.config).run()
