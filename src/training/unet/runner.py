from src.training.unet.trainer import UNetTrainer
import random
import numpy as np
import torch
import argparse

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

print("RUNNER STARTED!", flush=True)

if __name__ == "__main__":
    set_seed(903219991)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to UNet config YAML (defaults to config_unet.yml in this folder).",
    )
    args = parser.parse_args()
    
    trainer = UNetTrainer()
    trainer.training()