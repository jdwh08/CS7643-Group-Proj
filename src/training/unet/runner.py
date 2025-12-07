from src.training.unet.trainer import UNetTrainer


print("RUNNER STARTED!", flush=True)

if __name__ == "__main__":
    trainer = UNetTrainer()
    trainer.training()