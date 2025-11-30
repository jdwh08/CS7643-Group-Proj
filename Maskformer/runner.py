import os
import sys
import torch
from flood_maskformer import FloodMaskformer

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
LOG_DIR = os.path.join(BASE_DIR, "logs")
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


torch.manual_seed(31415935)
torch.cuda.manual_seed(31415935)
torch.cuda.manual_seed_all(31415935)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


maskformer = FloodMaskformer(dataset="weak")

# maskformer.training()


# print("LOSS HISTORY TRAIN", maskformer.train_loss_history)

# print("VAL LOSS HISTORY", maskformer.val_loss_history)

# print("IOU HISTORY TRAIN", maskformer.train_iou_history)

# print("VAL IOU HISTORY", maskformer.val_iou_history)

# maskformer.save_model()

maskformer.load_weights("maskformer_run_20251130_08:46")
maskformer.save_model()
