import os
import sys
import torch
from flood_segformer import FloodSegformer

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../../.."))
LOG_DIR = os.path.join(BASE_DIR, "logs")
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


torch.manual_seed(31415935)
torch.cuda.manual_seed(31415935)
torch.cuda.manual_seed_all(31415935)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


maskformer = FloodSegformer(dataset="weak")

maskformer.training()

maskformer.save_model()

# maskformer.load_weights("segformer_run_weak_20251207_16:52")

# maskformer.test()
