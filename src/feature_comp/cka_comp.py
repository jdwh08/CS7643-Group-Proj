# general importss
import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import yaml
import argparse

# arg parser
parser = argparse.ArgumentParser(description = 'used to select test image you was to visualize forward pass for')
parser.add_argument("--unet-run", type = str, required = True, help = 'pass name of latest UNet run')
parser.add_argument("--seg-run", type = str, required = True, help = 'pass name of latest SegFormer run')

device = 'cuda' if torch.cuda.is_available else 'cpu'

# +
PROJECT_ROOT = os.path.expanduser("~/scratch/CS7643-Group-Proj/")
print("PROJECT_ROOT:", PROJECT_ROOT)

# add to python path
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
# -

# project specific imports
from src.training.unet.model import FloodUNet
from src.training.ViT.flood_segformer import FloodSegformer
from config import Config
from src.data.loaders import make_s1hand_loaders, make_s1weak_loader
from src.data.io import clean_hand_mask
from src.data.s1augmentations import S1_MEAN, S1_STD

# unet config
config_path = os.path.join(PROJECT_ROOT + "src/training/unet/config_unet.yml")
with open(config_path, "r") as f:
    config_dict = yaml.safe_load(f)
    config = Config(config_dict=config_dict)

batch_size = int(config.train.batch_size)
num_workers = int(config.train.num_workers)

# parse unet args
args = parser.parse_args()
latest_run_unet = args.unet_run

# load model
unet = FloodUNet(config)
unet.model.to(device)

unet_weights = torch.load(PROJECT_ROOT + 'src/training/unet/logs/' + latest_run_unet + '/weights.pth')
unet.load_state_dict(unet_weights)

# dataloaders
train_loader, val_loader, test_loader = make_s1hand_loaders(
            DATA_ROOT, batch_size, num_workers
        )

# get unet first and last layers
unet_enc_first_out = []
unet_enc_last_out = []
with torch.no_grad():
    unet.model.eval()
    for batch in test_loader:
        enc_out = unet.model.encoder(batch['image'].to(device))
        #print(len(enc_out))
        flat_first_layer = torch.flatten(enc_out[0], start_dim = 1)
        flat_last_layer = torch.flatten(enc_out[5], start_dim = 1)
        #print(enc_out)
        unet_enc_first_out.append(flat_first_layer)
        unet_enc_last_out.append(flat_last_layer)

unet_first_layer = torch.cat(unet_enc_first_out, dim = 0)
unet_last_layer = torch.cat(unet_enc_last_out, dim = 0)

# parse segformer args
latest_run_seg = args.seg_run

segformer = FloodSegformer()
segformer.load_weights(latest_run_seg)

# get first and last layer for segformer
segformer_first_layer_out = []
segformer_last_layer_out = []
with torch.no_grad():
    segformer.model.eval()
    for batch in test_loader:
        seg_out = segformer.model(batch['image'].to(device), output_hidden_states = True)['hidden_states']
        seg_out_first_flat = torch.flatten(seg_out[0], start_dim = 1)
        seg_out_last_flat = torch.flatten(seg_out[3], start_dim =1)
        segformer_first_layer_out.append(seg_out_first_flat)
        segformer_last_layer_out.append(seg_out_last_flat)

segformer_first_layer = torch.cat(segformer_first_layer_out, dim = 0)
segformer_last_layer = torch.cat(segformer_last_layer_out, dim = 0)


# normalize values
def norm(values):
    values_norm = (values - values.mean())/values.std()
    return values_norm


segformer_first_layer_norm = norm(segformer_first_layer)
unet_first_layer_norm = norm(unet_first_layer)
segformer_last_layer_norm = norm(segformer_last_layer)
unet_last_layer_norm = norm(unet_last_layer)


# modified for pytorch base on: https://roberttlange.com/posts/2021/10/all-cnn-c-cka/
def linear_kernel(x, y):
    k = x@ x.T
    l = y@ y.T
    return k,l


def HSIC(k, l):
    m = k.shape[0]
    h = torch.eye(m) - 1/m * torch.ones((m,m))
    h = h.to(device)
    num = torch.trace(k@h@l@h).to(device)
    return num/(m-1)**2


def lin_cka(x, y):
    k, l = linear_kernel(x,y)
    hsic_kl = HSIC(k, l)
    hsic_kk = HSIC(k,k)
    hsic_ll = HSIC(l, l)
    return hsic_kl/torch.sqrt(hsic_kk*hsic_ll)


first_layers_cka = lin_cka(segformer_first_layer_norm, unet_first_layer_norm).detach().cpu().numpy()

last_layers_cka = lin_cka(segformer_last_layer_norm, unet_last_layer_norm).detach().cpu().numpy()

print(f'Linear CKA between first layers: {first_layers_cka}')

print(f'Linear CKA between last layers: {last_layers_cka}')
