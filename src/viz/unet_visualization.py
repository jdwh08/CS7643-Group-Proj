# General Imports
import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    MaskFormerConfig,
    MaskFormerForInstanceSegmentation,
    MaskFormerImageProcessor,
)
import yaml
import argparse
import random

# arg parser
parser = argparse.ArgumentParser(description = 'used to select test image you was to visualize forward pass for')
parser.add_argument("--image-ind", type = int, required = False, help = 'index of image in test dataloader')
parser.add_argument("--run", type = str, required = True, help = 'pass name of latest UNet run')
parser.add_argument("--label-type", type = str, required = False, help = 'weak or hand')

# adding project root for project-specific imports
PROJECT_ROOT = os.path.expanduser("~/scratch/CS7643-Group-Proj/")
print("PROJECT_ROOT:", PROJECT_ROOT)

# add to python path
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")

# project-specific imports
from src.training.unet.model import FloodUNet
from config import Config
from src.data.loaders import make_s1hand_loaders, make_s1weak_loader
from src.data.io import clean_hand_mask
from src.data.s1augmentations import S1_MEAN, S1_STD

# +
# get params from segformer config
config_path = os.path.join(PROJECT_ROOT + "src/training/unet/config_unet.yml")
with open(config_path, "r") as f:
    config_dict = yaml.safe_load(f)
    config = Config(config_dict=config_dict)

batch_size = int(config.train.batch_size)
num_workers = int(config.train.num_workers)
# -

# parse args
args = parser.parse_args()
latest_run = args.run
if args.label_type:
    assert args.label_type == 'weak' or args.label_type == 'hand', 'label-type arg should be "weak" or "hand"'
    label_type = args.label_type
else:
    label_type = 'hand'
if label_type == 'hand':
    train_loader, val_loader, viz_loader = make_s1hand_loaders(
            DATA_ROOT, batch_size, num_workers
        )
else:
    viz_loader = make_s1weak_loader(DATA_ROOT, batch_size, num_workers)
if args.image_ind:
    print(f'Visualzing image {args.image_ind}')
    image_index = args.image_ind
else:
    image_index = random.randint(0, len(viz_loader.dataset))
    print(f'Visualizing random image, image {image_index} chosen')


# used to denorm images
def denorm(band: np.ndarray, mean: float, std: float) -> np.ndarray:
    return np.clip(band * std + mean, 0.0, 1.0)


# initiate model with desired weights
unet = FloodUNet(config)
unet_weights = torch.load(PROJECT_ROOT + 'src/training/unet/logs/' + latest_run + '/weights.pth')
unet.load_state_dict(unet_weights)

# showing inputs, hidden dims, and outputs
with torch.no_grad():
    unet.model.eval()
    
    # pull image
    image = torch.unsqueeze(viz_loader.dataset[image_index]['image'], dim = 0) # both vv and vh
    mask = torch.unsqueeze(viz_loader.dataset[image_index]['mask'], dim = 0)
    enc_out = unet.model.encoder(image)
    dec_out = unet.model.decoder(enc_out)
    pred_masks = unet.model.segmentation_head(dec_out)
    tot_images = 2 + len(enc_out) + dec_out.shape[0] + pred_masks.shape[1] + 2
    rows = int(np.ceil(tot_images/4))
    fig, axes = plt.subplots(rows, 4, figsize = (10,8))
    for ax in axes.flatten():
        ax.yaxis.set_visible(False)
        ax.xaxis.set_visible(False)
        ax.set_aspect("auto")
        
    ## vv
    vv = denorm(image[0, 0].cpu().numpy(), S1_MEAN[0], S1_STD[0])
    im0 = axes[0][0].imshow(vv, cmap = 'gray')
    axes[0][0].set_title('VV')
    
    # vh
    vh = denorm(image[0, 1].cpu().numpy(), S1_MEAN[0], S1_STD[0])
    im1 = axes[0][1].imshow(vh, cmap = 'gray')
    axes[0][1].set_title('VH')

    # encoder outputs
    for idx, encoder_out in enumerate(enc_out):
        row_index = int(np.floor(1 + idx + 1)/4)
        col_index = int((1+idx+ 1)%4)
        im = axes[row_index][col_index].imshow(encoder_out[0].mean(dim = 0))
        axes[row_index][col_index].set_title(f'Encoder Layer {idx + 1}')
    if col_index == 3:
        row_index += 1
        col_index = 0
    else:
        col_index += 1
        
    # decoder output
    im4 = axes[row_index][col_index].imshow(dec_out[0].mean(dim = 0))
    axes[row_index][col_index].set_title('Decoder Output')

    # logits
    if col_index == 3:
        row_index += 1
        col_index = 0
    else:
        col_index += 1
    im5 = axes[row_index][col_index].imshow(pred_masks[0][0])
    axes[row_index][col_index].set_title('Mask Logits Class 0')
    if col_index == 3:
        row_index += 1
        col_index = 0
    else:
        col_index += 1
    im6 = axes[row_index][col_index].imshow(pred_masks[0][1])
    axes[row_index][col_index].set_title('Mask Logits Class 1')

    # true mask
    if col_index == 3:
        row_index += 1
        col_index = 0
    else:
        col_index += 1
    mask_np = clean_hand_mask(mask.cpu().numpy())
    mask_vis = mask_np.astype(float)[0]
    mask_vis[mask_vis == 255] = np.nan
    im7 = axes[row_index][col_index].imshow(mask_vis, cmap = 'Reds')
    axes[row_index][col_index].set_title('True Mask')

    # predicted mask
    if col_index == 3:
        row_index += 1
        col_index = 0
    else:
        col_index += 1
    cleaned_pred = clean_hand_mask(torch.argmax(pred_masks, dim = 1)[0].cpu().numpy())
    im8 = axes[row_index][col_index].imshow(cleaned_pred, cmap = 'Reds')
    axes[row_index][col_index].set_title('Predicted Mask')
    plt.savefig(f'unet_viz_image{image_index}_{label_type}')

# pull and convolutional layer names
conv_keys = []
for key in unet.state_dict().keys():
    if 'conv' in key and 'weight' in key:
        conv_keys.append(key)

# kernel visualization
plt.figure()
row_index = 0
col_index = 0
tot_images = len(conv_keys)
rows = int(np.ceil(tot_images/8))
fig, axes = plt.subplots(rows, 8, figsize = (10,10))
for ax in axes.flatten():
    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)
    ax.set_aspect("auto")
for key in conv_keys:
    conv_weights = unet.state_dict()[key]
    try:
        viz_example_weight = conv_weights[0][0].cpu().numpy()
        img = axes[row_index][col_index].imshow(viz_example_weight)
        if col_index == 7:
            row_index += 1
            col_index = 0
        else:
            col_index += 1
    except IndexError:
        pass
cbar = fig.colorbar(img, ax=axes, shrink=0.5)
max_val = np.max(viz_example_weight)
min_val = np.min(viz_example_weight)
cbar.set_ticks([min_val, max_val])
cbar.set_ticklabels(['low', 'high'])
plt.savefig(f'kernel_viz_unet.png')
