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

parser = argparse.ArgumentParser(description = 'used to select test image you was to visualize forward pass for')
parser.add_argument("--image-ind", type = int, required = False, help = 'index of image in test dataloader')
parser.add_argument("--run", type = str, required = True, help = 'pass maskformer run used by load_weights method')

# +
PROJECT_ROOT = os.path.expanduser("~/scratch/CS7643-Group-Proj/")
print("PROJECT_ROOT:", PROJECT_ROOT)

# add to python path
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
# -

# using maskformer class
from src.maskformer.flood_maskformer import FloodMaskformer
from src.data.loaders import make_s1hand_loaders, make_s1weak_loader
from config import Config

# +
config_path = os.path.join(PROJECT_ROOT, "src/maskformer/config_maskformer.yml")
#print(config_path)
with open(config_path, "r") as file:
    config_dict = yaml.safe_load(file)
    config = Config(config_dict=config_dict)

batch_size = config.train.batch_size
num_workers = config.train.num_workers
# -

train_loader, val_loader, test_loader = make_s1hand_loaders(
            DATA_ROOT, batch_size, num_workers
        )

args = parser.parse_args()
latest_run = args.run
if args.image_ind:
    print(f'Visualzing image {args.image_ind}')
    image_index = args.image_ind
else:
    image_index = random.randint(0, len(test_loader.dataset))
    print(f'Visualizing random image, image {image_index} chosen')

maskformer = FloodMaskformer()
maskformer.load_weights(latest_run)

# showing inputs, hidden dims, and outputs
with torch.no_grad():
    maskformer.model.eval()
    # pull image
    image = torch.unsqueeze(test_loader.dataset[image_index]['image'], dim = 0) # both vv and vh
    mask = torch.unsqueeze(test_loader.dataset[image_index]['mask'], dim = 0)
    mask_labels, class_labels =  maskformer.reshape_mask(image, mask)
    out = maskformer.model.forward(pixel_values = image.to(maskformer.device), mask_labels = [labels.to(maskformer.device) for labels in mask_labels]\
                             , class_labels=[labels.to(maskformer.device) for labels in class_labels]\
                            , output_hidden_states = True)
    encoder_hidden_states = out['encoder_hidden_states']
    pixel_decoder_states = out['pixel_decoder_hidden_states']
    fix, axes = plt.subplots(3, 4, figsize = (10,8))
    for ax in axes.flatten():
        ax.yaxis.set_visible(False)
        ax.xaxis.set_visible(False)
        ax.set_aspect("auto")
    ## vv
    axes[0][0].imshow(image[0][0], aspect = 'auto')
    axes[0][0].set_title('VV Mask')
    # vh
    axes[0][1].imshow(image[0][1], aspect = 'auto')
    axes[0][1].set_title('VH Mask')
    #mask
    # batch_size x 2 x h x w
    axes[0][2].imshow(mask_labels[0][0], aspect = 'auto')
    axes[0][2].set_title('Class 0 Mask')
    axes[0][3].imshow(mask_labels[0][1], aspect = 'auto')
    axes[0][3].set_title('Class 1 Mask')
    # hidden_states x batch_size x channels x h x w
    axes[1][0].imshow(encoder_hidden_states[0][0].mean(dim = 0).cpu().numpy(), aspect = 'auto')
    axes[1][0].set_title('Encoder Layer 1')
    axes[1][1].imshow(encoder_hidden_states[1][0].mean(dim = 0).cpu().numpy(), aspect = 'auto')
    axes[1][1].set_title('Encoder Layer 2')
    axes[1][2].imshow(encoder_hidden_states[2][0].mean(dim = 0).cpu().numpy(), aspect = 'auto')
    axes[1][2].set_title('Encoder Layer 3')
    axes[1][3].imshow(encoder_hidden_states[3][0].mean(dim = 0).cpu().numpy(), aspect = 'auto')
    axes[1][3].set_title('Encoder Layer 4')
    # pixel deccoders
    axes[2][0].imshow(pixel_decoder_states[0][0].mean(dim = 0).cpu().numpy(), aspect = 'auto')
    axes[2][0].set_title('Pixel Decoder Layer 1')
    axes[2][1].imshow(pixel_decoder_states[1][0].mean(dim = 0).cpu().numpy(), aspect = 'auto')
    axes[2][1].set_title('Pixel Decoder Layer 2')
    axes[2][2].imshow(pixel_decoder_states[2][0].mean(dim = 0).cpu().numpy(), aspect = 'auto')
    axes[2][2].set_title('Pixel Decoder Layer 3')
    # predicted mask
    pred_mask = maskformer.processor.post_process_semantic_segmentation(out, target_sizes = [(256, 256)])
    #print(pred_mask)
    #pred_mask = torch.stack(pred_mask, dim = 0)
    #print(pred_mask)
    axes[2][3].imshow(pred_mask[0].cpu().numpy(), aspect ='auto')
    axes[2][3].set_title('Predicted Mask')
    plt.savefig(f'viz_test_image{image_index}.png')
