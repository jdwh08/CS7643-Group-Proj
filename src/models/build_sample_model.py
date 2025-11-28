LR = 5e-4
EPOCHS = 100
EPOCHS_PER_UPDATE = 1
RUNNAME = "Sen1Floods11"

import torch
from torchvision import transforms
import torchvision.transforms.functional as F
import random
from PIL import Image
from time import time
import csv
import os
import numpy as np
import rasterio
from tqdm.notebook import tqdm
from IPython.display import clear_output
import os
from IPython.display import display
import matplotlib.pyplot as plt
import torchvision.models as models
import torch.nn as nn
import sys

# +
PROJECT_ROOT = os.path.expanduser("~/CS7643-Group-Proj/")
print("PROJECT_ROOT:", PROJECT_ROOT)

# add to python path
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
# -

from src.data.loaders import make_s1hand_loaders

if torch.cuda.is_available:
    device = 'cuda'
else:
    device = 'cpu'

# +
print(f'using {device}')

train_loader, val_loader, test_loader = make_s1hand_loaders(
    data_root=DATA_ROOT,
    batch_size=4,
    num_workers=0,
)

# +
net = models.segmentation.fcn_resnet50(pretrained=False, num_classes=2, pretrained_backbone=False)
net.backbone.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)

criterion = nn.CrossEntropyLoss(weight=torch.tensor([1,8]).float().to(device), ignore_index=255)
optimizer = torch.optim.AdamW(net.parameters(),lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, len(train_loader) * 10, T_mult=2, eta_min=0, last_epoch=-1)

def convertBNtoGN(module, num_groups=16):
    if isinstance(module, torch.nn.modules.batchnorm.BatchNorm2d):
        return nn.GroupNorm(num_groups, module.num_features,
                            eps=module.eps, affine=module.affine)
        if module.affine:
            mod.weight.data = module.weight.data.clone().detach()
            mod.bias.data = module.bias.data.clone().detach()

    for name, child in module.named_children():
        module.add_module(name, convertBNtoGN(child, num_groups=num_groups))

    return module

net = convertBNtoGN(net)


# -

def train_loop(inputs, labels, net, optimizer, scheduler):
    global running_loss
    global running_iou
    global running_count
    global running_accuracy

    # zero the parameter gradients
    optimizer.zero_grad()
    net = net.to(device)

    # forward + backward + optimize
    outputs = net(inputs.to(device))
    loss = criterion(outputs["out"], labels.long().to(device))
    loss.backward()
    optimizer.step()
    scheduler.step()
def train_epoch(net, optimizer, scheduler, train_iter):
    for (inputs, labels) in train_iter:
        train_loop(inputs.to(device), labels.to(device), net.to(device), optimizer, scheduler)


def train(net, optimizer, scheduler, train_loader, num_epochs):
    for i in range(num_epochs):
        print(f'epoch: {i}')
        train_iter = iter(train_loader)
        #print(train_iter)
        train_epoch(net, optimizer, scheduler, train_iter)
    return net


trained_net = train(net, optimizer, scheduler, train_loader, 5)
