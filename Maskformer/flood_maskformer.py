from transformers import (
    MaskFormerConfig,
    MaskFormerForInstanceSegmentation,
    MaskFormerImageProcessor,
)

from PIL import Image
import requests
import os
import torch
from src.data.s1weak import S1WeakDataset
from torch.utils.data import DataLoader
from src.data.augmentations import get_train_transform, get_val_transform
from tqdm.auto import tqdm
import numpy as np
import torch.nn as nn
import evaluate
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import yaml
from config import Config

from src.data.loaders import make_s1hand_loaders, make_s1weak_loader
# from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib import cm


class FloodMaskformer:
    """
    Maskformer module for non-pretrained Sentinel 1 inference.
    """

    def __init__(
        self, dataset="weak"
    ):
        with open("", "r") as file:
            config_dict = yaml.safe_load(file)
            self.config = Config(config_dict=config_dict)

        self.batch_size = self.config.train.batch_size
        self.num_workers = self.config.train.num_workers
        self.lr = self.config.train.lr
        self.n_epochs = self.config.train.n_epochs
        self.weight_decay = self.config.optimizer.weight_decay
        self.maskformer_config = MaskFormerConfig(
            backbone="microsoft/resnet-50",
            use_pretrained_backbone=False,
            num_labels=2,
            id2label={i: str(i) for i in range(2)},
            label2id={str(i): i for i in range(2)},
            num_channels=2,
            backbone_kwargs={ #TODO: Should I make this a hyperparam? 
                "output_hidden_states": True,
                "out_indices": [
                    0,
                    1,
                    2,
                    3,
                ],  # Request features from all 4 stages (strides 4, 8, 16, 32)
            },
        )

        self.model = MaskFormerForInstanceSegmentation(self.maskformer_config)

        feature_extractor = self.model.model.pixel_level_module.encoder

        # first layer expects 3 channels, replace with 2 channels for Sen-1
        feature_extractor.embedder.num_channels = 2
        feature_extractor.embedder.embedder.convolution = nn.Conv2d(
            2, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.PROJECT_ROOT = os.path.abspath(".")
        self.DATA_ROOT = os.path.join(self.PROJECT_ROOT, "data")
        self.hand_train_loader, self.val_loader, self.test_loader = make_s1hand_loaders(self.DATA_ROOT, self.batch_size, self.num_workers)
  
        self.train_loader = make_s1weak_loader(self.DATA_ROOT, self.batch_size, self.num_workers) if dataset == "weak" else self.hand_train_loader


        self.metrics = evaluate.load("mean_iou")
        self.train_loss = []
        self.val_loss = []
        self.train_iou = []
        self.val_iou = []


    def training(self):
        for epoch in range(self.n_epochs):
                pass

    def train(self):
        self.model.train()
        progress_bar = tqdm(self.train_loader)
        curr_epoch_loss = []
        curr_epoch_iou = []
        for i, (img, mask) in enumerate(progress_bar):
            self.optimizer.zero_grad()
            mask_labels, class_labels = self.reshape_mask(mask)

            out = self.model.forward(
                pixel_values=img.to(self.device),  # (2,3,256, 256)
                mask_labels=[
                    labels.to(self.device) for labels in mask_labels
                ],  # (2, 2, 256, 256)
                class_labels=[labels.to(self.device) for labels in class_labels],  # (2)
            )

            loss = out.loss
            curr_epoch_loss.append(loss.item())
            curr_epoch_iou.append()
            loss.backward()
            self.optimizer.step()
        self.train_loss.append(sum(curr_epoch_loss)/len(curr_epoch_loss))
        self.train_iou.append(sum(curr_epoch_iou)/len(curr_epoch_iou))
                              
    def evaluate(self):
        self.model.eval()
        progress_bar = tqdm(self.val_loader)
        with torch.no_grad():
            for i, (img, mask) in enumerate(progress_bar):
            
                mask_labels, class_labels = self.reshape_mask(mask)

                out = self.model.forward(
                    pixel_values=img.to(self.device),  # (2,3,256, 256)
                    mask_labels=[
                        labels.to(self.device) for labels in mask_labels
                    ],  # (2, 2, 256, 256)
                    class_labels=[labels.to(self.device) for labels in class_labels],  # (2)
                )

                loss = out.loss
                self.val_loss.append(loss.item())
                self.val_iou.append()

    @staticmethod
    def get_iou(output, target):
        output = torch.argmax(output, dim=1).flatten() 
        target = target.flatten()
        
        no_ignore = target.ne(255).cuda()
        output = output.masked_select(no_ignore)
        target = target.masked_select(no_ignore)
        intersection = torch.sum(output * target)
        union = torch.sum(target) + torch.sum(output) - intersection
        iou = (intersection + .0000001) / (union + .0000001)
        
        if iou != iou:
            print("failed, replacing with 0")
            iou = torch.tensor(0).float()
        
        return iou

    def reshape_mask(self, masks):
        """
        Transformer mask_labels param requires shape (B,N,H,W), where N is the number of labels (binary masks).
        This function does the (B,H,W) to (B,N,H,W) transform and also generates class_labels.
        """
        mask_list = [mask for mask in masks]
        binary_mask_list = []  # (b, 2, h, w)

        for mask in mask_list:
            binary_masks = []

            for id in range(0, 2):
                binary_mask = (mask == id).float()
                binary_masks.append(binary_mask)
            binary_tensor = torch.stack(binary_masks, dim=0)
            binary_mask_list.append(binary_tensor)

        class_labels = [torch.tensor([0, 1]) for _ in range(self.batch_size)]  # (b,2)
        return binary_mask_list, class_labels

    def draw_semantic_segmentation(self, segmentation):
        # get the used color map
        viridis = cm.get_cmap("viridis", torch.max(segmentation))
        # get all the unique numbers
        labels_ids = torch.unique(segmentation).tolist()
        fig, ax = plt.subplots()
        ax.imshow(segmentation)
        handles = []
        for label_id in labels_ids:
            label = self.model.config.id2label[label_id]
            color = viridis(label_id)
            handles.append(mpatches.Patch(color=color, label=label))
        ax.legend(handles=handles)
        plt.show()
        return fig
