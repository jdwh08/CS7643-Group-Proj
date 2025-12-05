from transformers import (
    MaskFormerConfig,
    MaskFormerForInstanceSegmentation,
    MaskFormerImageProcessor,
)


import os
import torch

from tqdm.auto import tqdm
import numpy as np
import torch.nn as nn
import evaluate
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import yaml

import datetime
import sys


# Create paths for logging info and project root
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../.."))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.loaders import make_s1hand_loaders, make_s1weak_loader
from src.data.io import clean_hand_mask
from config import Config

S1_MEAN = [0.6851, 0.5235]
S1_STD = [0.0820, 0.1102]


class FloodMaskformer:
    """
    Maskformer module for non-pretrained Sentinel 1 inference.
    """

    def __init__(self, dataset: str = "weak"):

        self.config_path = os.path.join(BASE_DIR, "config_maskformer.yml")
        with open(self.config_path, "r") as file:
            self.config_dict = yaml.safe_load(file)
            self.config = Config(config_dict=self.config_dict)

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
            backbone_kwargs={
                "output_hidden_states": True,
                "out_indices": [
                    0,
                    1,
                    2,
                    3,
                ],  # pixel decoder wants feature maps at different stages
            },
        )

        self.model = MaskFormerForInstanceSegmentation(self.maskformer_config)

        self.save_path = ""
        feature_extractor = self.model.model.pixel_level_module.encoder

        # first layer expects 3 channels, replace with 2 channels for Sen-1
        feature_extractor.embedder.num_channels = 2
        feature_extractor.embedder.embedder.convolution = nn.Conv2d(
            2, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.model = self.convertBNtoGN(self.model)

        # print(self.model)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        print(f"Total trainable parameters: {trainable_params}")
        self.processor = MaskFormerImageProcessor(
            do_normalize=False,
            do_reduce_labels=False,
            do_resize=False,
            do_convert_rgb=False,
            do_rescale=False,
            ignore_index=255,
            num_labels=2,
            size=(256, 256),
        )

        self.hand_train_loader, self.val_loader, self.test_loader = make_s1hand_loaders(
            DATA_ROOT, self.batch_size, self.num_workers
        )

        self.train_loader = (
            make_s1weak_loader(DATA_ROOT, self.batch_size, self.num_workers)
            if dataset == "weak"
            else self.hand_train_loader
        )
        print("LOADER SIZES", len(self.train_loader), len(self.val_loader))
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            len(self.train_loader) * 10,
            T_mult=2,
            eta_min=0,
            last_epoch=-1,
        )
        self.train_metrics = evaluate.load("mean_iou")
        self.val_metrics = evaluate.load("mean_iou")

        # per epoch metrics
        self.train_loss_history = []
        self.val_loss_history = []
        self.train_iou_history = []
        self.val_iou_history = []

    def training(self):
        for epoch in range(self.n_epochs):
            print(f"epoch {epoch}")
            self.train()
            self.evaluate()

    def train(self):
        self.model.train()
        progress_bar = tqdm(self.train_loader)
        curr_epoch_loss = []
        epoch_loss = 0.0
        sample_count = 0
        for i, batch in enumerate(progress_bar):
            img = batch["image"]
            mask = batch["mask"]
            self.optimizer.zero_grad()
            mask_labels, class_labels = self.reshape_mask(img, mask)

            out = self.model.forward(
                pixel_values=img.to(self.device),  # (2,3,256, 256)
                mask_labels=[
                    labels.to(self.device) for labels in mask_labels
                ],  # (2, 2, 256, 256)
                class_labels=[labels.to(self.device) for labels in class_labels],  # (2)
            )

            loss = out.loss

            rescale_sizes = [(256, 256)] * img.shape[0]
            seg_mask = self.processor.post_process_semantic_segmentation(
                out, target_sizes=rescale_sizes
            )

            # seg_mask is a list, recombine into a batched tensor
            seg_mask = torch.stack(seg_mask, dim=0)
            self.train_metrics.add_batch(predictions=seg_mask, references=mask)
            epoch_loss += loss.item() * img.shape[0]
            sample_count += img.shape[0]
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

        epoch_iou = self.train_metrics.compute(
            num_labels=2,
            ignore_index=255,
        )
        self.train_loss_history.append(epoch_loss / sample_count)
        self.train_iou_history.append(epoch_iou["mean_iou"])

    def convertBNtoGN(self, module, num_groups=16):
        if isinstance(module, nn.BatchNorm2d):
            num_channels = module.num_features

            # Safety Check: GroupNorm requires num_groups to divide num_channels
            # If channels are too small (e.g. < 16), we reduce groups to 1 (LayerNorm style)
            groups = num_groups if num_channels % num_groups == 0 else 1

            new_layer = nn.GroupNorm(
                groups,
                num_channels,
                eps=module.eps,
                affine=module.affine,
                device=self.device,
            )

            # Copy weights (important if you switch to pretrained later)
            if module.affine:
                new_layer.weight.data = module.weight.data.clone().detach()
                new_layer.bias.data = module.bias.data.clone().detach()

            return new_layer

        for name, child in module.named_children():
            module.add_module(
                name, FloodMaskformer.convertBNtoGN(child, num_groups=num_groups)
            )

        return module

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        curr_epoch_loss = []
        progress_bar = tqdm(self.val_loader)
        epoch_loss = 0.0
        sample_count = 0
        for i, batch in enumerate(progress_bar):
            img = batch["image"]
            mask = batch["mask"]
            mask_labels, class_labels = self.reshape_mask(img, mask)

            out = self.model.forward(
                pixel_values=img.to(self.device),  # (2,3,256, 256)
                mask_labels=[
                    labels.to(self.device) for labels in mask_labels
                ],  # (2, 2, 256, 256)
                class_labels=[labels.to(self.device) for labels in class_labels],  # (2)
            )

            loss = out.loss
            rescale_sizes = [(256, 256)] * img.shape[0]
            seg_mask = self.processor.post_process_semantic_segmentation(
                out, target_sizes=rescale_sizes
            )

            # seg_mask is a list, recombine into a batched tensor
            seg_mask = torch.stack(seg_mask, dim=0)
            curr_epoch_loss.append(loss.item())
            epoch_loss += loss.item() * img.shape[0]
            sample_count += img.shape[0]
            # NOTE: May need to convert to numpy?
            self.val_metrics.add_batch(predictions=seg_mask, references=mask)

        epoch_iou = self.val_metrics.compute(
            num_labels=2,
            ignore_index=255,
        )
        self.val_loss_history.append(epoch_loss / sample_count)
        self.val_iou_history.append(epoch_iou["mean_iou"])

    @torch.no_grad()
    def inference(self):
        self.model.eval()
        batch_loss = []
        test_metrics = evaluate.load("mean_iou")
        progress_bar = tqdm(self.test_loader)
        for i, batch in enumerate(progress_bar):
            img = batch["image"]
            mask = batch["mask"]
            mask_labels, class_labels = self.reshape_mask(img, mask)

            out = self.model.forward(
                pixel_values=img.to(self.device),  # (2,3,256, 256)
                mask_labels=[
                    labels.to(self.device) for labels in mask_labels
                ],  # (2, 2, 256, 256)
                class_labels=[labels.to(self.device) for labels in class_labels],  # (2)
            )

            loss = out.loss
            rescale_sizes = [(256, 256)] * img.shape[0]
            seg_mask = self.processor.post_process_semantic_segmentation(
                out, target_sizes=rescale_sizes
            )
            batch_loss.append(loss.item())
            # seg_mask is a list, recombine into a batched tensor
            seg_mask = torch.stack(seg_mask, dim=0)
            test_metrics.add_batch(predictions=seg_mask, references=mask)

        epoch_iou = test_metrics.compute(
            num_labels=2,
            ignore_index=255,
        )
        print(f"Test loss: {sum(batch_loss)/len(batch_loss)}")
        print(f"Test IoU: {epoch_iou["mean_iou"]}")

    # @staticmethod
    # def get_iou(output, target):
    #     output = torch.argmax(output, dim=1).flatten()
    #     target = target.flatten()

    #     no_ignore = target.ne(255).cuda()
    #     output = output.masked_select(no_ignore)
    #     target = target.masked_select(no_ignore)
    #     intersection = torch.sum(output * target)
    #     union = torch.sum(target) + torch.sum(output) - intersection
    #     iou = (intersection + 0.0000001) / (union + 0.0000001)

    #     if iou != iou:
    #         print("failed, replacing with 0")
    #         iou = torch.tensor(0).float()

    #     return iou
    @staticmethod
    def denorm(band, mean, std):
        return np.clip(band * std + mean, 0.0, 1.0)

    def save_model(
        self,
    ):
        curr_time = datetime.datetime.now()
        format_time = "%Y%m%d_%H:%M"
        timestamp = curr_time.strftime(format_time)
        self.save_path = os.path.join(LOG_DIR, f"maskformer_run_{timestamp}")
        os.makedirs(self.save_path, exist_ok=True)
        state_dict_path = os.path.join(self.save_path, "weights.pth")
        torch.save(self.model.state_dict(), state_dict_path)
        self.generate_learning_curves()
        self.save_metrics()
        self.generate_plot()

    def generate_learning_curves(self):
        plt.plot(self.train_loss_history, label="Training")
        plt.plot(self.val_loss_history, label="Validation")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss")
        plt.title("Loss Learning Curve")
        plt.legend()
        plt.grid()
        plt.tight_layout()

        loss_path = os.path.join(self.save_path, "loss_curve.png")
        plt.savefig(loss_path)
        plt.close()

        plt.plot(self.train_iou_history, label="Training")
        plt.plot(self.val_iou_history, label="Validation")
        plt.xlabel("Epoch #")
        plt.ylabel("Mean IoU")
        plt.title("Mean IoU Learning Curve")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        iou_path = os.path.join(self.save_path, "iou_curve.png")
        plt.savefig(iou_path)
        plt.close()

    def save_metrics(self):
        try:

            file_name = os.path.join(self.save_path, "h_params_config.yml")
            with open(file_name, "w") as file:

                metric_dict = {
                    **self.config_dict,
                    "Loss": self.val_loss_history[-1],
                    "IoU": self.val_iou_history[-1].item(),
                }
                yaml.dump(metric_dict, file, default_flow_style=False, sort_keys=False)

            print(f"\nSuccessfully wrote data to {file_name}")
        except Exception as e:
            print(f"An error occurred: {e}")

    @torch.no_grad()
    def generate_plot(self):
        """
        saves image, target mask, and predicted mask for visualization.
        Uses first image/mask from validation set.
        """
        self.model.eval()
        iterator = iter(self.val_loader)

        batch = next(iterator)
        test_img, mask = batch["image"], batch["mask"]
        curr_img = test_img[2]
        curr_mask = mask[2]
        out = self.model.forward(pixel_values=curr_img.unsqueeze(0).to(self.device))

        # Post Process

        result = self.processor.post_process_semantic_segmentation(out)
        vv = curr_img[0]
        vh = curr_img[1]
        vv_denorm = FloodMaskformer.denorm(vv, S1_MEAN[0], S1_STD[0])
        vh_denorm = FloodMaskformer.denorm(vh, S1_MEAN[1], S1_STD[1])
        seg_mask = result[0]

        mask = clean_hand_mask(curr_mask.cpu().detach().numpy())
        mask_vis = mask.astype(float)
        mask_vis[mask_vis == 255] = np.nan
        pred_mask = clean_hand_mask(seg_mask.cpu().detach().numpy())
        pred_mask_vis = pred_mask.astype(float)
        pred_mask_vis[pred_mask_vis == 255] = np.nan
        fig, ax = plt.subplots(1, 4, figsize=(18, 5))
        ax[0].imshow(vv_denorm, cmap="gray")
        ax[0].set_title("VV")

        ax[1].imshow(vh_denorm, cmap="gray")
        ax[1].set_title("VH")

        ax[2].imshow(mask_vis, cmap="Reds", vmin=0, vmax=1)
        ax[2].set_title("Hand Mask (cleaned)")

        ax[3].imshow(pred_mask_vis, cmap="Reds", vmin=0, vmax=1)
        ax[3].set_title("Predicted Mask (cleaned)")
        plot_path = os.path.join(self.save_path, "visualization.png")
        plt.savefig(plot_path)
        plt.close()

    def reshape_mask(self, imgs, masks):
        """


        Transformer mask_labels param requires shape (B,N,H,W), where N is the number of labels (binary masks).


        This function does the (B,H,W) to (B,N,H,W) transform and also generates class_labels.


        """

        img_list = [img for img in imgs]

        # print("SHAPE", img_list[0].shape)

        mask_list = [mask for mask in masks]
        processed = self.processor(
            images=img_list,
            segmentation_maps=mask_list,
            return_tensors="pt",
            input_data_format="channels_first",
        )

        return processed["mask_labels"], processed["class_labels"]

    def load_weights(self, model_dir: str):
        model_path = os.path.join(LOG_DIR, model_dir, "weights.pth")

        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        else:
            print(f"Error:'{model_path}' was not found.")

    # def draw_semantic_segmentation(self, segmentation):
    #     # get the used color map
    #     viridis = cm.get_cmap("viridis", torch.max(segmentation))
    #     # get all the unique numbers
    #     labels_ids = torch.unique(segmentation).tolist()
    #     fig, ax = plt.subplots()
    #     ax.imshow(segmentation)
    #     handles = []
    #     for label_id in labels_ids:
    #         label = self.model.config.id2label[label_id]
    #         color = viridis(label_id)
    #         handles.append(mpatches.Patch(color=color, label=label))
    #     ax.legend(handles=handles)
    #     plt.show()
    #     return fig
