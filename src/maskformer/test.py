from transformers import (
    MaskFormerConfig,
    ResNetConfig,
    MaskFormerForInstanceSegmentation,
    ResNetModel,
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

# from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib import cm

# https://huggingface.co/docs/transformers/main/en/tasks/training_vision_backbone

# https://huggingface.co/docs/transformers/main/model_doc/maskformer

# https://pyimagesearch.com/2023/03/13/train-a-maskformer-segmentation-model-with-hugging-face-transformers/


# TODO: Check out indices - try more feature maps or less?
config = MaskFormerConfig(
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
        ],  # Request features from all 4 stages (strides 4, 8, 16, 32)
    },
)

model = MaskFormerForInstanceSegmentation(config)


feature_extractor = model.model.pixel_level_module.encoder


# first layer expects 3 channels, replace with 2 channels for Sen-1
feature_extractor.embedder.num_channels = 2
feature_extractor.embedder.embedder.convolution = nn.Conv2d(
    2, 64, kernel_size=7, stride=2, padding=3, bias=False
)


# for (
#     param
# ) in model.model.pixel_level_module.encoder.parameters():  # freezes backbone weights
#     param.requires_grad = False


# TODO: Look into loss metric
# TODO: may need to normalize 3rd channel

# # batch size is 16 - converts batch norm to group norm layers?
# processor = MaskFormerImageProcessor(
#     do_resize=False,
#     do_rescale=False,
#     num_labels=2,
#     size=(256, 256),
#     do_normalize=True,
#     image_mean=[0.6851, 0.5235],
#     image_std=[0.0820, 0.1102],
#     size_divisor=32,
# )

# processor()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
PROJECT_ROOT = os.path.abspath(".")
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")

optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
aug_ds = S1WeakDataset(
    data_root=DATA_ROOT,
    max_samples=2,
    transform=get_train_transform(image_size=256),
)


val_ds = S1WeakDataset(
    data_root=DATA_ROOT, max_samples=2, transform=get_val_transform(image_size=256)
)
BATCH_SIZE = 2

# TODO: Change num workers?
aug_loader = DataLoader(aug_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# def preprocess_sar(img_tensor):
#     """
#     Converts a 2-channel VV/VH tensor (2, H, W) to a 3-channel tensor (3, H, W)
#     by calculating the VV/VH ratio as the third channel.
#     """

#     if img_tensor.shape[0] != 2:
#         raise ValueError(
#             f"Input image tensor must be (2, H, W). Found shape: {img_tensor.shape}"
#         )

#     VV = img_tensor[0, :, :]
#     VH = img_tensor[1, :, :]

#     # epsilon to prevent divide by zero
#     epsilon = 1e-6
#     ratio = VV / (VH + epsilon)

#     stacked_bands = torch.stack([VV, VH, ratio], dim=0)

#     return stacked_bands


def reshape_mask(imgs, masks):
    """
    Transformer mask_labels param requires shape (B,N,H,W), where N is the number of labels (binary masks).
    This function does the (B,H,W) to (B,N,H,W) transform and also generates class_labels.
    """

    # img_list = [preprocess_sar(img) for img in imgs]

    # print("SHAPE", img_list[0].shape)
    mask_list = [mask for mask in masks]
    binary_mask_list = []  # (b, 2, h, w)

    for mask in mask_list:
        binary_masks = []

        for id in range(0, 2):
            binary_mask = (mask == id).float()
            binary_masks.append(binary_mask)
        binary_tensor = torch.stack(binary_masks, dim=0)
        binary_mask_list.append(binary_tensor)

    # processor = MaskFormerImageProcessor(
    #     do_normalize=False,
    #     do_reduce_labels=False,
    #     do_resize=False,
    #     do_rescale=False,
    #     ignore_index=255,
    #     num_labels=2,
    #     size=(256, 256),
    # )
    # processed = processor(
    #     images=img_list, segmentation_maps=mask_list, return_tensors="pt"
    # )
    class_labels = [torch.tensor([0, 1]) for _ in range(BATCH_SIZE)]  # (b,2)
    return binary_mask_list, class_labels


n_epochs = 2
train_loss, val_loss = [], []
for epoch in range(n_epochs):
    model.train()

    progress_bar = tqdm(aug_loader)
    for i, (img, mask) in enumerate(progress_bar):
        optimizer.zero_grad()
        mask_labels, class_labels = reshape_mask(img, mask)

        out = model.forward(
            pixel_values=img.to(device),  # (2,3,256, 256)
            mask_labels=[
                labels.to(device) for labels in mask_labels
            ],  # (2, 2, 256, 256)
            class_labels=[labels.to(device) for labels in class_labels],  # (2)
        )

        loss = out.loss
        train_loss.append(loss.item())
        loss.backward()
        optimizer.step()

        # Average train epoch loss

    print(f"Train loss for epoch: {epoch}, {sum(train_loss) / len(train_loss)}")

    model.eval()

    progress_bar = tqdm(val_loader)
    for i, (img, mask) in enumerate(progress_bar):
        with torch.no_grad():
            mask_labels, class_labels = reshape_mask(img, mask)

            out = model.forward(
                pixel_values=img.to(device),  # (2,3,256, 256)
                mask_labels=[
                    labels.to(device) for labels in mask_labels
                ],  # (2, 2, 256, 256)
                class_labels=[labels.to(device) for labels in class_labels],  # (2)
            )

            loss = out.loss
            val_loss.append(loss.item())
    print(f"VAl loss for epoch: {epoch}, {sum(val_loss)/len(val_loss)}")


# Inference

model.eval()
with torch.no_grad():
    iterator = iter(val_loader)
    test_img, mask = next(iterator)

    out = model.forward(pixel_values=test_img[0].unsqueeze(0).to(device))

    # Post Process

    processor = MaskFormerImageProcessor(
        do_normalize=False,
        do_reduce_labels=False,
        do_resize=False,
        do_rescale=False,
        ignore_index=255,
        num_labels=2,
        size=(256, 256),
    )

    result = processor.post_process_semantic_segmentation(out)

    seg_mask = result[0]


def draw_semantic_segmentation(segmentation):
    # get the used color map
    viridis = cm.get_cmap("viridis", torch.max(segmentation))
    # get all the unique numbers
    labels_ids = torch.unique(segmentation).tolist()
    fig, ax = plt.subplots()
    ax.imshow(segmentation)
    handles = []
    for label_id in labels_ids:
        label = model.config.id2label[label_id]
        color = viridis(label_id)
        handles.append(mpatches.Patch(color=color, label=label))
    ax.legend(handles=handles)
    plt.show()
    return fig


draw_semantic_segmentation(seg_mask)
# COLOR_MAP = {
#     0: [0.0, 1.0, 0.0],
#     1: [0.0, 0.0, 1.0],
# }

# color_image = np.zeros((seg_mask.shape[0], seg_mask.shape[1], 3), dtype=np.float32)

# # Iterate through your class IDs and apply the corresponding color
# for class_id, color in COLOR_MAP.items():
#     # Find all pixels where the semantic map value equals the class_id
#     # 'mask' is a boolean array (H, W) where True means the pixel belongs to this class
#     mask = seg_mask == class_id

#     # Use the mask to set the RGB values in the color image
#     color_image[mask] = color
# plt.figure(figsize=(8, 8))
# plt.imshow(color_image)
# plt.title(f"Semantic Segmentation Map (Predicted)")
# plt.axis("off")  # Hide axes for cleaner image
# plt.show()
