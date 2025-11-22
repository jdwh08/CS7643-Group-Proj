from transformers import (
    MaskFormerConfig,
    ResNetConfig,
    MaskFormerForInstanceSegmentation,
    ResNetModel,
    MaskFormerImageProcessor,
)

from PIL import Image
import requests

import torch

# https://huggingface.co/docs/transformers/main/en/tasks/training_vision_backbone

# https://huggingface.co/docs/transformers/main/model_doc/maskformer

# https://pyimagesearch.com/2023/03/13/train-a-maskformer-segmentation-model-with-hugging-face-transformers/


# NOTE: Sen1floods11 dataset has 3 labels (no data is -1), not water is 0, water is 1. May need to account for this.

config = MaskFormerConfig(
    backbone="microsoft/resnet-50",
    use_pretrained_backbone=True,
    num_labels=2,
    id2label={i: str(i) for i in range(2)},
    label2id={str(i): i for i in range(2)},
)
model = MaskFormerForInstanceSegmentation(config)

for param in model.model.pixel_level_module.parameters():  # freezes backbone weights
    param.requires_grad = False


# batch size is 16 - converts batch norm to group norm layers?
processor = MaskFormerImageProcessor(
    do_resize=False,
    do_rescale=False,
    num_labels=2,
    size=(256, 256),
    do_normalize=True,
    image_mean=[0.6851, 0.5235],
    image_std=[0.0820, 0.1102],
    size_divisor=32,
)

processor()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
