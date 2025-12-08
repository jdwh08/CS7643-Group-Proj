# training/unet/model.py

import torch.nn as nn
import segmentation_models_pytorch as smp

NUM_CLASSES = 2


class FloodUNet(nn.Module):
    """
    U-Net with ResNet-50 encoder for Sentinel-1 VV/VH flood segmentation.
    """

    def __init__(self, config) -> None:
        super().__init__()

        self.model = smp.Unet(
            encoder_name=config.model.encoder_name,     
            encoder_weights=config.model.encoder_weights,  
            in_channels=2,                           
            classes=NUM_CLASSES,
        )

    def forward(self, x):
        return self.model(x)