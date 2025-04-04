import segmentation_models_pytorch as smp

from torch import nn


class VesselSegmentor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = smp.UnetPlusPlus(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            activation='sigmoid'
        )

    def forward(self, x):
        return self.model(x)


class OpticDiscSegmentor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = smp.UnetPlusPlus(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            activation='sigmoid'
        )

    def forward(self, x):
        return self.model(x)
