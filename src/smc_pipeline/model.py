"""Segmentation network used for bright-field nuclear segmentation."""
import torch.nn as nn
import torch.nn.functional as F


class ConvEncoder(nn.Module):
    """Single strided convolution block (stride 4) producing a 256-channel feature map.

    Called ``DummySAMEncoder`` in the original research code. The attribute names
    (``encoder.conv``, ``decoder``) are kept so existing checkpoints load unchanged.
    """

    def __init__(self, out_channels: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, out_channels, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class SegmentationNet(nn.Module):
    """Encoder + light convolutional decoder returning per-pixel logits at input size."""

    def __init__(self, encoder: nn.Module, num_classes: int = 1):
        super().__init__()
        self.encoder = encoder
        for param in self.encoder.parameters():  # encoder frozen at the start of training
            param.requires_grad = False

        self.decoder = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, num_classes, kernel_size=2, stride=2),
        )

    def forward(self, x):
        out = self.decoder(self.encoder(x))
        return F.interpolate(out, size=x.shape[2:], mode="bilinear", align_corners=False)


def build_model(num_classes: int = 1) -> SegmentationNet:
    return SegmentationNet(ConvEncoder(256), num_classes=num_classes)
