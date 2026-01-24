import torch
from torch import nn
from torch.nn import functional as F
from .resnet import resnet50  # Import the existing modules resnet

import torch
import torch.nn as nn
import torch.nn.functional as F


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        # 1. Channel Attention Submodule
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared MLP for Channel Attention
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )

        # 2. Spatial Attention Submodule
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel Attention
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        channel_out = self.sigmoid(avg_out + max_out)
        x = x * channel_out

        # Spatial Attention
        avg_mask = torch.mean(x, dim=1, keepdim=True)
        max_mask, _ = torch.max(x, dim=1, keepdim=True)
        spatial_mask = torch.cat([avg_mask, max_mask], dim=1)
        spatial_out = self.sigmoid(self.spatial_conv(spatial_mask))
        x = x * spatial_out

        return x


"""class MultiScaleEnhancer(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        mid = in_channels // 4
        self.b1 = nn.Conv2d(in_channels, mid, 1)
        self.b2 = nn.Conv2d(in_channels, mid, 3, padding=1)
        self.b3 = nn.Conv2d(in_channels, mid, 3, padding=2, dilation=2)  # 5x5 field
        self.fuse = nn.Conv2d(mid * 3, in_channels, 1)
        self.bn = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        #out = torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1)
        out = torch.cat([self.b1(x), self.b2(x)], dim=1)
        return F.relu(self.bn(self.fuse(out)) + x)"""


class SAPH_MultiScale_CBAM(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # Each branch reduces channels to avoid a massive parameter explosion
        mid = in_channels // 4

        # Scale 1: 1x1 (Identity/Original)
        self.branch1 = nn.Conv2d(in_channels, mid, kernel_size=1)
        self.cbam1 = CBAM(mid)

        # Scale 2: 3x3 (Local)
        self.branch2 = nn.Conv2d(in_channels, mid, kernel_size=3, padding=1)
        self.cbam2 = CBAM(mid)

        # Scale 3: 5x5 (Global/Context - using dilation for efficiency)
        self.branch3 = nn.Conv2d(in_channels, mid, kernel_size=3, padding=2, dilation=2)
        self.cbam3 = CBAM(mid)

        # Fusion layer to bring mid back to in_channels
        self.fuse = nn.Conv2d(mid * 3, in_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        identity = x

        # Apply CBAM to each scale individually
        out1 = self.cbam1(self.branch1(x))
        out2 = self.cbam2(self.branch2(x))
        out3 = self.cbam3(self.branch3(x))

        # Concatenate and project
        feat = torch.cat([out1, out2, out3], dim=1)
        feat = self.fuse(feat)
        feat = self.bn(feat)

        # Final Element-wise Addition (Residual Connection)
        return F.relu(feat + identity)

# (Assume CBAM module is defined here as we did before)

class SAPH_ResNet50(nn.Module):
    def __init__(self, num_classes, loss='softmax', pretrained=True, **kwargs):
        super().__init__()
        # 1. Load the backbone from Torchreid (pretrained)
        kwargs.pop('pretrained', None)

        # 2. Load the backbone from Torchreid
        # We pass pretrained=pretrained explicitly here
        self.base = resnet50(
            num_classes=num_classes,
            loss=loss,
            pretrained=pretrained,
            **kwargs
        )
        # 2. Add your custom "SAPH" layers
        # Torchreid ResNet50 output is 2048 channels at the last conv layer
        #self.enhancer = MultiScaleEnhancer(2048)
        #self.cbam = CBAM(2048)  # Using the CBAM we designed
        self.saph_block = SAPH_MultiScale_CBAM(2048)
        # 3. Define the Global Head (for the CNN-only training phase)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(2048, num_classes)

    def forward(self, x):
        # Pass through ResNet up to the last convolutional layer
        # In modules's resnet50, features are accessible before the fc
        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = self.base.relu(x)
        x = self.base.maxpool(x)
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)

        # Apply the new parallel-attention enhancer
        x = self.saph_block(x)

        v = self.global_pool(x).view(x.size(0), -1)

        if not self.training:
            return v

        return self.classifier(v), v