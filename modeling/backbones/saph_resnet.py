import torch
from torch import nn
from torch.nn import functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )

        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        channel_out = self.sigmoid(avg_out + max_out)
        x = x * channel_out

        avg_mask = torch.mean(x, dim=1, keepdim=True)
        max_mask, _ = torch.max(x, dim=1, keepdim=True)
        spatial_mask = torch.cat([avg_mask, max_mask], dim=1)
        spatial_out = self.sigmoid(self.spatial_conv(spatial_mask))
        x = x * spatial_out

        return x




class SAPH_MultiScale_CBAM(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        mid = in_channels // 4

        self.branch1 = nn.Conv2d(in_channels, mid, kernel_size=1)
        self.cbam1 = CBAM(mid)

        self.branch2 = nn.Conv2d(in_channels, mid, kernel_size=3, padding=1)
        self.cbam2 = CBAM(mid)

        self.branch3 = nn.Conv2d(in_channels, mid, kernel_size=3, padding=2, dilation=2)
        self.cbam3 = CBAM(mid)

        self.fuse = nn.Conv2d(mid * 3, in_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        identity = x

        out1 = self.cbam1(self.branch1(x))
        out2 = self.cbam2(self.branch2(x))
        out3 = self.cbam3(self.branch3(x))

        feat = torch.cat([out1, out2, out3], dim=1)
        feat = self.fuse(feat)
        feat = self.bn(feat)

        # Final Element-wise Addition (Residual Connection)
        return F.relu(feat + identity)


class SAPH_ResNet50(nn.Module):
    def __init__(self, num_classes, loss='softmax', pretrained=True, **kwargs):
        super().__init__()
        kwargs.pop('pretrained', None)

        self.base = resnet50(
            num_classes=num_classes,
            loss=loss,
            pretrained=pretrained,
            **kwargs
        )

        self.saph_block = SAPH_MultiScale_CBAM(2048)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = self.base.relu(x)
        x = self.base.maxpool(x)
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)

        x = self.saph_block(x)

        v = self.global_pool(x).view(x.size(0), -1)

        if not self.training:
            return v

        return self.classifier(v), v
