"""个体分类器骨干网络构建（训练与推理共用）。"""

from __future__ import annotations

import torch.nn as nn
from torchvision import models

from paths import ensure_torch_home

ensure_torch_home()


def build_model(arch: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    arch = arch.lower().strip()
    if arch == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet50(weights=weights)
        for param in model.parameters():
            param.requires_grad = False
        nf = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(nf, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )
        for param in model.fc.parameters():
            param.requires_grad = True
        return model

    if arch == "efficientnet_v2_s":
        weights = (
            models.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
        )
        model = models.efficientnet_v2_s(weights=weights)
        for param in model.parameters():
            param.requires_grad = False
        in_f = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_f, num_classes),
        )
        for param in model.classifier.parameters():
            param.requires_grad = True
        return model

    raise ValueError(f"不支持的 arch: {arch}，可选: resnet50, efficientnet_v2_s")


def set_backbone_trainable(model: nn.Module, arch: str, trainable: bool) -> None:
    arch = arch.lower().strip()
    if arch == "resnet50":
        for param in model.parameters():
            param.requires_grad = trainable
        for param in model.fc.parameters():
            param.requires_grad = True
        return
    if arch == "efficientnet_v2_s":
        for param in model.features.parameters():
            param.requires_grad = trainable
        for param in model.classifier.parameters():
            param.requires_grad = True
        return
    raise ValueError(arch)
