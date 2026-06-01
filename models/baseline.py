import torch
import torch.nn as nn
from torchvision import models


class BaselineReID(nn.Module):
    def __init__(self, num_classes, feature_dim=2048, pretrained=True):
        super().__init__()

        if pretrained:
            weights = models.ResNet50_Weights.IMAGENET1K_V1
        else:
            weights = None

        resnet = models.resnet50(weights=weights)

        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        feat = self.backbone(x)
        feat = feat.view(feat.size(0), -1)
        logits = self.classifier(feat)

        return feat, logits
