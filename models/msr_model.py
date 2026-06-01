import torch
import torch.nn as nn
from torchvision import models


class MSRModel(nn.Module):
    def __init__(self, num_classes, num_views=6, pretrained=True):
        super().__init__()

        if pretrained:
            weights = models.ResNet50_Weights.IMAGENET1K_V1
        else:
            weights = None

        visible_resnet = models.resnet50(weights=weights)
        infrared_resnet = models.resnet50(weights=weights)

        self.visible_backbone = nn.Sequential(*list(visible_resnet.children())[:-1])
        self.infrared_backbone = nn.Sequential(*list(infrared_resnet.children())[:-1])

        feature_dim = 2048

        self.visible_classifier = nn.Linear(feature_dim, num_classes)
        self.infrared_classifier = nn.Linear(feature_dim, num_classes)
        self.shared_classifier = nn.Linear(feature_dim, num_classes)
        self.view_classifier = nn.Linear(feature_dim, num_views)

    def extract_visible(self, x):
        feat = self.visible_backbone(x)
        feat = feat.view(feat.size(0), -1)
        return feat

    def extract_infrared(self, x):
        feat = self.infrared_backbone(x)
        feat = feat.view(feat.size(0), -1)
        return feat

    def forward(self, rgb_imgs, ir_imgs):
        rgb_feat = self.extract_visible(rgb_imgs)
        ir_feat = self.extract_infrared(ir_imgs)

        outputs = {
            "rgb_feat": rgb_feat,
            "ir_feat": ir_feat,

            "rgb_logits": self.visible_classifier(rgb_feat),
            "ir_logits": self.infrared_classifier(ir_feat),

            "rgb_shared_logits": self.shared_classifier(rgb_feat),
            "ir_shared_logits": self.shared_classifier(ir_feat),

            "rgb_view_logits": self.view_classifier(rgb_feat),
            "ir_view_logits": self.view_classifier(ir_feat),
        }

        return outputs
