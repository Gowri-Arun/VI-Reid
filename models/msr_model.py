
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class MSRReID(nn.Module):
    """
    Warm-startable MSR-style two-branch model.

    Branches:
    - RGB branch for visible images
    - IR branch for infrared images

    Each branch has:
    - ResNet-50 backbone
    - embedding layer

    Shared part:
    - shared_fc maps modality-specific embedding into shared space
    - shared_classifier predicts identity from shared feature

    Classifiers:
    - rgb_classifier for RGB branch identity loss
    - ir_classifier for IR branch identity loss
    - shared_classifier for modality-shared identity loss
    """

    def __init__(self, num_classes: int, feature_dim: int = 512, pretrained: bool = True):
        super().__init__()

        weights = models.ResNet50_Weights.DEFAULT if pretrained else None

        rgb_resnet = models.resnet50(weights=weights)
        ir_resnet = models.resnet50(weights=weights)

        self.rgb_backbone = nn.Sequential(*list(rgb_resnet.children())[:-1])
        self.ir_backbone = nn.Sequential(*list(ir_resnet.children())[:-1])

        self.rgb_embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
        )

        self.ir_embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
        )

        self.shared_fc = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
        )

        self.rgb_classifier = nn.Linear(feature_dim, num_classes)
        self.ir_classifier = nn.Linear(feature_dim, num_classes)
        self.shared_classifier = nn.Linear(feature_dim, num_classes)

    def forward_rgb(self, x):
        x = self.rgb_backbone(x)
        specific_feat = self.rgb_embedding(x)
        shared_feat = self.shared_fc(specific_feat)

        rgb_logits = self.rgb_classifier(specific_feat)
        shared_logits = self.shared_classifier(shared_feat)

        return rgb_logits, shared_logits, specific_feat, shared_feat

    def forward_ir(self, x):
        x = self.ir_backbone(x)
        specific_feat = self.ir_embedding(x)
        shared_feat = self.shared_fc(specific_feat)

        ir_logits = self.ir_classifier(specific_feat)
        shared_logits = self.shared_classifier(shared_feat)

        return ir_logits, shared_logits, specific_feat, shared_feat

    def forward(self, x, modality):
        """
        modality:
            "rgb" or "ir"
        """
        if modality == "rgb":
            return self.forward_rgb(x)
        elif modality == "ir":
            return self.forward_ir(x)
        else:
            raise ValueError(f"Unknown modality: {modality}")

    @torch.no_grad()
    def extract_features(self, x, modality: str, normalize: bool = True):
        if modality == "rgb":
            _, _, _, shared_feat = self.forward_rgb(x)
        elif modality == "ir":
            _, _, _, shared_feat = self.forward_ir(x)
        else:
            raise ValueError(f"Unknown modality: {modality}")

        if normalize:
            shared_feat = F.normalize(shared_feat, p=2, dim=1)

        return shared_feat


def load_baseline_into_msr(msr_model, baseline_checkpoint_path, device="cpu"):
    """
    Warm-start MSR from shared baseline checkpoint.

    Copies:
    baseline.backbone  -> rgb_backbone + ir_backbone
    baseline.embedding -> rgb_embedding + ir_embedding
    baseline.classifier -> rgb_classifier + ir_classifier + shared_classifier
    """

    ckpt = torch.load(baseline_checkpoint_path, map_location=device)
    baseline_state = ckpt["model_state_dict"]

    def copy_module(prefix_from, module_to, module_name):
        new_state = {}

        for key, value in baseline_state.items():
            if key.startswith(prefix_from + "."):
                new_key = key.replace(prefix_from + ".", "")
                new_state[new_key] = value

        missing, unexpected = module_to.load_state_dict(new_state, strict=False)

        print(f"Loaded {prefix_from} -> {module_name}")
        print(f"  Missing keys: {len(missing)}")
        print(f"  Unexpected keys: {len(unexpected)}")

    copy_module("backbone", msr_model.rgb_backbone, "rgb_backbone")
    copy_module("backbone", msr_model.ir_backbone, "ir_backbone")

    copy_module("embedding", msr_model.rgb_embedding, "rgb_embedding")
    copy_module("embedding", msr_model.ir_embedding, "ir_embedding")

    copy_module("classifier", msr_model.rgb_classifier, "rgb_classifier")
    copy_module("classifier", msr_model.ir_classifier, "ir_classifier")
    copy_module("classifier", msr_model.shared_classifier, "shared_classifier")

    return msr_model, ckpt
