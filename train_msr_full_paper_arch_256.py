import os
import re
import random

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from data.dataset_roots import get_sysu_root
from models.msr_model import MSRModel


def read_ids(txt_path):
    with open(txt_path, "r") as f:
        content = f.read()

    ids = re.findall(r"\d+", content)
    return [x.zfill(4) for x in ids]


def collect_camera_samples(root, cam_names, train_ids, id_to_label):
    samples = []

    for cam_name in cam_names:
        cam_path = os.path.join(root, cam_name)

        if not os.path.isdir(cam_path):
            print(f"Warning: {cam_path} missing")
            continue

        camid = int(cam_name.replace("cam", ""))
        view_label = camid - 1

        for pid_str in train_ids:
            person_dir = os.path.join(cam_path, pid_str)

            if not os.path.isdir(person_dir):
                continue

            label = id_to_label[pid_str]

            for img_name in os.listdir(person_dir):
                if img_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    img_path = os.path.join(person_dir, img_name)
                    samples.append((img_path, label, view_label))

    return samples


class PairedSYSUCameraDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform

        train_id_path = os.path.join(root, "exp", "train_id.txt")
        train_ids = read_ids(train_id_path)

        self.id_to_label = {
            pid_str: idx for idx, pid_str in enumerate(sorted(train_ids))
        }

        self.num_classes = len(self.id_to_label)

        visible_cams = ["cam1", "cam2", "cam4", "cam5"]
        infrared_cams = ["cam3", "cam6"]

        self.rgb_samples = collect_camera_samples(
            root=root,
            cam_names=visible_cams,
            train_ids=train_ids,
            id_to_label=self.id_to_label,
        )

        self.ir_samples = collect_camera_samples(
            root=root,
            cam_names=infrared_cams,
            train_ids=train_ids,
            id_to_label=self.id_to_label,
        )

        self.rgb_by_pid = {}
        self.ir_by_pid = {}

        for sample in self.rgb_samples:
            _, label, _ = sample
            self.rgb_by_pid.setdefault(label, []).append(sample)

        for sample in self.ir_samples:
            _, label, _ = sample
            self.ir_by_pid.setdefault(label, []).append(sample)

        self.identities = sorted(
            list(set(self.rgb_by_pid.keys()) & set(self.ir_by_pid.keys()))
        )

        print("RGB samples:", len(self.rgb_samples))
        print("IR samples:", len(self.ir_samples))
        print("Shared train identities:", len(self.identities))
        print("Number of classifier classes:", self.num_classes)
        print("Min label:", min(self.identities))
        print("Max label:", max(self.identities))

    def __len__(self):
        return max(len(self.rgb_samples), len(self.ir_samples))

    def __getitem__(self, index):
        label = random.choice(self.identities)

        rgb_path, _, rgb_view = random.choice(self.rgb_by_pid[label])
        ir_path, _, ir_view = random.choice(self.ir_by_pid[label])

        rgb_img = Image.open(rgb_path).convert("RGB")
        ir_img = Image.open(ir_path).convert("RGB")

        if self.transform is not None:
            rgb_img = self.transform(rgb_img)
            ir_img = self.transform(ir_img)

        return rgb_img, ir_img, label, rgb_view, ir_view


def find_latest_checkpoint(folder, prefix):
    if not os.path.exists(folder):
        return None

    candidates = []

    for name in os.listdir(folder):
        if name.startswith(prefix) and name.endswith(".pth"):
            match = re.search(r"epoch_(\d+)", name)

            if match:
                epoch = int(match.group(1))
                candidates.append((epoch, os.path.join(folder, name)))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def load_baseline_backbone_into_msr(model, baseline_ckpt_path):
    print("Loading baseline backbone checkpoint:", baseline_ckpt_path)

    baseline_state = torch.load(baseline_ckpt_path, map_location="cpu")

    if isinstance(baseline_state, dict) and "state_dict" in baseline_state:
        baseline_state = baseline_state["state_dict"]

    backbone_state = {}

    for key, value in baseline_state.items():
        if key.startswith("backbone."):
            backbone_state[key.replace("backbone.", "")] = value

    if len(backbone_state) == 0:
        raise ValueError("No backbone weights found in baseline checkpoint.")

    model.visible_backbone.load_state_dict(backbone_state, strict=True)
    model.infrared_backbone.load_state_dict(backbone_state, strict=True)

    print("Warm-start complete: baseline backbone copied into both branches.")
    print("Classifiers are newly initialized because labels are remapped.")

    return model


def cross_modality_euclidean_loss(rgb_feat, ir_feat):
    rgb_feat = F.normalize(rgb_feat, p=2, dim=1)
    ir_feat = F.normalize(ir_feat, p=2, dim=1)

    return torch.mean(torch.sum((rgb_feat - ir_feat) ** 2, dim=1))


def batch_hard_triplet_loss(features, labels, margin=0.3):
    features = F.normalize(features, p=2, dim=1)
    dist = torch.cdist(features, features, p=2)

    labels = labels.view(-1, 1)
    same = labels.eq(labels.t())
    different = ~same

    eye = torch.eye(labels.size(0), dtype=torch.bool, device=labels.device)
    same = same & ~eye

    losses = []

    for i in range(labels.size(0)):
        pos_dist = dist[i][same[i]]
        neg_dist = dist[i][different[i]]

        if len(pos_dist) == 0 or len(neg_dist) == 0:
            continue

        hardest_pos = pos_dist.max()
        hardest_neg = neg_dist.min()

        losses.append(F.relu(hardest_pos - hardest_neg + margin))

    if len(losses) == 0:
        return torch.tensor(0.0, device=features.device, requires_grad=True)

    return torch.stack(losses).mean()


def train_one_epoch(
    model,
    loader,
    ce_loss,
    optimizer,
    device,
    epoch,
    lambda_cmec=0.1,
    lambda_msdm=0.1,
    lambda_view=0.01,
):
    model.train()

    total_loss = 0.0
    total_id = 0.0
    total_cmec = 0.0
    total_msdm = 0.0
    total_view = 0.0

    correct_rgb = 0
    correct_ir = 0
    total = 0

    for batch_idx, (rgb_imgs, ir_imgs, labels, rgb_views, ir_views) in enumerate(loader):
        rgb_imgs = rgb_imgs.to(device, non_blocking=True)
        ir_imgs = ir_imgs.to(device, non_blocking=True)

        labels = labels.to(device, non_blocking=True)
        rgb_views = rgb_views.to(device, non_blocking=True)
        ir_views = ir_views.to(device, non_blocking=True)

        outputs = model(rgb_imgs, ir_imgs)

        rgb_feat = outputs["rgb_feat"]
        ir_feat = outputs["ir_feat"]

        loss_rgb_id = ce_loss(outputs["rgb_logits"], labels)
        loss_ir_id = ce_loss(outputs["ir_logits"], labels)

        loss_rgb_shared = ce_loss(outputs["rgb_shared_logits"], labels)
        loss_ir_shared = ce_loss(outputs["ir_shared_logits"], labels)

        id_loss = (
            loss_rgb_id
            + loss_ir_id
            + loss_rgb_shared
            + loss_ir_shared
        )

        loss_cmec = cross_modality_euclidean_loss(rgb_feat, ir_feat)

        loss_msdm_rgb = batch_hard_triplet_loss(rgb_feat, labels)
        loss_msdm_ir = batch_hard_triplet_loss(ir_feat, labels)
        loss_msdm = loss_msdm_rgb + loss_msdm_ir

        loss_view_rgb = ce_loss(outputs["rgb_view_logits"], rgb_views)
        loss_view_ir = ce_loss(outputs["ir_view_logits"], ir_views)
        view_loss = loss_view_rgb + loss_view_ir

        loss = (
            id_loss
            + lambda_cmec * loss_cmec
            + lambda_msdm * loss_msdm
            + lambda_view * view_loss
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        total_id += id_loss.item()
        total_cmec += loss_cmec.item()
        total_msdm += loss_msdm.item()
        total_view += view_loss.item()

        rgb_preds = outputs["rgb_logits"].argmax(dim=1)
        ir_preds = outputs["ir_logits"].argmax(dim=1)

        correct_rgb += (rgb_preds == labels).sum().item()
        correct_ir += (ir_preds == labels).sum().item()
        total += labels.size(0)

        if batch_idx % 50 == 0:
            rgb_acc = 100.0 * correct_rgb / total
            ir_acc = 100.0 * correct_ir / total

            print(
                f"Epoch [{epoch}] Batch [{batch_idx}/{len(loader)}] "
                f"Loss: {loss.item():.4f} "
                f"ID: {id_loss.item():.4f} "
                f"CMEC: {loss_cmec.item():.4f} "
                f"MSDM: {loss_msdm.item():.4f} "
                f"View: {view_loss.item():.4f} "
                f"RGB Acc: {rgb_acc:.2f}% "
                f"IR Acc: {ir_acc:.2f}%"
            )

    n = len(loader)

    return {
        "loss": total_loss / n,
        "id": total_id / n,
        "cmec": total_cmec / n,
        "msdm": total_msdm / n,
        "view": total_view / n,
        "rgb_acc": 100.0 * correct_rgb / total,
        "ir_acc": 100.0 * correct_ir / total,
    }


def main():
    root = get_sysu_root()
    print("Dataset path:", root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    transform = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    dataset = PairedSYSUCameraDataset(root, transform=transform)

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    model = MSRModel(
        num_classes=dataset.num_classes,
        num_views=6,
        pretrained=False,
    )

    baseline_ckpt = find_latest_checkpoint("checkpoints", "baseline_epoch_")

    if baseline_ckpt is None:
        raise FileNotFoundError(
            "No baseline checkpoint found. Run train_baseline.py first until "
            "checkpoints/baseline_epoch_5.pth exists."
        )

    model = load_baseline_backbone_into_msr(model, baseline_ckpt)
    model = model.to(device)

    ce_loss = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=1e-3,
        momentum=0.9,
        weight_decay=1e-4,
    )

    num_epochs = 10

    lambda_cmec = 0.1
    lambda_msdm = 0.1
    lambda_view = 0.01

    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(1, num_epochs + 1):
        stats = train_one_epoch(
            model=model,
            loader=loader,
            ce_loss=ce_loss,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            lambda_cmec=lambda_cmec,
            lambda_msdm=lambda_msdm,
            lambda_view=lambda_view,
        )

        print(
            f"\nFull Paper-Architecture MSR 256x128 Epoch {epoch} finished | "
            f"Loss: {stats['loss']:.4f} | "
            f"ID: {stats['id']:.4f} | "
            f"CMEC: {stats['cmec']:.4f} | "
            f"MSDM: {stats['msdm']:.4f} | "
            f"View: {stats['view']:.4f} | "
            f"RGB Acc: {stats['rgb_acc']:.2f}% | "
            f"IR Acc: {stats['ir_acc']:.2f}%\n"
        )

        if epoch % 5 == 0:
            save_path = f"checkpoints/msr_full_paper_arch_256_epoch_{epoch}.pth"
            torch.save(model.state_dict(), save_path)
            print(f"Saved checkpoint: {save_path}")

    final_path = "checkpoints/msr_full_paper_arch_256_final.pth"
    torch.save(model.state_dict(), final_path)

    print(f"Saved final checkpoint: {final_path}")
    print("Full paper-architecture MSR 256x128 training finished.")


if __name__ == "__main__":
    main()
