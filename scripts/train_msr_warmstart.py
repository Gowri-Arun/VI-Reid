import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))
import argparse
import random
from pathlib import Path
from typing import List, Tuple, Optional

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from models.msr_model import MSRReID, load_baseline_into_msr


IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMG_EXTENSIONS


def find_sysu_root(dataset_root: str) -> Path:
    root = Path(dataset_root)

    if (root / "SYSU-MM01").exists():
        return root / "SYSU-MM01"

    if (root / "regdb_sysu_dataset" / "SYSU-MM01").exists():
        return root / "regdb_sysu_dataset" / "SYSU-MM01"

    for path in root.rglob("*"):
        if path.is_dir() and path.name.lower() == "sysu-mm01":
            return path

    raise FileNotFoundError(f"Could not find SYSU-MM01 inside {dataset_root}")


def read_id_file(path: Path) -> Optional[set]:
    if not path.exists():
        return None

    text = path.read_text().strip()
    ids = []

    for token in text.replace("\n", ",").split(","):
        token = token.strip()
        if token:
            ids.append(int(token))

    return set(ids)


def parse_pid_from_path(path: Path) -> int:
    return int(path.parent.name)


def collect_sysu_images(sysu_root: Path, cams: List[str], allowed_ids: Optional[set] = None):
    samples = []

    for cam_name in cams:
        cam_dir = sysu_root / cam_name

        if not cam_dir.exists():
            print(f"Warning: missing {cam_dir}")
            continue

        camid = int(cam_name.replace("cam", ""))

        for img_path in cam_dir.rglob("*"):
            if not img_path.is_file() or not is_image(img_path):
                continue

            pid = parse_pid_from_path(img_path)

            if allowed_ids is not None and pid not in allowed_ids:
                continue

            samples.append((str(img_path), pid, camid))

    return samples


def build_label_map(samples):
    pids = sorted(set(pid for _, pid, _ in samples))
    return {pid: idx for idx, pid in enumerate(pids)}


class SYSUModalityDataset(Dataset):
    def __init__(self, samples, label_map, modality: str, transform=None):
        self.samples = samples
        self.label_map = label_map
        self.modality = modality
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, pid, camid = self.samples[idx]

        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        label = self.label_map[pid]

        return img, label


def cmec_loss(rgb_shared, ir_shared):
    """
    Simple Cross-Modality Euclidean Constraint.

    Aligns batch-level RGB and IR shared feature centers.
    This is intentionally simple and stable for restart.
    """
    rgb_center = rgb_shared.mean(dim=0)
    ir_center = ir_shared.mean(dim=0)

    return F.mse_loss(rgb_center, ir_center)


def train_one_epoch(
    model,
    rgb_loader,
    ir_loader,
    criterion,
    optimizer,
    device,
    epoch,
    cmec_weight,
):
    model.train()

    total_loss = 0.0
    total_batches = 0

    ir_iter = iter(ir_loader)

    for batch_idx, (rgb_imgs, rgb_labels) in enumerate(rgb_loader):
        try:
            ir_imgs, ir_labels = next(ir_iter)
        except StopIteration:
            ir_iter = iter(ir_loader)
            ir_imgs, ir_labels = next(ir_iter)

        rgb_imgs = rgb_imgs.to(device, non_blocking=True)
        rgb_labels = rgb_labels.to(device, non_blocking=True)

        ir_imgs = ir_imgs.to(device, non_blocking=True)
        ir_labels = ir_labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        rgb_logits, rgb_shared_logits, rgb_specific, rgb_shared = model(rgb_imgs, modality="rgb")
        ir_logits, ir_shared_logits, ir_specific, ir_shared = model(ir_imgs, modality="ir")

        loss_rgb_id = criterion(rgb_logits, rgb_labels)
        loss_ir_id = criterion(ir_logits, ir_labels)

        loss_rgb_shared = criterion(rgb_shared_logits, rgb_labels)
        loss_ir_shared = criterion(ir_shared_logits, ir_labels)

        loss_cmec = cmec_loss(rgb_shared, ir_shared)

        loss = (
            loss_rgb_id
            + loss_ir_id
            + loss_rgb_shared
            + loss_ir_shared
            + cmec_weight * loss_cmec
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1

        if batch_idx % 50 == 0:
            print(
                f"Epoch {epoch} | Batch {batch_idx}/{len(rgb_loader)} | "
                f"Loss: {loss.item():.4f} | "
                f"RGB_ID: {loss_rgb_id.item():.4f} | "
                f"IR_ID: {loss_ir_id.item():.4f} | "
                f"CMEC: {loss_cmec.item():.4f}"
            )

    return total_loss / max(total_batches, 1)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--baseline-checkpoint", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--feature-dim", type=int, default=512)
    parser.add_argument("--cmec-weight", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    set_seed(args.seed)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs("results", exist_ok=True)

    sysu_root = find_sysu_root(args.data_root)
    print(f"Using SYSU root: {sysu_root}")

    exp_dir = sysu_root / "exp"
    train_ids = read_id_file(exp_dir / "train_id.txt")
    val_ids = read_id_file(exp_dir / "val_id.txt")

    if train_ids is not None:
        allowed_ids = set(train_ids)
        if val_ids is not None:
            allowed_ids |= set(val_ids)
        print(f"Using official train/val IDs: {len(allowed_ids)} IDs")
    else:
        allowed_ids = None
        print("Warning: train_id.txt not found. Using all IDs.")

    rgb_cams = ["cam1", "cam2", "cam4", "cam5"]
    ir_cams = ["cam3", "cam6"]

    rgb_samples = collect_sysu_images(sysu_root, rgb_cams, allowed_ids)
    ir_samples = collect_sysu_images(sysu_root, ir_cams, allowed_ids)

    all_samples = rgb_samples + ir_samples
    label_map = build_label_map(all_samples)
    num_classes = len(label_map)

    print(f"RGB train samples: {len(rgb_samples)}")
    print(f"IR train samples: {len(ir_samples)}")
    print(f"Number of train classes: {num_classes}")

    transform = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    rgb_dataset = SYSUModalityDataset(rgb_samples, label_map, modality="rgb", transform=transform)
    ir_dataset = SYSUModalityDataset(ir_samples, label_map, modality="ir", transform=transform)

    rgb_loader = DataLoader(
        rgb_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    ir_loader = DataLoader(
        ir_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = MSRReID(
        num_classes=num_classes,
        feature_dim=args.feature_dim,
        pretrained=False,
    )

    model, baseline_ckpt = load_baseline_into_msr(
        msr_model=model,
        baseline_checkpoint_path=args.baseline_checkpoint,
        device="cpu",
    )

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        avg_loss = train_one_epoch(
            model=model,
            rgb_loader=rgb_loader,
            ir_loader=ir_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            cmec_weight=args.cmec_weight,
        )

        print(f"Epoch {epoch} complete | Avg Loss: {avg_loss:.4f}")

        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "num_classes": num_classes,
            "feature_dim": args.feature_dim,
            "label_map": label_map,
            "sysu_root": str(sysu_root),
            "cmec_weight": args.cmec_weight,
            "args": vars(args),
        }

        epoch_path = os.path.join(args.checkpoint_dir, f"msr_warmstart_epoch_{epoch}.pth")
        torch.save(ckpt, epoch_path)
        print(f"Saved checkpoint: {epoch_path}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(args.checkpoint_dir, "msr_warmstart_best.pth")
            torch.save(ckpt, best_path)
            print(f"Saved best checkpoint: {best_path}")

    print("Warm-start MSR training complete.")


if __name__ == "__main__":
    main()
