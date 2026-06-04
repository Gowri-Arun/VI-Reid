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
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from models.baseline import BaselineReID


IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMG_EXTENSIONS


def find_sysu_root(dataset_root: str) -> Path:
    """
    Finds SYSU-MM01 folder inside the KaggleHub dataset root.
    """
    root = Path(dataset_root)

    if (root / "SYSU-MM01").exists():
        return root / "SYSU-MM01"

    if (root / "regdb_sysu_dataset" / "SYSU-MM01").exists():
        return root / "regdb_sysu_dataset" / "SYSU-MM01"

    for path in root.rglob("*"):
        if path.is_dir() and path.name.lower() == "sysu-mm01":
            return path

    raise FileNotFoundError(
        f"Could not find SYSU-MM01 folder inside {dataset_root}. "
        f"Run os.walk() to inspect the dataset structure."
    )


def read_id_file(path: Path) -> Optional[set]:
    """
    Reads train_id.txt / val_id.txt / test_id.txt if available.

    SYSU files usually contain comma-separated IDs.
    Example:
        0001,0002,0003,...
    """
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
    """
    SYSU usually stores images as:
        cam1/0001/xxxx.jpg

    Person ID is the parent folder name.
    """
    try:
        return int(path.parent.name)
    except ValueError:
        raise ValueError(
            f"Could not parse person ID from path: {path}. "
            f"Expected format like cam1/0001/image.jpg"
        )


def collect_sysu_images(
    sysu_root: Path,
    cams: List[str],
    allowed_ids: Optional[set] = None,
) -> List[Tuple[str, int, int]]:
    samples = []

    for cam_name in cams:
        cam_dir = sysu_root / cam_name

        if not cam_dir.exists():
            print(f"Warning: missing camera folder {cam_dir}", flush=True)
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


class SYSUFolderDataset(torch.utils.data.Dataset):
    def __init__(self, samples, label_map, transform=None):
        self.samples = samples
        self.label_map = label_map
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


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=False)
        labels = labels.to(device, non_blocking=False)

        optimizer.zero_grad(set_to_none=True)

        logits, _ = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if batch_idx % 20 == 0:
            print(
                f"Epoch {epoch} | Batch {batch_idx}/{len(loader)} | "
                f"Loss: {loss.item():.4f}",
                flush=True,
            )

    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--feature-dim", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--seed", type=int, default=42)

    # For quick sanity/debug runs. Use 0 for full dataset.
    parser.add_argument("--max-samples", type=int, default=0)

    args = parser.parse_args()

    set_seed(args.seed)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs("results", exist_ok=True)

    sysu_root = find_sysu_root(args.data_root)
    print(f"Using SYSU root: {sysu_root}", flush=True)

    exp_dir = sysu_root / "exp"
    train_ids = read_id_file(exp_dir / "train_id.txt")
    val_ids = read_id_file(exp_dir / "val_id.txt")

    if train_ids is not None:
        allowed_ids = set(train_ids)
        if val_ids is not None:
            allowed_ids |= set(val_ids)
        print(f"Using official train/val IDs: {len(allowed_ids)} IDs", flush=True)
    else:
        allowed_ids = None
        print("Warning: exp/train_id.txt not found. Using all IDs for training.", flush=True)

    cams = ["cam1", "cam2", "cam3", "cam4", "cam5", "cam6"]
    samples = collect_sysu_images(sysu_root, cams=cams, allowed_ids=allowed_ids)

    if len(samples) == 0:
        raise RuntimeError("No training images found. Check dataset structure.")

    random.shuffle(samples)

    if args.max_samples and args.max_samples > 0:
        samples = samples[:args.max_samples]
        print(f"DEBUG MODE: using only {len(samples)} samples", flush=True)

    label_map = build_label_map(samples)
    num_classes = len(label_map)

    print(f"Total training images: {len(samples)}", flush=True)
    print(f"Number of training identities/classes: {num_classes}", flush=True)

    transform = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    dataset = SYSUFolderDataset(
        samples=samples,
        label_map=label_map,
        transform=transform,
    )

    # Safer Colab settings:
    # num_workers=0 and pin_memory=False avoid many runtime/multiprocessing issues.
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, flush=True)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0), flush=True)

    model = BaselineReID(
        num_classes=num_classes,
        feature_dim=args.feature_dim,
        pretrained=True,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        loss, acc = train_one_epoch(
            model=model,
            loader=loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
        )

        print(f"Epoch {epoch} complete | Loss: {loss:.4f} | Acc: {acc:.4f}", flush=True)

        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "num_classes": num_classes,
            "feature_dim": args.feature_dim,
            "label_map": label_map,
            "sysu_root": str(sysu_root),
            "args": vars(args),
        }

        epoch_path = os.path.join(args.checkpoint_dir, f"baseline_epoch_{epoch}.pth")
        torch.save(ckpt, epoch_path)
        print(f"Saved checkpoint: {epoch_path}", flush=True)

        if loss < best_loss:
            best_loss = loss
            best_path = os.path.join(args.checkpoint_dir, "baseline_best.pth")
            torch.save(ckpt, best_path)
            print(f"Saved best checkpoint: {best_path}", flush=True)

    print("Baseline training complete.", flush=True)


if __name__ == "__main__":
    main()
