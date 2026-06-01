import os
import re
import random
import itertools

from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from data.dataset_roots import get_sysu_root
from models.baseline import BaselineReID


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
            print(f"Warning: missing camera folder {cam_path}")
            continue

        for pid_str in train_ids:
            person_dir = os.path.join(cam_path, pid_str)

            if not os.path.isdir(person_dir):
                continue

            label = id_to_label[pid_str]

            for img_name in os.listdir(person_dir):
                if img_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    img_path = os.path.join(person_dir, img_name)
                    samples.append((img_path, label, cam_name))

    return samples


class SYSUCameraDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, label, cam_name = self.samples[index]

        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, label, cam_name


def train_one_epoch(model, rgb_loader, ir_loader, criterion, optimizer, device, epoch):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    ir_iter = itertools.cycle(ir_loader)

    for batch_idx, (rgb_imgs, rgb_labels, _) in enumerate(rgb_loader):
        ir_imgs, ir_labels, _ = next(ir_iter)

        rgb_imgs = rgb_imgs.to(device, non_blocking=True)
        rgb_labels = rgb_labels.to(device, non_blocking=True)

        ir_imgs = ir_imgs.to(device, non_blocking=True)
        ir_labels = ir_labels.to(device, non_blocking=True)

        imgs = torch.cat([rgb_imgs, ir_imgs], dim=0)
        labels = torch.cat([rgb_labels, ir_labels], dim=0)

        _, logits = model(imgs)

        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if batch_idx % 50 == 0:
            acc = 100.0 * correct / total
            print(
                f"Epoch [{epoch}] Batch [{batch_idx}/{len(rgb_loader)}] "
                f"Loss: {loss.item():.4f} Acc: {acc:.2f}%"
            )

    avg_loss = total_loss / len(rgb_loader)
    avg_acc = 100.0 * correct / total

    return avg_loss, avg_acc


def main():
    root = get_sysu_root()
    print("Dataset path:", root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_id_path = os.path.join(root, "exp", "train_id.txt")
    train_ids = read_ids(train_id_path)

    id_to_label = {
        pid_str: idx for idx, pid_str in enumerate(sorted(train_ids))
    }

    num_classes = len(id_to_label)

    visible_cams = ["cam1", "cam2", "cam4", "cam5"]
    infrared_cams = ["cam3", "cam6"]

    rgb_samples = collect_camera_samples(
        root=root,
        cam_names=visible_cams,
        train_ids=train_ids,
        id_to_label=id_to_label,
    )

    ir_samples = collect_camera_samples(
        root=root,
        cam_names=infrared_cams,
        train_ids=train_ids,
        id_to_label=id_to_label,
    )

    print("RGB samples:", len(rgb_samples))
    print("IR samples:", len(ir_samples))
    print("Number of classes:", num_classes)

    transform = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    rgb_dataset = SYSUCameraDataset(rgb_samples, transform=transform)
    ir_dataset = SYSUCameraDataset(ir_samples, transform=transform)

    rgb_loader = DataLoader(
        rgb_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    ir_loader = DataLoader(
        ir_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    model = BaselineReID(
        num_classes=num_classes,
        pretrained=True,
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=3e-4,
        weight_decay=5e-4,
    )

    num_epochs = 5

    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(1, num_epochs + 1):
        loss, acc = train_one_epoch(
            model=model,
            rgb_loader=rgb_loader,
            ir_loader=ir_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
        )

        print(f"Epoch {epoch} finished | Loss: {loss:.4f} | Acc: {acc:.2f}%")

        save_path = f"checkpoints/baseline_epoch_{epoch}.pth"
        torch.save(model.state_dict(), save_path)
        print(f"Saved checkpoint: {save_path}")

    print("Baseline training finished.")


if __name__ == "__main__":
    main()
