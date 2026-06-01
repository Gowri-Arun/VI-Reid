import os
import re
import random
import numpy as np

from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from data.dataset_roots import get_sysu_root
from models.msr_model import MSRModel
from metrics.reid_metrics import compute_cmc_map


def read_ids(txt_path):
    with open(txt_path, "r") as f:
        content = f.read()

    ids = re.findall(r"\d+", content)
    return [x.zfill(4) for x in ids]


def collect_samples(root, cam_names, test_ids):
    samples = []

    for cam_name in cam_names:
        cam_path = os.path.join(root, cam_name)

        if not os.path.isdir(cam_path):
            print(f"Warning: missing {cam_path}")
            continue

        camid = int(cam_name.replace("cam", ""))

        for pid_str in test_ids:
            person_dir = os.path.join(cam_path, pid_str)

            if not os.path.isdir(person_dir):
                continue

            pid = int(pid_str)

            for img_name in os.listdir(person_dir):
                if img_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    img_path = os.path.join(person_dir, img_name)
                    samples.append((img_path, pid, camid))

    return samples


def build_sysu_eval_sets(root):
    test_id_path = os.path.join(root, "exp", "test_id.txt")
    test_ids = read_ids(test_id_path)

    print("Number of test IDs:", len(test_ids))
    print("First 10 test IDs:", test_ids[:10])

    query_cams = ["cam3", "cam6"]
    gallery_cams = ["cam1", "cam2", "cam4", "cam5"]

    query_samples = collect_samples(root, query_cams, test_ids)
    gallery_samples = collect_samples(root, gallery_cams, test_ids)

    return query_samples, gallery_samples


def sample_single_shot_gallery(gallery_samples, seed):
    random.seed(seed)

    grouped = {}

    for img_path, pid, camid in gallery_samples:
        key = (pid, camid)
        grouped.setdefault(key, []).append((img_path, pid, camid))

    selected = []

    for key, items in grouped.items():
        selected.append(random.choice(items))

    return selected


class SYSUImageDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, pid, camid = self.samples[index]

        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, pid, camid, img_path


@torch.no_grad()
def extract_infrared_features(model, loader, device):
    model.eval()

    features = []
    pids = []
    camids = []

    for batch_idx, (imgs, batch_pids, batch_camids, _) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True)

        feats = model.extract_infrared(imgs)
        feats = F.normalize(feats, p=2, dim=1)

        features.append(feats.cpu().numpy())
        pids.extend(batch_pids.numpy())
        camids.extend(batch_camids.numpy())

        if batch_idx % 20 == 0:
            print(f"Infrared batch {batch_idx}/{len(loader)}")

    return np.concatenate(features, axis=0), np.asarray(pids), np.asarray(camids)


@torch.no_grad()
def extract_visible_features(model, loader, device):
    model.eval()

    features = []
    pids = []
    camids = []

    for batch_idx, (imgs, batch_pids, batch_camids, _) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True)

        feats = model.extract_visible(imgs)
        feats = F.normalize(feats, p=2, dim=1)

        features.append(feats.cpu().numpy())
        pids.extend(batch_pids.numpy())
        camids.extend(batch_camids.numpy())

        if batch_idx % 20 == 0:
            print(f"Visible batch {batch_idx}/{len(loader)}")

    return np.concatenate(features, axis=0), np.asarray(pids), np.asarray(camids)


def infer_num_classes_from_checkpoint(checkpoint):
    if "visible_classifier.weight" in checkpoint:
        return checkpoint["visible_classifier.weight"].shape[0]

    raise KeyError("Could not infer num_classes from checkpoint.")


def print_eval_sanity_checks(q_pids, q_camids, g_pids=None, g_camids=None):
    print("\nSanity checks")
    print("Unique query IDs:", len(set(q_pids.tolist())))
    print("Query cams:", sorted(set(q_camids.tolist())))

    if g_pids is not None and g_camids is not None:
        print("Unique gallery IDs:", len(set(g_pids.tolist())))
        print("Gallery cams:", sorted(set(g_camids.tolist())))

        query_ids = set(q_pids.tolist())
        gallery_ids = set(g_pids.tolist())
        missing_gallery_ids = sorted(query_ids - gallery_ids)

        print("Query IDs missing from gallery:", len(missing_gallery_ids))

        if len(missing_gallery_ids) > 0:
            print("First missing IDs:", missing_gallery_ids[:10])


def main():
    root = get_sysu_root()
    print("Dataset path:", root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    checkpoint_path = "checkpoints/msr_full_paper_arch_256_final.pth"

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"{checkpoint_path} not found. Run train_msr_full_paper_arch_256.py first."
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    num_classes = infer_num_classes_from_checkpoint(checkpoint)

    print("Detected num_classes from checkpoint:", num_classes)

    model = MSRModel(
        num_classes=num_classes,
        num_views=6,
        pretrained=False,
    ).to(device)

    model.load_state_dict(checkpoint)

    transform = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    query_samples, full_gallery_samples = build_sysu_eval_sets(root)

    print("Query images:", len(query_samples))
    print("Full gallery images:", len(full_gallery_samples))

    query_loader = DataLoader(
        SYSUImageDataset(query_samples, transform=transform),
        batch_size=128,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    print("\nExtracting IR query features once...")
    q_feats, q_pids, q_camids = extract_infrared_features(
        model,
        query_loader,
        device,
    )

    print_eval_sanity_checks(q_pids, q_camids)

    all_cmc = []
    all_mAP = []

    for trial_seed in range(10):
        print(f"\nTrial seed {trial_seed}")

        gallery_samples = sample_single_shot_gallery(
            full_gallery_samples,
            seed=trial_seed,
        )

        print("Single-shot gallery images:", len(gallery_samples))

        gallery_loader = DataLoader(
            SYSUImageDataset(gallery_samples, transform=transform),
            batch_size=128,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

        g_feats, g_pids, g_camids = extract_visible_features(
            model,
            gallery_loader,
            device,
        )

        print_eval_sanity_checks(q_pids, q_camids, g_pids, g_camids)

        distmat = 1 - np.matmul(q_feats, g_feats.T)

        cmc, mAP = compute_cmc_map(
            distmat,
            q_pids,
            g_pids,
            q_camids,
            g_camids,
            max_rank=20,
        )

        all_cmc.append(cmc)
        all_mAP.append(mAP)

        print(
            f"Trial {trial_seed + 1}/10 | "
            f"Rank-1: {cmc[0] * 100:.2f}% | "
            f"Rank-10: {cmc[9] * 100:.2f}% | "
            f"Rank-20: {cmc[19] * 100:.2f}% | "
            f"mAP: {mAP * 100:.2f}%"
        )

    mean_cmc = np.mean(np.asarray(all_cmc), axis=0)
    mean_mAP = np.mean(np.asarray(all_mAP))

    print("\nFinal SYSU-MM01 all-search single-shot result")
    print(f"Rank-1 : {mean_cmc[0] * 100:.2f}%")
    print(f"Rank-5 : {mean_cmc[4] * 100:.2f}%")
    print(f"Rank-10: {mean_cmc[9] * 100:.2f}%")
    print(f"Rank-20: {mean_cmc[19] * 100:.2f}%")
    print(f"mAP    : {mean_mAP * 100:.2f}%")


if __name__ == "__main__":
    main()
