import os
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from models.msr_model import MSRReID


IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


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


class EvalFolderDataset(Dataset):
    def __init__(self, samples, modality: str, transform=None):
        self.samples = samples
        self.modality = modality
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, pid, camid = self.samples[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, pid, camid


@torch.no_grad()
def extract_features(model, loader, device, modality: str, normalize=True):
    model.eval()

    features = []
    labels = []
    camids = []

    for images, pids, cams in loader:
        images = images.to(device, non_blocking=True)

        feat = model.extract_features(
            images,
            modality=modality,
            normalize=normalize,
        )

        features.append(feat.cpu().numpy())
        labels.append(pids.numpy())
        camids.append(cams.numpy())

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    camids = np.concatenate(camids, axis=0)

    return features, labels, camids


def compute_distmat_batched(query_features, gallery_features, batch_size=64):
    q = torch.tensor(query_features, dtype=torch.float32)
    g = torch.tensor(gallery_features, dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q = q.to(device)
    g = g.to(device)

    all_dists = []

    for start in range(0, q.size(0), batch_size):
        end = min(start + batch_size, q.size(0))
        q_batch = q[start:end]

        dist = torch.cdist(q_batch, g, p=2)
        all_dists.append(dist.cpu().numpy())

        print(f"Computed distances for queries {start} to {end}")

    return np.concatenate(all_dists, axis=0)


def evaluate_rank_map(
    query_features,
    query_labels,
    query_camids,
    gallery_features,
    gallery_labels,
    gallery_camids,
    max_rank=20,
):
    num_g = gallery_features.shape[0]

    if num_g < max_rank:
        max_rank = num_g

    print("Computing batched distance matrix...")
    distmat = compute_distmat_batched(
        query_features=query_features,
        gallery_features=gallery_features,
        batch_size=64,
    )

    print("Sorting distance matrix...")
    indices = np.argsort(distmat, axis=1)
    matches = (gallery_labels[indices] == query_labels[:, None]).astype(np.int32)

    all_cmc = []
    all_ap = []
    valid_q = 0

    for q_idx in range(query_features.shape[0]):
        q_pid = query_labels[q_idx]
        q_camid = query_camids[q_idx]

        order = indices[q_idx]

        remove = (gallery_labels[order] == q_pid) & (gallery_camids[order] == q_camid)
        keep = np.invert(remove)

        raw_cmc = matches[q_idx][keep]

        if not np.any(raw_cmc):
            continue

        cmc = raw_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        valid_q += 1

        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        precision = tmp_cmc / (np.arange(len(raw_cmc)) + 1)
        ap = (precision * raw_cmc).sum() / num_rel
        all_ap.append(ap)

    if valid_q == 0:
        raise RuntimeError("No valid query identities found in gallery.")

    min_len = min(len(c) for c in all_cmc)
    all_cmc = np.asarray([c[:min_len] for c in all_cmc]).astype(float)

    cmc = all_cmc.sum(axis=0) / valid_q
    mAP = np.mean(all_ap)

    return cmc, mAP


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--normalize-features", action="store_true")
    parser.add_argument("--output", type=str, default="results/msr_warmstart_result.json")

    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    sysu_root = find_sysu_root(args.data_root)
    print(f"Using SYSU root: {sysu_root}")

    exp_dir = sysu_root / "exp"
    test_ids = read_id_file(exp_dir / "test_id.txt")

    if test_ids is None:
        print("Warning: exp/test_id.txt not found. Evaluation will use all IDs.")
    else:
        print(f"Using official test IDs: {len(test_ids)} IDs")

    query_cams = ["cam3", "cam6"]
    gallery_cams = ["cam1", "cam2", "cam4", "cam5"]

    query_samples = collect_sysu_images(sysu_root, query_cams, test_ids)
    gallery_samples = collect_sysu_images(sysu_root, gallery_cams, test_ids)

    print(f"Query samples: {len(query_samples)}")
    print(f"Gallery samples: {len(gallery_samples)}")

    if len(query_samples) == 0 or len(gallery_samples) == 0:
        raise RuntimeError("Query or gallery set is empty.")

    ckpt = torch.load(args.checkpoint, map_location="cpu")

    num_classes = ckpt["num_classes"]
    feature_dim = ckpt.get("feature_dim", 512)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = MSRReID(
        num_classes=num_classes,
        feature_dim=feature_dim,
        pretrained=False,
    )

    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model = model.to(device)

    transform = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    query_dataset = EvalFolderDataset(query_samples, modality="ir", transform=transform)
    gallery_dataset = EvalFolderDataset(gallery_samples, modality="rgb", transform=transform)

    query_loader = DataLoader(
        query_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    gallery_loader = DataLoader(
        gallery_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print("Extracting query IR features...")
    q_feat, q_pids, q_camids = extract_features(
        model=model,
        loader=query_loader,
        device=device,
        modality="ir",
        normalize=args.normalize_features,
    )

    print("Extracting gallery RGB features...")
    g_feat, g_pids, g_camids = extract_features(
        model=model,
        loader=gallery_loader,
        device=device,
        modality="rgb",
        normalize=args.normalize_features,
    )

    print("Evaluating...")
    cmc, mAP = evaluate_rank_map(
        query_features=q_feat,
        query_labels=q_pids,
        query_camids=q_camids,
        gallery_features=g_feat,
        gallery_labels=g_pids,
        gallery_camids=g_camids,
        max_rank=20,
    )

    result = {
        "model": "warmstart_msr",
        "protocol": "SYSU simple all-search: IR query cam3/cam6, RGB gallery cam1/cam2/cam4/cam5",
        "checkpoint": args.checkpoint,
        "normalize_features": args.normalize_features,
        "rank1": float(cmc[0] * 100),
        "rank5": float(cmc[4] * 100) if len(cmc) > 4 else None,
        "rank10": float(cmc[9] * 100) if len(cmc) > 9 else None,
        "rank20": float(cmc[19] * 100) if len(cmc) > 19 else None,
        "map": float(mAP * 100),
        "num_query": int(len(q_pids)),
        "num_gallery": int(len(g_pids)),
    }

    print(json.dumps(result, indent=2))

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Saved result to {args.output}")


if __name__ == "__main__":
    main()
