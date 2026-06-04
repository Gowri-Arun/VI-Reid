
# Visible-Infrared Person Re-Identification with Warm-Started MSR

This repository contains a research-engineering implementation of a warm-started **Modality-Shared and Modality-Specific Representation (MSR)** model for visible-infrared person re-identification on the SYSU-MM01 dataset.

The goal is to improve cross-modal retrieval between infrared query images and visible/RGB gallery images.

---

## Overview

Visible-infrared person re-identification is the task of matching a person captured by an infrared camera with images of the same person captured by visible/RGB cameras.

This project compares two approaches:

1. **Shared baseline**
   - A single ResNet-50 based model is used for both RGB and infrared images.

2. **Warm-started MSR model**
   - A two-branch RGB/IR model is initialized from the trained shared baseline.
   - Both RGB and IR branches start from the same baseline weights.
   - The branches then specialize using modality-specific and shared identity losses.

---

## Key Result

Under the same SYSU-MM01 simple all-search protocol, warm-started MSR improves both Rank-1 accuracy and mean Average Precision (mAP).

| Model / Checkpoint | Rank-1 | Rank-5 | Rank-10 | Rank-20 | mAP |
|---|---:|---:|---:|---:|---:|
| Shared baseline | 28.48 | 47.46 | 56.80 | 65.50 | 18.85 |
| Warm-start MSR epoch 2 | 32.08 | 48.20 | 55.40 | 64.71 | 22.84 |

### Improvement Over Baseline

| Metric | Baseline | Warm-start MSR | Improvement |
|---|---:|---:|---:|
| Rank-1 | 28.48 | 32.08 | +3.60 |
| mAP | 18.85 | 22.84 | +3.99 |

---

## Evaluation Protocol

Current reported results use a simplified SYSU-MM01 all-search protocol.

- Query modality: infrared
- Query cameras: `cam3`, `cam6`
- Gallery modality: visible/RGB
- Gallery cameras: `cam1`, `cam2`, `cam4`, `cam5`
- Official test identities: 96
- Query samples: 3803
- Gallery samples: 6775
- Feature normalization: L2 normalization enabled
- Distance metric: Euclidean distance
- Distance computation: batched computation to avoid memory issues

All reported comparisons use the same protocol.

Changing the query/gallery split, feature normalization, checkpoint, distance function, or metric implementation invalidates direct comparison.

---

## Method

### Shared Baseline

The baseline uses a ResNet-50 feature extractor shared across both RGB and infrared images.

The model is trained using identity classification loss.

### Warm-Started MSR

The warm-started MSR model uses two modality-specific branches:

- RGB branch for visible images
- IR branch for infrared images

Both branches are initialized from the trained shared baseline checkpoint.

The model includes:

- RGB-specific backbone and embedding
- IR-specific backbone and embedding
- Shared feature projection layer
- RGB identity classifier
- IR identity classifier
- Shared identity classifier

### Training Losses

The warm-started MSR model is trained using:

- RGB identity classification loss
- IR identity classification loss
- RGB shared identity loss
- IR shared identity loss
- Cross-Modality Euclidean Constraint (CMEC)

---

## Why Warm-Start?

Training a two-branch RGB/IR model directly can cause the RGB and IR branches to drift into poorly aligned feature spaces.

Warm-starting both branches from the shared baseline gives the model a stable aligned initialization. After that, each branch can specialize for its own modality while still preserving shared identity-discriminative features.

In this experiment, warm-starting improved both Rank-1 and mAP compared to the shared baseline.

---

## Checkpoint Sweep

The model was trained for 5 epochs. Each epoch checkpoint was evaluated separately because the lowest training loss checkpoint is not always the best retrieval checkpoint.

| Checkpoint | Rank-1 | Rank-5 | Rank-10 | Rank-20 | mAP |
|---|---:|---:|---:|---:|---:|
| MSR epoch 1 | 31.82 | 47.91 | 56.40 | 65.29 | 22.50 |
| MSR epoch 2 | 32.08 | 48.20 | 55.40 | 64.71 | 22.84 |
| MSR epoch 3 | 32.00 | 48.09 | 56.17 | 64.63 | 22.82 |
| MSR epoch 4 | 31.66 | 47.46 | 55.25 | 63.66 | 22.14 |
| MSR epoch 5 | 30.82 | 46.54 | 54.38 | 62.85 | 22.03 |

Best checkpoint by retrieval performance:

```text
checkpoints/msr_warmstart_epoch_2.pth
```

Checkpoint files are not committed to GitHub due to file size constraints.

---

## Key Observation

Epoch 2 gives the best retrieval performance.

Later epochs continue reducing training loss, but retrieval performance drops. This suggests that the model begins to overfit to the training identities after epoch 2.

---

## Repository Structure

```text
VI-Reid/
│
├── README.md
├── .gitignore
├── requirements.txt
├── experiment_log.md
│
├── commands/
│   ├── day01_protocol_lock.txt
│   ├── train_baseline.txt
│   ├── train_msr_warmstart.txt
│   └── evaluate_checkpoint_sweep.txt
│
├── configs/
│   ├── baseline.yaml
│   └── msr_warmstart.yaml
│
├── data/
│   ├── __init__.py
│   └── dataset_roots.py
│
├── models/
│   ├── __init__.py
│   ├── baseline.py
│   └── msr_model.py
│
├── metrics/
│   ├── __init__.py
│   └── reid_metrics.py
│
├── results/
│   ├── baseline_result.json
│   ├── msr_epoch1_result.json
│   ├── msr_epoch2_result.json
│   ├── msr_epoch3_result.json
│   ├── msr_epoch4_result.json
│   ├── msr_epoch5_result.json
│   ├── msr_warmstart_result.json
│   └── summary.md
│
├── scripts/
│   ├── train_baseline.py
│   ├── train_msr_warmstart.py
│   ├── evaluate_baseline.py
│   └── evaluate_msr_warmstart.py
│
└── docs/
    ├── protocol.md
    ├── methodology.md
    ├── results.md
    └── failure_analysis.md
```

---

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Expected core dependencies:

- PyTorch
- Torchvision
- NumPy
- Pillow
- tqdm
- PyYAML

---

## Dataset

This project uses the SYSU-MM01 visible-infrared person re-identification dataset.

The scripts expect a dataset root that contains the `SYSU-MM01` folder, either directly or inside the KaggleHub dataset structure.

Example KaggleHub path:

```text
/root/.cache/kagglehub/datasets/gowr1arun/regdb-sysu-dataset/versions/1
```

The scripts automatically search for:

```text
SYSU-MM01/
```

or:

```text
regdb_sysu_dataset/SYSU-MM01/
```

---

## Reproduce

These commands are provided for reproducibility. They are not required for repository setup.

Checkpoint files are not committed to GitHub due to file size constraints. To exactly reproduce the reported result, use the saved checkpoint from this run.

### Train Shared Baseline

```bash
python scripts/train_baseline.py \
  --data-root /path/to/dataset \
  --epochs 5 \
  --batch-size 16 \
  --lr 0.0003 \
  --checkpoint-dir checkpoints
```

### Train Warm-Started MSR

```bash
python scripts/train_msr_warmstart.py \
  --data-root /path/to/dataset \
  --baseline-checkpoint checkpoints/baseline_best.pth \
  --epochs 5 \
  --batch-size 32 \
  --lr 0.0001 \
  --cmec-weight 0.1 \
  --checkpoint-dir checkpoints
```

### Evaluate Best MSR Checkpoint

```bash
python scripts/evaluate_msr_warmstart.py \
  --data-root /path/to/dataset \
  --checkpoint checkpoints/msr_warmstart_epoch_2.pth \
  --normalize-features \
  --output results/msr_epoch2_result.json
```

### Run Checkpoint Sweep

See:

```text
commands/evaluate_checkpoint_sweep.txt
```

---

## Results

Detailed result files are stored in:

```text
results/
```

Summary:

```text
results/summary.md
```

Full experiment log:

```text
experiment_log.md
```

Additional documentation:

```text
docs/protocol.md
docs/methodology.md
docs/results.md
docs/failure_analysis.md
```

---

## Notes on Checkpoints

Model checkpoints are excluded from GitHub because of file size constraints.

Ignored checkpoint formats:

```text
*.pth
*.pt
checkpoints/
```

Best checkpoint from the current run:

```text
msr_warmstart_epoch_2.pth
```

To preserve checkpoints across Colab sessions, save them to Google Drive, Kaggle output, or another external artifact store.

---

## Current Status

Completed:

- Shared baseline training and evaluation
- Warm-started MSR implementation
- Warm-started MSR training
- Checkpoint sweep across epochs 1–5
- Batched distance evaluation to avoid memory issues
- Result logging and summary documentation

Next possible experiments:

- CMEC weight sweep
- Stronger data augmentation
- Identity-balanced sampler
- Triplet loss with balanced batches
- SYSU single-shot 10-trial evaluation
- Test-time augmentation

---

## Main Takeaway

Warm-starting a modality-specific two-branch MSR model from a shared baseline improves cross-modal visible-infrared person retrieval.

The best result in the current experiment is achieved at epoch 2:

```text
Rank-1: 32.08
mAP: 22.84
```

This improves over the shared baseline by:

```text
Rank-1: +3.60
mAP: +3.99
```
