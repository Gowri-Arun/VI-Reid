# Failure Analysis

## Observation

Warm-started MSR improves from epoch 1 to epoch 2, then begins to drop after epoch 3.

## Evidence

| Checkpoint | Rank-1 | mAP |
|---|---:|---:|
| MSR epoch 1 | 31.82 | 22.50 |
| MSR epoch 2 | 32.08 | 22.84 |
| MSR epoch 3 | 32.00 | 22.82 |
| MSR epoch 4 | 31.66 | 22.14 |
| MSR epoch 5 | 30.82 | 22.03 |

## Interpretation

Training loss continued decreasing, but retrieval performance dropped after epoch 2.

This suggests overfitting to training identities. The checkpoint with the lowest training loss was not the best retrieval checkpoint.

## Next Experiments

- Early stopping based on retrieval metrics
- CMEC weight sweep
- Stronger data augmentation
- Identity-balanced sampling
- Triplet loss with balanced batches
- SYSU single-shot 10-trial evaluation
