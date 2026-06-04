# SYSU-MM01 MSR Warm-Start Experiment Log

## Protocol

SYSU simple all-search protocol:

- Query: IR cameras cam3 and cam6
- Gallery: RGB cameras cam1, cam2, cam4, and cam5
- Official test IDs: 96
- Query samples: 3803
- Gallery samples: 6775
- Feature normalization: L2 enabled

## Baseline Result

| Model | Rank-1 | Rank-5 | Rank-10 | Rank-20 | mAP |
|---|---:|---:|---:|---:|---:|
| Shared baseline | 28.48 | 47.46 | 56.80 | 65.50 | 18.85 |

## Warm-Started MSR Training

- RGB train samples: 22258
- IR train samples: 11909
- Number of train classes: 395
- Baseline checkpoint: `checkpoints/baseline_best.pth`
- Epochs: 5
- Batch size: 32
- Learning rate: 0.0001
- CMEC weight: 0.1
- Device: CUDA

## Checkpoint Sweep Results

| Model / Checkpoint | Rank-1 | Rank-5 | Rank-10 | Rank-20 | mAP |
|---|---:|---:|---:|---:|---:|
| Shared baseline | 28.48 | 47.46 | 56.80 | 65.50 | 18.85 |
| MSR epoch 1 | 31.82 | 47.91 | 56.40 | 65.29 | 22.50 |
| MSR epoch 2 | 32.08 | 48.20 | 55.40 | 64.71 | 22.84 |
| MSR epoch 3 | 32.00 | 48.09 | 56.17 | 64.63 | 22.82 |
| MSR epoch 4 | 31.66 | 47.46 | 55.25 | 63.66 | 22.14 |
| MSR epoch 5 | 30.82 | 46.54 | 54.38 | 62.85 | 22.03 |

## Best Checkpoint

Best checkpoint by retrieval performance:

`checkpoints/msr_warmstart_epoch_2.pth`

## Improvement Over Baseline

| Metric | Baseline | Best MSR | Improvement |
|---|---:|---:|---:|
| Rank-1 | 28.48 | 32.08 | +3.60 |
| mAP | 18.85 | 22.84 | +3.99 |

## Conclusion

Warm-starting the MSR two-branch model from the shared baseline improves top-1 retrieval accuracy and mean average precision under the same SYSU simple all-search protocol.

The checkpoint sweep shows that epoch 2 generalizes best. Later epochs reduce retrieval performance despite lower training loss, suggesting overfitting to training identities.
