# Results

## SYSU-MM01 Simple All-Search Results

| Model / Checkpoint | Rank-1 | Rank-5 | Rank-10 | Rank-20 | mAP |
|---|---:|---:|---:|---:|---:|
| Shared baseline | 28.48 | 47.46 | 56.80 | 65.50 | 18.85 |
| MSR epoch 1 | 31.82 | 47.91 | 56.40 | 65.29 | 22.50 |
| MSR epoch 2 | 32.08 | 48.20 | 55.40 | 64.71 | 22.84 |
| MSR epoch 3 | 32.00 | 48.09 | 56.17 | 64.63 | 22.82 |
| MSR epoch 4 | 31.66 | 47.46 | 55.25 | 63.66 | 22.14 |
| MSR epoch 5 | 30.82 | 46.54 | 54.38 | 62.85 | 22.03 |

## Best Result

Best checkpoint:

`checkpoints/msr_warmstart_epoch_2.pth`

## Improvement Over Baseline

| Metric | Baseline | Best MSR | Improvement |
|---|---:|---:|---:|
| Rank-1 | 28.48 | 32.08 | +3.60 |
| mAP | 18.85 | 22.84 | +3.99 |
