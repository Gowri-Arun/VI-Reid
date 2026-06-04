# Evaluation Protocol

## Dataset

SYSU-MM01 visible-infrared person re-identification dataset.

## Current Protocol

This repository reports the current main results using a simplified all-search setup.

- Query modality: infrared
- Query cameras: `cam3`, `cam6`
- Gallery modality: visible/RGB
- Gallery cameras: `cam1`, `cam2`, `cam4`, `cam5`
- Official test IDs: 96
- Query samples: 3803
- Gallery samples: 6775

## Metrics

- Rank-1
- Rank-5
- Rank-10
- Rank-20
- mean Average Precision, also called mAP

## Comparison Rule

All model comparisons must use the same:

- Query/gallery split
- Test identity set
- Feature normalization setting
- Distance computation
- Metric implementation

Changing the protocol invalidates direct comparison.
