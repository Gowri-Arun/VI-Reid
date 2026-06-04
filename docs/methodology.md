# Methodology

## Shared Baseline

The shared baseline uses a single ResNet-50 based feature extractor for both RGB and infrared images.

The model is trained with identity classification loss.

## Warm-Started MSR

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

## Training Objective

The warm-started MSR model is trained with:

- RGB identity classification loss
- IR identity classification loss
- RGB shared identity loss
- IR shared identity loss
- Cross-Modality Euclidean Constraint, also called CMEC

## Why Warm-Start?

Training a two-branch RGB/IR model from scratch can cause the two branches to drift into poorly aligned feature spaces.

Warm-starting both branches from the shared baseline gives the model a stable aligned initialization, then allows RGB and IR branches to specialize.
