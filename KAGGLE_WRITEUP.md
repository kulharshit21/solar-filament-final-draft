# Kaggle write-up — calibrated multi-model solar-filament segmentation

## Summary

Our final verified approach combines a high-resolution YOLO11x segmentation model with Mask2Former Swin-Large global context and SAM 2.1 boundary refinement. The principal leaderboard improvement did not come from changing the archived weights; it came from applying the correct out-of-fold-calibrated fusion and instance-selection policy.

## Method

1. Group observations by date before validation splitting to reduce temporal leakage.
2. Train YOLO segmentation at high spatial resolution for strong instance recall.
3. Use Mask2Former to recover global filament topology.
4. Use SAM 2.1 as a box-prompted boundary refiner.
5. Calibrate mask, confidence, and area thresholds only on training-derived OOF predictions.
6. Fuse YOLO and Mask2Former equally (`w=1:1`).
7. Resolve overlaps so each pixel belongs to at most one predicted instance.

The locked final selector uses `mask_thr=0.45`, `score_thr=0.53`, `min_area=319`, and `max_area=70502`. It produces 1,261 instances across 180 test images.

## Reproducibility

- Private code repository: https://github.com/kulharshit21/solar-filament-final-draft
- Verified weights and exact submission: https://drive.google.com/open?id=1qikTNZkhUWEC3vB0Yz2_5l5Zwr1AehFN
- Exact submission SHA-256: `4ee79cb588f9415eb2f48ebb1a129382f0d7f5e6410724be8b9f6b213b18e512`

The repository excludes model binaries and submission files. The Drive release includes a SHA-256 manifest and a report showing 100 consecutive full weight-verification passes. An independent resubmission of the canonical CSV on 2026-11-10 completed with a public score of `0.40`.

## Integrity

The work uses the official competition training data for training and calibration and blind test images for inference only. It does not use test labels or organizer-only metadata. Dataset files are not redistributed.

