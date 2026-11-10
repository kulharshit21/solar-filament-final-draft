# FINAL DRAFT — Solar Filament Segmentation 0.40 Pipeline

This private repository contains the code-only release associated with the verified **0.40 Kaggle public leaderboard submission** for the Solar Filament Segmentation Challenge 2026.

The model weights, exact submitted CSV, and their checksum manifests are deliberately stored outside GitHub in the separate Google Drive artifact release documented in [ARTIFACTS.md](ARTIFACTS.md). No competition dataset, model weight, cached prediction, credential, or submission CSV is committed here.

## What produced the 0.40 result

The archived weights also supported nearby 0.39 submissions. The improvement to 0.40 came from the exact selector and fusion policy preserved in `scripts/generate_w11.py`:

- equal YOLO/Mask2Former fusion: `w=1:1`;
- mask threshold: `0.45`;
- score threshold: `0.53`;
- minimum instance area: `319` pixels (`p05` from training annotations);
- maximum instance area: `70,502` pixels;
- disjoint-mask arbitration before RLE encoding.

The exact historical CSV contains **1,261 instances across all 180 test images**, with no overlapping instance pixels. Its SHA-256 is:

```text
4ee79cb588f9415eb2f48ebb1a129382f0d7f5e6410724be8b9f6b213b18e512
```

## Repository layout

```text
FINAL DRAFT/
├── config.yaml
├── requirements.txt
├── scripts/
│   ├── 00_verify.py
│   ├── 01_prepare.py
│   ├── 02_train_yolo.py
│   ├── 03_train_mask2former.py
│   ├── 04_train_sam2.py
│   ├── 05_predict.py
│   ├── 06_select_and_submit.py
│   ├── generate_w11.py
│   ├── verify_code_release.py
│   └── verify_package.py
├── src/solar/
├── reports/validation_report.json
├── ARTIFACTS.md
├── INTEGRITY.md
└── KAGGLE_WRITEUP.md
```

## Reproduction

1. Use only the official Kaggle competition data and place it under the paths expected by `src/solar/common.py`.
2. Install the pinned minimum dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

3. Download the separately released weights and verify every file against the Drive `SHA256SUMS.txt` manifest.
4. Run the integrity preflight:

   ```bash
   python scripts/00_verify.py
   python scripts/verify_code_release.py
   ```

5. Generate predictions with `scripts/05_predict.py`, then apply the locked 0.40 policy:

   ```bash
   python scripts/generate_w11.py
   ```

The quick generation step requires the cached test candidates produced by `05_predict.py`. Full training commands and model stages are represented by scripts `01` through `06`.

## Important provenance note

The historical source files in `scripts/`, `src/`, `config.yaml`, and `reports/validation_report.json` are preserved byte-for-byte from the recovered 0.40 package. The repository documentation and code-release verifier were added for this clean publication. See [INTEGRITY.md](INTEGRITY.md) for the evidence boundary.

