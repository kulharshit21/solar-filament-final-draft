# FINAL DRAFT - Solar Filament Segmentation Challenge 2026

This is the clean, code-only release for the rules-compliant pipeline associated with our **0.40 public leaderboard submission**. The canonical submission was independently resubmitted to Kaggle on 10 November 2026 and returned:

```text
status: COMPLETE
public score: 0.40
competition: filament-segmentation-2026
```

This release intentionally separates reproducible source from large binary artifacts:

- **Private GitHub:** source code, configuration, documentation, tests, and integrity checks only.
- **Google Drive:** exact historical weights, exact submission CSV, validation report, Kaggle score proof, and SHA-256 manifests.

## Links

- Private code repository: https://github.com/kulharshit21/solar-filament-final-draft
- Verified weights and submission folder: https://drive.google.com/open?id=1qikTNZkhUWEC3vB0Yz2_5l5Zwr1AehFN
- Competition: https://www.kaggle.com/competitions/filament-segmentation-2026

The Drive artifact folder is named `Weights Competion Submmsion` to match the requested handoff name. The exact submission is `submission/submission_w11_champion.csv`.

## 1. Project objective

Solar filaments are elongated, low-contrast absorption structures in H-alpha imagery. The task is instance segmentation: each filament must be returned as a separate, valid run-length encoded mask. The official evaluator compares predictions against each human rater independently, so false positives, fragmentation, over-merging, and missed filaments all affect Panoptic Quality (PQ).

The central modeling decision was therefore not simply maximizing detector recall. We optimized the complete chain:

```text
official training data
        |
        v
date-grouped folds and annotator-aware targets
        |
        +--> YOLO11x-seg: high-resolution local instance proposals
        |
        +--> Mask2Former Swin-Large: global topology and context
        |
        +--> SAM 2.1 Hiera-L: box-prompted optical-boundary refinement
        |
        v
OOF candidate generation -> threshold/fusion calibration -> disjoint masks -> RLE CSV
```

## 2. Complete work chronology

### Audit and recovery

1. Inspected the competition evaluator and reproduced its matching/PQ behavior locally.
2. Recovered the historical clean baseline package and compared candidate submissions, weights, validation reports, and manifests.
3. Established the integrity rule: no test labels, hidden annotations, cluster metadata, or prohibited organizer artifacts.
4. Preserved unrelated DGX work and removed only project-specific temporary Solar files after verified Drive backups.

### Validation design

1. Parsed the official training annotation JSON.
2. Grouped observations by date so related observations cannot cross validation boundaries.
3. Kept annotation views/rater identity explicit instead of collapsing disagreement into a single synthetic mask.
4. Measured TP, FP, FN, PQ, SQ, RQ, sum-IoU, overlap pixels, and RLE decodability.
5. Used OOF predictions to select a plateau policy rather than selecting a one-off validation spike.

### Model training

#### YOLO11x-seg

- Native high-resolution segmentation model for local filament proposals.
- Full-data model trained at high spatial resolution.
- Historical champion checkpoint reported approximately 0.788 mask mAP50 and 0.792 box mAP50 on its training validation workflow.
- The model is effective at finding narrow instances and suppressing chromospheric clutter.

#### Mask2Former Swin-Large

- COCO-initialized Swin-Large pixel backbone and universal segmentation decoder.
- Fine-tuned for one filament foreground class plus background/no-object handling.
- 1280-pixel training/inference canvas in the historical package.
- Global attention helps preserve long filament topology and reduces fragmented local detections.

#### SAM 2.1 Hiera-L

- Frozen image/prompt encoders with a fine-tuned mask decoder.
- Box-prompted crop refinement around candidate filaments.
- Optical absorption-valley structure was used to preserve narrow spines and barbs without aggressive erosion.

### Post-processing and calibration

The archived weights appeared in nearby 0.39 and 0.40 submissions. The measured lift came from the selector/fusion policy applied to the same model family:

```text
variant       = ens[avg[yolo]|mask2former;w=1:1]
mask_thr      = 0.45
score_thr     = 0.53
min_area      = 319 pixels (training-derived p05)
max_area      = 70,502 pixels
CRF           = disabled
overlap rule  = disjoint pixel arbitration
```

Every candidate is filtered by confidence, area, and topology checks. Overlapping masks are arbitrated so each pixel belongs to at most one output instance. RLEs are encoded in the official expected order and validated by decoding them again.

## 3. Architecture in detail

### Data and target layer

- Official training images and annotation JSON are read from the configured data root.
- Each rater view remains a separate supervision instance.
- Date-grouped folds prevent temporal leakage across nearby observations.
- Training-only area statistics define the minimum-area policy.

### Detection layer

YOLO11x-seg supplies dense instance candidates with local boundaries and confidence scores. Its high-resolution path is particularly important for thin filaments and small barbs.

### Global segmentation layer

Mask2Former receives the full resized image and predicts masks through a Swin-Large pixel decoder and transformer mask/class predictors. Its receptive field supplies long-range context when local contrast is ambiguous.

### Boundary refinement layer

SAM 2.1 receives candidate boxes and refines their boundaries on a context crop. The decoder is fine-tuned while the expensive image encoder remains frozen in the historical design.

### Fusion layer

YOLO candidates are averaged within their model family, then fused with the native Mask2Former candidates using equal weights. The OOF-calibrated score threshold controls cross-model agreement. The final selector performs area filtering, maximum-area rejection, duplicate arbitration, and deterministic RLE encoding.

### Evaluation layer

The official evaluator is multi-rater and IoU-cliff based. A prediction must match at IoU >= 0.50 to become a TP; unmatched predictions become FPs and unmatched human instances become FNs. PQ combines recognition quality and segmentation quality, so reducing false positives and fragmentation is as important as improving mask boundaries.

## 4. Reported evidence for the 0.40 package

The recovered validation report records:

| Quantity | Value |
|---|---:|
| OOF PQ | 0.4622 |
| OOF SQ | 0.6857 |
| OOF RQ | 0.6741 |
| OOF TP / FP / FN | 1031 / 454 / 543 |
| Test images | 180 / 180 |
| Submission instances | 1,261 |
| Overlap pixels | 0 |
| Undecodable RLEs | 0 |
| Historical Kaggle public score | 0.40 |

The exact submission SHA-256 is:

```text
4ee79cb588f9415eb2f48ebb1a129382f0d7f5e6410724be8b9f6b213b18e512
```

## 5. Exact sequence of implementation steps

1. Inventory official data, models, recovered packages, and previous submissions.
2. Reproduce the evaluator and validate metric parity.
3. Build date-grouped folds and training statistics.
4. Prepare identical image normalization for train and test.
5. Train YOLO11x-seg on high-resolution masks.
6. Train Mask2Former Swin-Large with decoder/backbone learning-rate separation.
7. Fine-tune the SAM 2.1 decoder on box-prompted context crops.
8. Produce honest OOF candidates for every model source.
9. Sweep mask, confidence, area, and fusion policies only on OOF data.
10. Freeze the best plateau policy and apply it once to the blind test split.
11. Validate row schema, unique filament IDs, RLE decodability, image coverage, and pixel disjointness.
12. Record hashes, archive the exact submission, and upload artifacts to Drive.
13. Submit the canonical CSV to Kaggle and record the returned score.

## 6. Reproduction commands

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run code-only integrity checks:

```bash
python scripts/verify_code_release.py
python -m compileall -q scripts src
```

For a full reproduction with official data and separately downloaded weights:

```bash
python scripts/00_verify.py
python scripts/01_prepare.py
CUDA_VISIBLE_DEVICES=0 python scripts/02_train_yolo.py --mode full --device 0
CUDA_VISIBLE_DEVICES=0 python scripts/03_train_mask2former.py --mode fold --device cuda:0
CUDA_VISIBLE_DEVICES=0 python scripts/04_train_sam2.py --mode fold --device cuda:0
CUDA_VISIBLE_DEVICES=0 python scripts/05_predict.py --split test --source yolo --overwrite
CUDA_VISIBLE_DEVICES=0 python scripts/05_predict.py --split test --source mask2former --overwrite
CUDA_VISIBLE_DEVICES=0 python scripts/05_predict.py --split test --source sam2 --overwrite
python scripts/generate_w11.py
```

The historical package no longer contains its original blind-test prediction caches. Therefore the generator was audited locally and correctly stopped with a missing `data/test_images` error rather than silently fabricating a new CSV. The canonical byte-identical CSV remains preserved, hash-verified, and independently scored 0.40 by Kaggle.

## 7. Integrity and Kaggle compliance

- No test labels were used for training or calibration.
- No hidden organizer annotations or prohibited cluster files are included.
- The GitHub repository contains no dataset, weights, credentials, cached predictions, or submission CSV.
- Drive artifacts contain only the verified weights, exact submission, reports, and manifests.
- All submission masks are disjoint and RLE-decodable.
- The Drive manifest records the source hashes and the Kaggle score proof.

## 8. Artifact handoff

Download `Weights Competion Submmsion` from Drive and verify:

```bash
python verify_artifacts.py
```

The canonical 0.40 file is `submission/submission_w11_champion.csv`. Do not substitute the newer Python4 candidate; that separate candidate returned 0.38 and is intentionally excluded from this final draft.

