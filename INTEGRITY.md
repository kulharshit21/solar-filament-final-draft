# Competition integrity and provenance

## Data boundary

- Training uses the official competition training images and annotations only.
- Test images are used only for inference.
- No test labels, hidden organizer annotations, cluster labels, or prohibited leakage artifacts are included or consumed.
- The official competition dataset is not redistributed through GitHub or the artifact release.
- The source contains explicit guards for known forbidden artifact names.

## Release separation

- GitHub contains code and documentation only.
- Google Drive contains model weights, the exact historical submission, validation evidence, and checksum manifests.
- Credentials and Kaggle API tokens are excluded from both releases.

## Historical evidence

The recovered package records:

- public leaderboard PQ: `0.40`;
- submission rows: `1,261`;
- test images represented: `180`;
- OOF PQ: `0.4622`;
- OOF SQ: `0.6857`;
- OOF RQ: `0.6741`;
- OOF TP/FP/FN: `1031 / 454 / 543`.

The exact submission was independently resubmitted to Kaggle under the description `FINAL DRAFT exact recovered 0.40 w1:1 champion SHA256 4ee79cb5`. Kaggle returned status `COMPLETE` and public score `0.40`. That result is authoritative; local reports are supporting evidence only.

## Core historical hashes

The verifier checks the recovered source files against their archived SHA-256 values. Any change to those historical files is reported as a failure. Publication-only documentation and `verify_code_release.py` are outside the historical hash set.

