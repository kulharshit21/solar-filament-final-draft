# Artifact release

Large binaries and the exact competition submission are stored in Google Drive under:

```text
Weights Competion Submmsion/
├── README.md
├── weights/
│   ├── yolo_full.pt
│   ├── sam2_fold_decoder.pt
│   ├── sam2_full_decoder.pt
│   └── mask2former_fold/
│       ├── model.safetensors
│       ├── config.json
│       └── astra_val.json
├── submission/
│   └── submission_w11_champion.csv
├── reports/
│   └── validation_report.json
└── manifests/
    ├── ARTIFACT_MANIFEST.json
    ├── SHA256SUMS.txt
    └── WEIGHT_VERIFICATION_100_PASSES.json
```

Drive link: https://drive.google.com/open?id=1qikTNZkhUWEC3vB0Yz2_5l5Zwr1AehFN

Private GitHub repository: https://github.com/kulharshit21/solar-filament-final-draft

Always compare the downloaded file hashes with `SHA256SUMS.txt` before inference or submission. The canonical competition CSV hash is `4ee79cb588f9415eb2f48ebb1a129382f0d7f5e6410724be8b9f6b213b18e512`.

