#!/usr/bin/env python
"""Read-only pre-flight check: data counts, training-only annotations, train/test disjointness,
training GT statistics, and GPU inventory. Writes nothing except work/verify_report.json."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from solar.common import FORBIDDEN_TOKENS, ROOT, TrainAnnotations, image_files, load_config, log, paths  # noqa: E402


def digest(path: Path) -> str:
    return hashlib.blake2b(path.read_bytes(), digest_size=16).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--skip-hash", action="store_true", help="skip the content-hash disjointness check")
    args = ap.parse_args()
    cfg = load_config(args.config)
    P = paths(cfg)

    train_files = image_files(P["train_images"])
    test_files = image_files(P["test_images"])
    A = TrainAnnotations(P["ann"])
    train_stems = {p.stem for p in train_files}
    test_stems = {p.stem for p in test_files}
    scan_dirs = ["data", "scripts", "src", "models", "work"]
    forbidden = []
    for d in scan_dirs:
        dp = ROOT / d
        if dp.exists():
            for p in dp.rglob("*"):
                if p.is_file() and any(t in p.name.lower() for t in FORBIDDEN_TOKENS):
                    forbidden.append(str(p.relative_to(ROOT)))

    checks = {
        "707 training images on disk": len(train_files) == 707,
        "180 test images on disk": len(test_files) == 180,
        "training JSON covers exactly the 707 training stems": set(A.stems) == train_stems,
        "training JSON names no test stem": not (set(A.stems) & test_stems),
        "train/test file names disjoint": not (train_stems & test_stems),
        "no forbidden (test-label) artifacts in package": not forbidden,
        "training annotations count exactly 8,199 (zero test GT leaks)": A.n_annotations == 8199,
        "annotator views count exactly 1,154": len(A.views) == 1154,
    }
    if not args.skip_hash:
        test_hashes = {digest(p) for p in test_files}
        checks["train/test image contents disjoint"] = not (test_hashes & {digest(p) for p in train_files})

    areas = np.asarray([a["area"] for anns in A.anns.values() for a in anns], np.float64)
    per_view = np.asarray([len(A.anns.get(v, [])) for v in A.views], np.int64)
    views_per_stem = np.asarray([len(v) for v in A.stem_views.values()], np.int64)
    stats = {
        "training_images": len(A.stems),
        "annotator_views": len(A.views),
        "annotations": A.n_annotations,
        "views_per_image": {str(k): int((views_per_stem == k).sum()) for k in sorted(set(views_per_stem.tolist()))},
        "instances_per_view": {q: float(np.percentile(per_view, p)) for q, p in (("min", 0), ("median", 50), ("p90", 90), ("max", 100))},
        "area_percentiles": {f"p{p:02d}": float(np.percentile(areas, p)) for p in (1, 5, 10, 25, 50, 75, 95, 99)},
    }

    gpus = []
    try:
        import torch

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                prop = torch.cuda.get_device_properties(i)
                gpus.append({"index": i, "name": prop.name, "memory_gb": round(prop.total_memory / 1e9, 1)})
        torch_info = {"torch": torch.__version__, "cuda": torch.version.cuda, "available": torch.cuda.is_available()}
    except Exception as exc:  # pragma: no cover - reported, not fatal
        torch_info = {"error": repr(exc)}
    try:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        smi = "nvidia-smi unavailable"

    log("=" * 72)
    log("PRE-FLIGHT VERIFICATION (read-only)")
    log("=" * 72)
    for name, ok in checks.items():
        log(f"  {'PASS' if ok else 'FAIL'}  {name}")
    log(f"  training images {stats['training_images']} | annotator views {stats['annotator_views']} | "
        f"annotations {stats['annotations']}")
    log("  NOTE: the official training JSON has 8,199 annotations. A count of ~10,244 would mean")
    log("        test annotations were mixed in (10,244 - 8,199 = 2,045 = test GT count).")
    log(f"  views per image {stats['views_per_image']}")
    log(f"  instances per view {stats['instances_per_view']}")
    log(f"  training GT area percentiles {stats['area_percentiles']}")
    log(f"  min-area candidate from training data only: p05 = {stats['area_percentiles']['p05']:.0f} px")
    log(f"  torch {torch_info}")
    log(f"  GPUs visible to torch: {gpus}")
    log(f"  nvidia-smi:\n{smi}")
    if forbidden:
        log(f"  forbidden artifacts: {forbidden}")

    report = {"checks": checks, "stats": stats, "torch": torch_info, "gpus": gpus, "forbidden": forbidden}
    (P["work"] / "verify_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not all(checks.values()):
        raise SystemExit("Verification FAILED - fix the data before training.")
    log("Verification PASSED.")


if __name__ == "__main__":
    main()
