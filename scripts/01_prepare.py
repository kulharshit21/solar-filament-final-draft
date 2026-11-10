#!/usr/bin/env python
"""Prepare clean training assets.

* Date-grouped 5-fold split of the 707 training stems  -> work/folds.csv
* Training-data statistics (GT area percentiles)      -> work/train_stats.json
* Identically preprocessed train/test images          -> work/prep/{train,test}/<stem>.png
* YOLO-seg datasets, one sample per annotator view    -> work/yolo/{fold,full}/
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from solar.common import (  # noqa: E402
    IMG, TrainAnnotations, image_files, link_or_copy, load_config, log, make_folds, paths, prep_path,
    preprocess, read_gray,
)


def _prep_one(job):
    src, dst, cfg = job
    if dst.exists():
        return 0
    gray = read_gray(src)
    if gray.shape != (IMG, IMG):
        raise RuntimeError(f"Unexpected image size {gray.shape}: {src}")
    out = preprocess(gray, cfg)
    if not cv2.imwrite(str(dst), out):
        raise RuntimeError(f"Could not write {dst}")
    return 1


def yolo_label_lines(A: TrainAnnotations, view: str) -> list[str]:
    info = A.views[view]
    w, h = float(info["width"]), float(info["height"])
    lines = []
    for ann in A.anns.get(view, []):
        for poly in A.polygons(ann):
            pts = np.clip(poly / np.asarray([w, h], np.float32), 0.0, 1.0)
            if len(pts) < 3:
                continue
            lines.append("0 " + " ".join(f"{v:.6f}" for v in pts.reshape(-1)))
    return lines


def build_yolo(P, A, mode: str, train_stems, val_stems) -> Path:
    root = P["yolo"] / mode
    if root.exists():
        shutil.rmtree(root)
    for split, stems in (("train", train_stems), ("val", val_stems)):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        for stem in stems:
            for view in A.stem_views[stem]:
                link_or_copy(prep_path(P, "train", stem), root / "images" / split / f"{view}.png")
                (root / "labels" / split / f"{view}.txt").write_text("\n".join(yolo_label_lines(A, view)) + "\n", encoding="utf-8")
    yaml_path = root / "data.yaml"
    yaml_path.write_text(
        f"path: {root.resolve().as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: filament\n",
        encoding="utf-8",
    )
    return yaml_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--all-folds", action="store_true", help="build datasets for all N folds")
    args = ap.parse_args()
    cfg = load_config(args.config)
    P = paths(cfg)
    A = TrainAnnotations(P["ann"])
    d = cfg["data"]

    stems = A.stems
    if d.get("limit_train_stems"):
        stems = stems[: int(d["limit_train_stems"])]
    folds = make_folds(stems, int(d["n_splits"]), int(d["group_chars"]))
    k = int(d["fold_index"])
    frame = pd.DataFrame(
        {
            "stem": sorted(folds),
            "date": [s[: int(d["group_chars"])] for s in sorted(folds)],
            "fold": [folds[s] for s in sorted(folds)],
            "n_views": [len(A.stem_views[s]) for s in sorted(folds)],
        }
    )
    frame.to_csv(P["work"] / "folds.csv", index=False)
    by_date = frame.groupby("date").fold.nunique()
    if (by_date > 1).any():
        raise RuntimeError("A date group was split across folds")
    log(f"[folds] {len(frame)} stems, {frame.date.nunique()} dates, stems per fold "
        f"{frame.groupby('fold').size().to_dict()}, views per fold {frame.groupby('fold').n_views.sum().to_dict()}")

    train_stems = sorted(s for s, f in folds.items() if f != k)
    val_stems = sorted(s for s, f in folds.items() if f == k)
    if d.get("limit_val_stems"):
        val_stems = val_stems[: int(d["limit_val_stems"])]
        frame = frame[frame.stem.isin(train_stems) | frame.stem.isin(val_stems)]
        frame.to_csv(P["work"] / "folds.csv", index=False)
    areas_train = np.asarray([a["area"] for s in train_stems for v in A.stem_views[s] for a in A.anns.get(v, [])], np.float64)
    areas_all = np.asarray([a["area"] for s in stems for v in A.stem_views[s] for a in A.anns.get(v, [])], np.float64)
    stats = {
        "fold_index": k,
        "p05": float(np.percentile(areas_train, 5)),
        "area_percentiles_train_folds": {f"p{p:02d}": float(np.percentile(areas_train, p)) for p in (1, 5, 10, 25, 50, 75, 95)},
        "area_percentiles_all_training": {f"p{p:02d}": float(np.percentile(areas_all, p)) for p in (1, 5, 10, 25, 50, 75, 95)},
        "area_max_train_folds": float(areas_train.max()),
        "max_area": int(round(float(cfg["select"].get("max_area_factor", 2.0)) * float(areas_train.max()))),
        "n_train_stems": len(train_stems),
        "n_val_stems": len(val_stems),
    }
    (P["work"] / "train_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    log(f"[stats] min-area candidate p05 = {stats['p05']:.0f} px | max area = {stats['max_area']} px "
        f"(training folds only)")

    test_files = image_files(P["test_images"])
    if d.get("limit_test_images"):
        test_files = test_files[: int(d["limit_test_images"])]
    for split in ("train", "test"):
        (P["prep"] / split).mkdir(parents=True, exist_ok=True)
    train_files = {p.stem: p for p in image_files(P["train_images"])}
    jobs = [(train_files[s], prep_path(P, "train", s), cfg) for s in sorted(folds)]
    jobs += [(p, prep_path(P, "test", p.stem), cfg) for p in test_files]
    with ProcessPoolExecutor(max_workers=max(1, int(d["workers"]))) as ex:
        written = sum(ex.map(_prep_one, jobs, chunksize=4))
    log(f"[prep] mode={d['preprocess']} wrote {written} new images ({len(jobs)} total)")

    fold_yaml = build_yolo(P, A, "fold", train_stems, val_stems)
    full_yaml = build_yolo(P, A, "full", sorted(folds), val_stems)
    n_fold = sum(len(A.stem_views[s]) for s in train_stems)
    n_full = sum(len(A.stem_views[s]) for s in folds)
    log(f"[yolo] fold dataset {fold_yaml} ({n_fold} train views)")
    log(f"[yolo] full dataset {full_yaml} ({n_full} train views; its val split overlaps train and is monitoring only)")
    if args.all_folds:
        for f_idx in range(int(d["n_splits"])):
            tr_stems = sorted(s for s, f in folds.items() if f != f_idx)
            v_stems = sorted(s for s, f in folds.items() if f == f_idx)
            build_yolo(P, A, f"fold_{f_idx}", tr_stems, v_stems)
            log(f"[yolo] built fold_{f_idx} dataset ({len(tr_stems)} train stems, {len(v_stems)} val stems)")
    log("[done] preparation complete")


if __name__ == "__main__":
    main()
