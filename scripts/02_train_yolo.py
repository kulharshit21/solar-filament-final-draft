#!/usr/bin/env python
"""Train YOLO11x-seg on the clean per-annotator-view dataset.

--mode fold : trains on folds != fold_index (honest OOF model), init from yolo.init_fold
--mode full : trains on all 707 training stems (test model), init from yolo.init_full
Run under CUDA_VISIBLE_DEVICES; --device 0 (single GPU) or 0,1 (ultralytics DDP).
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from solar.common import ROOT, guard, load_config, log, paths, resolve  # noqa: E402

TRAIN_KEYS = (
    "imgsz", "epochs", "batch", "workers", "optimizer", "lr0", "lrf", "weight_decay", "warmup_epochs", "cos_lr",
    "patience", "mosaic", "copy_paste", "mixup", "fliplr", "flipud", "degrees", "translate", "scale", "hsv_h",
    "hsv_s", "hsv_v", "close_mosaic", "overlap_mask", "mask_ratio",
)  # 'time' is added from yolo.time_hours when > 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--mode", required=True, help="dataset mode: fold, full, or fold_N (e.g. fold_0..4)")
    ap.add_argument("--device", default="0")
    args = ap.parse_args()
    cfg = load_config(args.config)
    P = paths(cfg)
    y = cfg["yolo"]

    os.chdir(ROOT)  # ultralytics resolves its AMP-check weights (yolo26n.pt) relative to cwd
    amp_ref = ROOT / "models" / "yolo26n.pt"
    if amp_ref.exists() and not (ROOT / "yolo26n.pt").exists():
        shutil.copy2(amp_ref, ROOT / "yolo26n.pt")

    from ultralytics import YOLO

    init = guard(resolve(y["init_fold"] if args.mode.startswith("fold") else y["init_full"]))
    data = P["yolo"] / args.mode / "data.yaml"
    if not data.exists():
        raise SystemExit(f"Missing {data}; run scripts/01_prepare.py first")
    cache = y.get("cache", "ram")
    train_args = {k: y[k] for k in TRAIN_KEYS if k in y}
    if float(y.get("time_hours", 0) or 0) > 0:
        train_args["time"] = float(y["time_hours"])
    log(f"[yolo:{args.mode}] init={init.name} data={data} device={args.device} {train_args}")

    model = YOLO(str(init))
    model.train(
        data=str(data),
        device=args.device,
        project=str(P["models"] / "yolo_runs"),
        name=args.mode,
        exist_ok=True,
        cache=False if str(cache).lower() == "false" else cache,
        amp=True,
        seed=0,
        deterministic=False,
        plots=False,
        single_cls=True,
        **train_args,
    )
    best = P["models"] / "yolo_runs" / args.mode / "weights" / "best.pt"
    if not best.exists():
        raise SystemExit(f"Training finished without {best}")
    out = P["models"] / f"yolo_{args.mode}.pt"
    shutil.copy2(best, out)
    log(f"[yolo:{args.mode}] saved {out}")


if __name__ == "__main__":
    main()
