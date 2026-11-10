#!/usr/bin/env python
"""Cache instance candidates per image.

--split oof  : held-out-fold training images, using the *fold* models (honest OOF)
--split test : the blind test images, using the full-data model when available (yolo.test_use_full)
--source     : yolo | mask2former | sam2   (sam2 refines every cached source in sam2.refine_sources)
Output: work/preds/<split>/<source>/<stem>.pkl
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from solar.common import guard, image_files, load_config, log, paths, read_prep, resolve, split_stems  # noqa: E402


def stems_for(cfg, P, split):
    if split == "oof":
        return "train", split_stems(cfg, P, "fold")[1]
    files = image_files(P["test_images"])
    limit = int(cfg["data"].get("limit_test_images") or 0)
    return "test", [p.stem for p in (files[:limit] if limit else files)]


def pick(P, split, fold_path: Path, full_path: Path, prefer_full: bool) -> Path:
    if split == "test" and prefer_full and full_path.exists():
        return full_path
    if not fold_path.exists():
        raise SystemExit(f"Missing model {fold_path}")
    return fold_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", choices=["oof", "test"], required=True)
    ap.add_argument("--source", choices=["yolo", "mask2former", "sam2"], required=True)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    P = paths(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    img_split, stems = stems_for(cfg, P, args.split)

    from solar import models as M

    if args.source == "yolo":
        weights = pick(P, args.split, P["models"] / "yolo_fold.pt", P["models"] / "yolo_full.pt", bool(cfg["yolo"]["test_use_full"]))
        model = M.yolo_load(weights)
        runs = {"yolo": lambda gray, _stem: M.yolo_candidates(model, gray, cfg, "0" if device == "cuda" else "cpu")}
    elif args.source == "mask2former":
        weights = pick(P, args.split, P["models"] / "mask2former_fold", P["models"] / "mask2former_full", True)
        model = M.m2f_load(weights, device).eval()
        runs = {"mask2former": lambda gray, _stem: M.m2f_candidates(model, gray, cfg, device)}
    else:
        weights = pick(P, args.split, P["models"] / "sam2_fold_decoder.pt", P["models"] / "sam2_full_decoder.pt", True)
        model = M.sam2_load(resolve(cfg["sam2"]["pretrained"]), device, decoder_state=weights).eval()
        runs = {}
        for src in cfg["sam2"]["refine_sources"]:
            src_dir = P["preds"] / args.split / src
            if not src_dir.exists():
                log(f"[predict] skip sam2[{src}]: no cached {src} predictions for split {args.split}")
                continue

            def refine(gray, stem, _src=src, _dir=src_dir):
                with open(guard(_dir / f"{stem}.pkl"), "rb") as f:
                    base = pickle.load(f)
                return M.sam2_refine(model, gray, base, cfg, device, f"sam2[{_src}]")

            runs[f"sam2_{src}"] = refine
        if not runs:
            raise SystemExit("sam2 needs cached yolo and/or mask2former predictions first")
    log(f"[predict] split={args.split} source={args.source} weights={weights} images={len(stems)}")

    t0 = time.time()
    for name, fn in runs.items():
        out_dir = P["preds"] / args.split / name
        out_dir.mkdir(parents=True, exist_ok=True)
        n_cands = 0
        for i, stem in enumerate(stems, 1):
            out = out_dir / f"{stem}.pkl"
            if out.exists() and not args.overwrite:
                continue
            cands = fn(read_prep(P, img_split, stem), stem)
            n_cands += len(cands)
            tmp = out.with_suffix(".tmp")
            with open(tmp, "wb") as f:
                pickle.dump(cands, f, protocol=pickle.HIGHEST_PROTOCOL)
            tmp.replace(out)
            if i % 20 == 0 or i == len(stems):
                log(f"[predict:{name}:{args.split}] {i}/{len(stems)} images | {n_cands} candidates | "
                    f"{(time.time() - t0) / 60:.1f} min")
    log(f"[predict] done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
