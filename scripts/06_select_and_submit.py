#!/usr/bin/env python
"""Choose the post-processing policy on the held-out training fold, then build the test CSV.

Variants evaluated on OOF (only those whose caches exist):
  <src>            native masks (yolo, mask2former)
  sam2[<src>]      SAM2-refined masks for that source's boxes
  avg[<src>]       pixel average of native and SAM2-refined masks
  ens[...]         cross-model fusion of one yolo-family and one mask2former-family variant;
                   a fused score is the weighted mean over BOTH models (a missing model counts 0),
                   so the score threshold decides how much cross-model agreement is required.
Each variant is swept over mask thresholds x minimum areas x score thresholds with the exact
multi-annotator PQ. The policy is the best plateau (not spike) of OOF PQ. Optional DenseCRF is
kept only if it raises OOF PQ. The policy is then applied once, unchanged, to the test images.
No test labels exist in this package and none are read.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from solar.common import (  # noqa: E402
    TrainAnnotations, average_pairs, final_masks, fuse_sources, guard, image_files, load_config, log, paths,
    pq_from, read_prep, rle_encode, split_stems, strip_cache, sweep_image, train_stats, validate_submission,
)

FAMILIES = ("yolo", "mask2former")
_G: dict = {}


def available_sources(P, split, stems) -> list[str]:
    names = []
    for name in ("yolo", "mask2former", "sam2_yolo", "sam2_mask2former"):
        d = P["preds"] / split / name
        if d.exists() and all((d / f"{s}.pkl").exists() for s in stems):
            names.append(name)
    return names


def build_variants(sources: list[str], cfg: dict) -> list[dict]:
    fam = {f: [] for f in FAMILIES}
    for f in FAMILIES:
        if f in sources:
            fam[f].append({"name": f, "kind": "native", "src": f})
            if f"sam2_{f}" in sources:
                fam[f].append({"name": f"sam2[{f}]", "kind": "sam2", "src": f})
                fam[f].append({"name": f"avg[{f}]", "kind": "avg", "src": f})
    variants = fam["yolo"] + fam["mask2former"]
    for a in fam["yolo"]:
        for b in fam["mask2former"]:
            for w in cfg["select"]["fuse_weights"]:
                variants.append({"name": f"ens[{a['name']}|{b['name']};w={w[0]}:{w[1]}]", "kind": "ens",
                                 "parts": [a, b], "w": [float(w[0]), float(w[1])]})
    return variants


def load_sources(P, split, stem, sources) -> dict:
    out = {}
    for name in sources:
        with open(guard(P["preds"] / split / name / f"{stem}.pkl"), "rb") as f:
            out[name] = pickle.load(f)
    return out


def make_variant(v: dict, S: dict, cfg: dict, memo: dict) -> list[dict]:
    if v["name"] in memo:
        return memo[v["name"]]
    if v["kind"] == "native":
        out = S[v["src"]]
    elif v["kind"] == "sam2":
        out = S[f"sam2_{v['src']}"]
    elif v["kind"] == "avg":
        out = average_pairs(S[v["src"]], S[f"sam2_{v['src']}"])
    else:
        parts = [strip_cache(make_variant(p, S, cfg, memo)) for p in v["parts"]]
        out = fuse_sources(parts, v["w"], float(cfg["select"]["fuse_iou"]))
    memo[v["name"]] = out
    return out


def crf_refine(cands: list[dict], gray: np.ndarray, cfg: dict) -> list[dict]:
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax

    s = cfg["select"]
    out = []
    H, W = gray.shape
    for c in cands:
        y1, y2, x1, x2 = c["win"]
        Y1, Y2, X1, X2 = max(0, y1 - 8), min(H, y2 + 8), max(0, x1 - 8), min(W, x2 + 8)
        p = np.zeros((Y2 - Y1, X2 - X1), np.float32)
        p[y1 - Y1 : y2 - Y1, x1 - X1 : x2 - X1] = c["prob"].astype(np.float32)
        p = np.clip(p, 1e-4, 1 - 1e-4)
        h, w = p.shape
        d = dcrf.DenseCRF2D(w, h, 2)
        d.setUnaryEnergy(np.ascontiguousarray(unary_from_softmax(np.stack([1 - p, p]))))
        d.addPairwiseGaussian(sxy=3, compat=3)
        rgb = np.ascontiguousarray(np.repeat(gray[Y1:Y2, X1:X2, None], 3, axis=2))
        d.addPairwiseBilateral(sxy=float(s["crf_sxy"]), srgb=float(s["crf_srgb"]), rgbim=rgb, compat=10)
        q = np.asarray(d.inference(int(s["crf_iters"]))).reshape(2, h, w)[1]
        out.append({**{k: v for k, v in c.items() if not isinstance(k, tuple)}, "win": (Y1, Y2, X1, X2),
                    "prob": q.astype(np.float16)})
    return out


def _init(state):
    _G.update(state)


def _sweep_job(job):
    stem, gt_views = job
    cfg = _G["cfg"]
    S = load_sources(_G["P"], "oof", stem, _G["sources"])
    memo: dict = {}
    res = np.zeros((len(_G["variants"]), len(_G["mts"]), len(_G["mas"]), 4, len(_G["grid"])), np.float32)
    trim = _G.get("trim", False)
    for vi, v in enumerate(_G["variants"]):
        cands = make_variant(v, S, cfg, memo)
        for ti, mt in enumerate(_G["mts"]):
            for ai, ma in enumerate(_G["mas"]):
                res[vi, ti, ai] = sweep_image(cands, gt_views, mt, ma, _G["grid"], _G["max_area"], trim=trim)
    return res


def _crf_job(job):
    stem, gt_views = job
    cfg = _G["cfg"]
    S = load_sources(_G["P"], "oof", stem, _G["sources"])
    trim = _G.get("trim", False)
    cands = crf_refine(strip_cache(make_variant(_G["variant"], S, cfg, {})), read_prep(_G["P"], "train", stem), cfg)
    return sweep_image(cands, gt_views, _G["mt"], _G["ma"], _G["grid"], _G["max_area"], trim=trim).astype(np.float32)


def smooth(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        return x
    out = np.zeros_like(x)
    cnt = np.zeros(x.shape[-1])
    T = x.shape[-1]
    for o in range(-k, k + 1):
        lo, hi = max(0, -o), min(T, T - o)
        out[..., lo:hi] += x[..., lo + o : hi + o]
        cnt[lo:hi] += 1
    return out / cnt


def pq_grid(stats: np.ndarray) -> np.ndarray:
    tp, fp, fn, s = stats[..., 0, :], stats[..., 1, :], stats[..., 2, :], stats[..., 3, :]
    den = tp + 0.5 * fp + 0.5 * fn
    return np.divide(s, den, out=np.zeros_like(s, dtype=np.float64), where=den > 0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--full-oof", action="store_true", help="evaluate on all 707 training stems if complete OOF exists")
    args = ap.parse_args()
    cfg = load_config(args.config)
    P = paths(cfg)
    sel = cfg["select"]
    t0 = time.time()

    A = TrainAnnotations(P["ann"])
    if args.full_oof:
        val_stems = sorted(A.stems)
    else:
        val_stems = split_stems(cfg, P, "fold")[1]
    stats = train_stats(P)
    sources = available_sources(P, "oof", val_stems)
    if not sources:
        raise SystemExit("No complete OOF prediction caches found; run scripts/05_predict.py --split oof first")
    variants = build_variants(sources, cfg)
    mts = [float(v) for v in sel["mask_thresholds"]]
    ma_names = [str(v) for v in sel["min_areas"]]
    mas = [int(round(stats["p05"])) if v == "p05" else int(v) for v in ma_names]
    lo, hi, step = (float(v) for v in sel["score_grid"])
    grid = np.round(np.arange(lo, hi + step / 2, step), 4)
    gts = {s: A.gt_flat_views(s) for s in val_stems}
    n_gt = sum(len(x) for v in gts.values() for x in v)
    log(f"[select] OOF images {len(val_stems)} | GT instances {n_gt} | sources {sources} | {len(variants)} variants x "
        f"{len(mts)} mask thr x {len(mas)} min areas {dict(zip(ma_names, mas))} x {len(grid)} score thr")

    max_area = int(stats["max_area"])
    trim = bool(sel.get("topology_guarded_trim", False))
    state = {"cfg": cfg, "P": P, "sources": sources, "variants": variants, "mts": mts, "mas": mas, "grid": grid,
             "max_area": max_area, "trim": trim}
    jobs = [(s, gts[s]) for s in val_stems]
    with ProcessPoolExecutor(max_workers=max(1, int(cfg["data"]["workers"])), initializer=_init, initargs=(state,)) as ex:
        per_image = np.stack(list(ex.map(_sweep_job, jobs)))  # (I, V, MT, MA, 4, T)
    total = per_image.sum(0)
    pq = pq_grid(total)
    pq_s = smooth(pq, int(sel["smooth"]))
    vi, ti, ai, gi = (int(v) for v in np.unravel_index(int(np.argmax(pq_s)), pq_s.shape))
    best = {"variant": variants[vi]["name"], "mask_thr": mts[ti], "min_area": mas[ai], "min_area_rule": ma_names[ai],
            "max_area": max_area, "score_thr": float(grid[gi]), "crf": False, **pq_from(total[vi, ti, ai, :, gi])}

    rows = []
    for v_i, v in enumerate(variants):
        for t_i, mt in enumerate(mts):
            for a_i, ma in enumerate(mas):
                g = int(np.argmax(pq_s[v_i, t_i, a_i]))
                rows.append({"variant": v["name"], "mask_thr": mt, "min_area": ma, "score_thr": float(grid[g]),
                             "PQ_plateau": float(pq_s[v_i, t_i, a_i, g]), **pq_from(total[v_i, t_i, a_i, :, g])})
    table = pd.DataFrame(rows).sort_values("PQ_plateau", ascending=False)
    table.to_csv(P["work"] / "oof_selection.csv", index=False)
    per_variant = table.groupby("variant", sort=False).head(1)
    log("[select] best OOF configuration per variant:")
    for r in per_variant.itertuples():
        log(f"   {r.variant:<55} PQ {r.PQ:.4f} SQ {r.SQ:.4f} RQ {r.RQ:.4f} TP {r.TP} FP {r.FP} FN {r.FN} "
            f"(mask {r.mask_thr}, area {r.min_area}, score {r.score_thr:.2f})")

    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(val_stems), size=(1000, len(val_stems)))
    boot = [pq_from(per_image[i, vi, ti, ai, :, gi].sum(0))["PQ"] for i in idx]
    best["bootstrap_PQ_p05"], best["bootstrap_PQ_p95"] = float(np.percentile(boot, 5)), float(np.percentile(boot, 95))

    if sel.get("try_crf"):
        try:
            import pydensecrf.densecrf  # noqa: F401
            crf_ok = True
        except Exception:
            crf_ok = False
            log("[select] pydensecrf not importable; DenseCRF skipped")
        if crf_ok:
            crf_state = {**state, "variant": variants[vi], "mt": mts[ti], "ma": mas[ai]}
            with ProcessPoolExecutor(max_workers=max(1, int(cfg["data"]["workers"])), initializer=_init, initargs=(crf_state,)) as ex:
                crf_total = np.stack(list(ex.map(_crf_job, jobs))).sum(0)
            crf_pq = smooth(pq_grid(crf_total[None])[0], int(sel["smooth"]))
            g = int(np.argmax(crf_pq))
            log(f"[select] DenseCRF on best variant: plateau PQ {crf_pq[g]:.4f} vs {pq_s[vi, ti, ai, gi]:.4f}")
            if crf_pq[g] > pq_s[vi, ti, ai, gi]:
                best.update({"crf": True, "score_thr": float(grid[g]), **pq_from(crf_total[:, g])})
    log(f"[select] chosen policy: {json.dumps(best)}")
    (P["work"] / "selected_policy.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    write_submission(cfg, P, best, variants[vi], ma_names, per_variant, t0)


def write_submission(cfg, P, best, variant, ma_names, per_variant, t0) -> None:
    files = image_files(P["test_images"])
    limit = int(cfg["data"].get("limit_test_images") or 0)
    allow_partial = bool(cfg["submission"].get("allow_partial"))
    if limit and not allow_partial:
        raise SystemExit("limit_test_images is set but allow_partial is false")
    stems = [p.stem for p in (files[:limit] if limit else files)]
    needed = available_sources(P, "test", stems)
    rows, fallbacks = [], []
    for stem in stems:
        S = load_sources(P, "test", stem, needed)
        try:
            cands = make_variant(variant, S, cfg, {})
        except KeyError as exc:
            raise SystemExit(f"Test cache {exc} required by the chosen variant is missing") from exc
        if best["crf"]:
            cands = crf_refine(strip_cache(cands), read_prep(P, "test", stem), cfg)
        cap = best["max_area"]
        lowest = min(cfg["select"]["mask_thresholds"])
        trim = bool(cfg.get("select", {}).get("topology_guarded_trim", False))
        masks = final_masks(cands, best["mask_thr"], best["min_area"], best["score_thr"], cap, trim=trim)
        if not masks:  # keep the single most confident plausible instance rather than leave the image empty
            fallbacks.append(stem)
            masks = final_masks(strip_cache(cands), lowest, 1, -np.inf, cap, trim=trim)
            for name in needed:
                if masks:
                    break
                masks = final_masks(strip_cache(S[name]), lowest, 1, -np.inf, cap, trim=trim)
            if not masks:
                masks = final_masks(strip_cache(cands), lowest, 1, -np.inf, None, trim=trim)[:1]
        if not masks:
            log(f"[select] zero filament detections for test image {stem}")
            continue
        rows += [(f"{stem}_{i}", rle_encode(m)) for i, m in enumerate(masks, 1)]

    frame = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    report = validate_submission(frame, stems, min_area=1)
    out = P["sub"] / cfg["submission"]["name"]
    tmp = out.with_suffix(".tmp.csv")
    frame.to_csv(tmp, index=False)
    check = validate_submission(pd.read_csv(tmp, dtype=str), stems, min_area=1)
    if not (report["ok"] and check["ok"]):
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"Submission failed validation: {report['problems'] or check['problems']}")
    tmp.replace(out)

    lines = [
        "CLEAN PIPELINE VALIDATION REPORT",
        "=" * 64,
        f"submission           : {out}",
        f"test images          : {report['images']} / {len(stems)}",
        f"instances            : {report['rows']} (per image min {report['per_image']['min']}, "
        f"median {report['per_image']['median']}, mean {report['per_image']['mean']:.2f}, max {report['per_image']['max']})",
        f"mean / median area   : {report['mean_area']:.0f} / {report['median_area']:.0f} px",
        f"overlapping pixels   : {report['overlap_pixels']}",
        f"undecodable RLEs     : {report['undecodable']}",
        f"fallback images      : {len(fallbacks)} {fallbacks[:10]}",
        "",
        "SELECTED POLICY (chosen on the held-out training fold only)",
        f"  variant            : {best['variant']}",
        f"  mask threshold     : {best['mask_thr']}",
        f"  min area           : {best['min_area']} px ({best['min_area_rule']})",
        f"  max area           : {best['max_area']} px (2 x largest training-fold filament)",
        f"  score threshold    : {best['score_thr']:.2f}",
        f"  DenseCRF           : {best['crf']}",
        f"  OOF PQ / SQ / RQ   : {best['PQ']:.4f} / {best['SQ']:.4f} / {best['RQ']:.4f}",
        f"  OOF TP / FP / FN   : {best['TP']} / {best['FP']} / {best['FN']}",
        f"  OOF PQ bootstrap   : 90% interval {best['bootstrap_PQ_p05']:.4f} - {best['bootstrap_PQ_p95']:.4f}",
        "",
        "BEST OOF RESULT PER VARIANT",
    ]
    for r in per_variant.itertuples():
        lines.append(f"  {r.variant:<55} PQ {r.PQ:.4f}  SQ {r.SQ:.4f}  TP {r.TP}  FP {r.FP}  FN {r.FN}")
    lines += [
        "",
        "NOTES",
        "  * OOF = one held-out, date-grouped training fold scored with fold-trained models.",
        "  * No leaderboard projection is printed: earlier 'local PQ -> LB' formulas were fitted on",
        "    test-label scores and do not apply to honest OOF PQ.",
        "  * This file was NOT submitted. Review it, then submit manually if you choose to.",
        f"  * total selection + submission time: {(time.time() - t0) / 60:.1f} min",
    ]
    text = "\n".join(lines)
    (P["sub"] / "validation_report.txt").write_text(text + "\n", encoding="utf-8")
    (P["sub"] / "validation_report.json").write_text(json.dumps({"policy": best, "integrity": report}, indent=2), encoding="utf-8")
    log(text)


if __name__ == "__main__":
    main()
