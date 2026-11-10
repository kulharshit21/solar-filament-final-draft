"""Shared utilities: config, clean data access, folds, candidates, fusion, PQ, submission checks.

Every non-image input goes through ``guard()``, which refuses files that could carry
test-set labels (full public MAGFiLO COCO, test cluster databases, old submissions).
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml
from pycocotools import mask as mask_util

ROOT = Path(__file__).resolve().parents[2]
IMG = 2048
FORBIDDEN_TOKENS = (
    "magfilo_full_coco",
    "test_image_clusters",
    "test_gt",
    "base_submissions",
    "v70_baseline",
    "v73_baseline",
)


def log(msg: str) -> None:
    print(msg, flush=True)


def guard(path) -> Path:
    """Refuse any input whose name suggests it carries test-set labels."""
    p = Path(path).resolve()
    name = str(p).replace("\\", "/").lower()
    if any(token in name for token in FORBIDDEN_TOKENS):
        raise RuntimeError(f"Refusing to read forbidden input: {p}")
    return p


def deep_merge(base: dict, override: dict) -> dict:
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def load_config(path=None) -> dict:
    """config.yaml, optionally overlaid by another YAML (e.g. config.smoke.yaml)."""
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    if path is not None:
        p = Path(path)
        p = p if p.is_absolute() else ROOT / p
        if p.resolve() != (ROOT / "config.yaml").resolve():
            deep_merge(cfg, yaml.safe_load(p.read_text(encoding="utf-8")) or {})
    return cfg


def paths(cfg: dict) -> dict:
    work = ROOT / cfg["paths"]["work"]
    P = {
        "root": ROOT,
        "train_images": ROOT / "data" / "train_images",
        "test_images": ROOT / "data" / "test_images",
        "ann": ROOT / "data" / "annotations_train.json",
        "work": work,
        "prep": work / "prep",
        "yolo": work / "yolo",
        "models": work / "models",
        "preds": work / "preds",
        "logs": work / "logs",
        "sub": ROOT / cfg["paths"]["submissions"],
    }
    for key in ("work", "prep", "yolo", "models", "preds", "logs", "sub"):
        P[key].mkdir(parents=True, exist_ok=True)
    return P


def resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else ROOT / p


def image_files(folder: Path) -> list[Path]:
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in {".jpeg", ".jpg", ".png"})


# ----------------------------------------------------------------------------- images


def read_gray(path) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Could not read image: {path}")
    if img.ndim == 3:
        img = img[..., 0]
    return img


def preprocess(gray: np.ndarray, cfg: dict) -> np.ndarray:
    mode = cfg["data"]["preprocess"]
    if mode == "raw":
        return gray
    if mode == "clahe":
        tile = int(cfg["data"]["clahe_tile"])
        clahe = cv2.createCLAHE(clipLimit=float(cfg["data"]["clahe_clip"]), tileGridSize=(tile, tile))
        return clahe.apply(gray)
    raise ValueError(f"Unknown preprocess mode: {mode}")


def prep_path(P: dict, split: str, stem: str) -> Path:
    return P["prep"] / split / f"{stem}.png"


def read_prep(P: dict, split: str, stem: str) -> np.ndarray:
    img = read_gray(prep_path(P, split, stem))
    if img.shape != (IMG, IMG):
        raise RuntimeError(f"Unexpected preprocessed shape {img.shape} for {stem}")
    return img


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.symlink(src.resolve(), dst)
    except (OSError, NotImplementedError):
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)


# ------------------------------------------------------------------ training annotations


class TrainAnnotations:
    """Official Kaggle *training* COCO file: one COCO image per annotator view."""

    def __init__(self, path):
        data = json.loads(guard(path).read_text(encoding="utf-8"))
        self.views: dict[str, dict] = {}
        self.stem_views: dict[str, list[str]] = defaultdict(list)
        self.anns: dict[str, list[dict]] = defaultdict(list)
        for image in data["images"]:
            view = str(image["id"])
            stem = Path(image["file_name"]).stem
            self.views[view] = {
                "stem": stem,
                "file_name": image["file_name"],
                "annotator": view.split("-", 1)[0],
                "height": int(image["height"]),
                "width": int(image["width"]),
            }
            self.stem_views[stem].append(view)
        for ann in data["annotations"]:
            self.anns[str(ann["image_id"])].append(ann)
        self.n_annotations = len(data["annotations"])
        if len(self.stem_views) != 707 or len(self.views) != 1154 or self.n_annotations != 8199:
            raise ValueError(
                f"Unexpected training annotation counts: {len(self.stem_views)} stems, "
                f"{len(self.views)} views, {self.n_annotations} annotations. "
                "Expected exactly 707 stems, 1154 views, 8199 annotations."
            )

    @property
    def stems(self) -> list[str]:
        return sorted(self.stem_views)

    def polygons(self, ann: dict) -> list[np.ndarray]:
        seg = ann["segmentation"]
        if not isinstance(seg, list):
            raise ValueError("Expected polygon segmentation in the training JSON")
        return [np.asarray(poly, np.float32).reshape(-1, 2) for poly in seg if len(poly) >= 6]

    def rle(self, ann: dict, view: str) -> dict:
        info = self.views[view]
        rles = mask_util.frPyObjects(ann["segmentation"], info["height"], info["width"])
        return mask_util.merge(rles) if isinstance(rles, list) else rles

    def gt_flat_views(self, stem: str) -> list[list[np.ndarray]]:
        """Per annotator view, the flat (row-major) pixel indices of every GT instance."""
        out = []
        for view in self.stem_views[stem]:
            items = []
            for ann in self.anns.get(view, []):
                m = mask_util.decode(self.rle(ann, view))
                items.append(np.flatnonzero(m).astype(np.int64))
            out.append(items)
        return out


def make_folds(stems, n_splits: int, group_chars: int) -> dict[str, int]:
    """GroupKFold over observation dates (first ``group_chars`` characters of the stem)."""
    from sklearn.model_selection import GroupKFold

    stems = sorted(stems)
    groups = [s[:group_chars] for s in stems]
    folds = {}
    for k, (_, val_idx) in enumerate(GroupKFold(n_splits=n_splits).split(stems, groups=groups)):
        for i in val_idx:
            folds[stems[i]] = k
    return folds


def load_folds(P: dict) -> dict[str, int]:
    import pandas as pd

    frame = pd.read_csv(guard(P["work"] / "folds.csv"))
    return dict(zip(frame["stem"], frame["fold"].astype(int)))


def split_stems(cfg: dict, P: dict, mode: str) -> tuple[list[str], list[str]]:
    """(train stems, validation stems) for mode 'fold' (hold out fold_index) or 'full'."""
    folds = load_folds(P)
    k = int(cfg["data"]["fold_index"])
    val = sorted(s for s, f in folds.items() if f == k)
    train = sorted(folds) if mode == "full" else sorted(s for s, f in folds.items() if f != k)
    return train, val


def train_stats(P: dict) -> dict:
    return json.loads(guard(P["work"] / "train_stats.json").read_text(encoding="utf-8"))


# ------------------------------------------------------------------------- candidates
# A candidate is a dict: win=(y1, y2, x1, x2) native window, prob=float16 map over the window,
# score=float, src=str, idx=int (position in its source list), box=(x1, y1, x2, y2).


def cand_from_prob(prob_full: np.ndarray, score: float, src: str, idx: int, box=None, thr: float = 0.05, pad: int = 2):
    """Crop a full-canvas probability (or binary) map to its support; None if empty."""
    support = prob_full > thr
    rows = np.flatnonzero(support.any(1))
    if rows.size == 0:
        return None
    cols = np.flatnonzero(support.any(0))
    y1, y2 = max(0, rows[0] - pad), min(IMG, rows[-1] + 1 + pad)
    x1, x2 = max(0, cols[0] - pad), min(IMG, cols[-1] + 1 + pad)
    if box is None:
        box = (float(x1), float(y1), float(x2), float(y2))
    return {
        "win": (int(y1), int(y2), int(x1), int(x2)),
        "prob": np.ascontiguousarray(prob_full[y1:y2, x1:x2]).astype(np.float16),
        "score": float(score),
        "src": src,
        "idx": int(idx),
        "box": tuple(float(v) for v in box),
    }


def cand_from_local(prob_local: np.ndarray, win, score: float, src: str, idx: int, box, thr: float = 0.05, pad: int = 2):
    """Same as cand_from_prob for a map that already lives on window ``win``."""
    y1, y2, x1, x2 = win
    support = prob_local > thr
    rows = np.flatnonzero(support.any(1))
    if rows.size == 0:
        return None
    cols = np.flatnonzero(support.any(0))
    r0, r1 = max(0, rows[0] - pad), min(prob_local.shape[0], rows[-1] + 1 + pad)
    c0, c1 = max(0, cols[0] - pad), min(prob_local.shape[1], cols[-1] + 1 + pad)
    return {
        "win": (int(y1 + r0), int(y1 + r1), int(x1 + c0), int(x1 + c1)),
        "prob": np.ascontiguousarray(prob_local[r0:r1, c0:c1]).astype(np.float16),
        "score": float(score),
        "src": src,
        "idx": int(idx),
        "box": tuple(float(v) for v in box),
    }


def _overlap(a, b):
    y1, y2 = max(a[0], b[0]), min(a[1], b[1])
    x1, x2 = max(a[2], b[2]), min(a[3], b[3])
    return (y1, y2, x1, x2) if y2 > y1 and x2 > x1 else None


def _binary(c: dict, thr: float) -> tuple[np.ndarray, int]:
    key = ("_bin", thr)
    if key not in c:
        b = c["prob"] > thr
        c[key] = (b, int(np.count_nonzero(b)))
    return c[key]


def cand_iou(a: dict, b: dict, thr: float = 0.5) -> float:
    ov = _overlap(a["win"], b["win"])
    if ov is None:
        return 0.0
    ba, na = _binary(a, thr)
    bb, nb = _binary(b, thr)
    if na == 0 or nb == 0:
        return 0.0
    y1, y2, x1, x2 = ov
    sa = ba[y1 - a["win"][0] : y2 - a["win"][0], x1 - a["win"][2] : x2 - a["win"][2]]
    sb = bb[y1 - b["win"][0] : y2 - b["win"][0], x1 - b["win"][2] : x2 - b["win"][2]]
    inter = int(np.count_nonzero(sa & sb))
    return inter / float(na + nb - inter)


def strip_cache(cands: list[dict]) -> list[dict]:
    return [{k: v for k, v in c.items() if not isinstance(k, tuple)} for c in cands]


def fuse_sources(lists, weights, iou_thr: float = 0.5, bin_thr: float = 0.5) -> list[dict]:
    """Cluster candidates across sources (one member per source) and average them.

    Fused probability = sum_s w_s * prob_s / sum_{s in members} w_s over participating sources,
    so singletons preserve intact geometry without zero-dilution.
    Fused score = sum_{s in members} w_s * score_s / sum_s w_s over all sources, so candidates
    confirmed by multiple sources receive higher rank.
    """
    total = float(sum(weights))
    items = [(c["score"], s, c) for s, lst in enumerate(lists) for c in lst]
    items.sort(key=lambda t: -t[0])
    clusters: list[dict] = []
    for _, s, c in items:
        best, best_iou = None, iou_thr
        for cluster in clusters:
            if s in cluster["members"]:
                continue
            iou = cand_iou(cluster["rep"], c, bin_thr)
            if iou > best_iou:
                best, best_iou = cluster, iou
        if best is None:
            clusters.append({"rep": c, "members": {s: c}})
        else:
            best["members"][s] = c
    out = []
    for cluster in clusters:
        members = cluster["members"]
        y1 = min(m["win"][0] for m in members.values())
        y2 = max(m["win"][1] for m in members.values())
        x1 = min(m["win"][2] for m in members.values())
        x2 = max(m["win"][3] for m in members.values())
        prob = np.zeros((y2 - y1, x2 - x1), np.float32)
        for s, m in members.items():
            a, b, c0, d = m["win"]
            prob[a - y1 : b - y1, c0 - x1 : d - x1] += weights[s] * m["prob"].astype(np.float32)
        member_weight = float(sum(weights[s] for s in members.keys()))
        prob /= member_weight
        boxes = np.asarray([m["box"] for m in members.values()], np.float32)
        out.append(
            {
                "win": (y1, y2, x1, x2),
                "prob": prob.astype(np.float16),
                "score": float(sum(weights[s] * m["score"] for s, m in members.items()) / total),
                "src": "+".join(sorted({m["src"] for m in members.values()})),
                "idx": len(out),
                "box": (float(boxes[:, 0].min()), float(boxes[:, 1].min()), float(boxes[:, 2].max()), float(boxes[:, 3].max())),
                "n_src": len(members),
                "members": sorted((int(s), int(m["idx"])) for s, m in members.items()),
            }
        )
    out.sort(key=lambda c: -c["score"])
    for i, c in enumerate(out):
        c["idx"] = i
    return out


def average_pairs(native: list[dict], refined: list[dict]) -> list[dict]:
    """Pixel-average each native candidate with its refined counterpart (same ``idx``)."""
    by_idx = {c["idx"]: c for c in refined}
    out = []
    for c in native:
        r = by_idx.get(c["idx"])
        if r is None:
            out.append(c)
            continue
        y1, y2 = min(c["win"][0], r["win"][0]), max(c["win"][1], r["win"][1])
        x1, x2 = min(c["win"][2], r["win"][2]), max(c["win"][3], r["win"][3])
        prob = np.zeros((y2 - y1, x2 - x1), np.float32)
        for m in (c, r):
            a, b, c0, d = m["win"]
            prob[a - y1 : b - y1, c0 - x1 : d - x1] += 0.5 * m["prob"].astype(np.float32)
        out.append({**{k: v for k, v in c.items() if not isinstance(k, tuple)}, "win": (y1, y2, x1, x2), "prob": prob.astype(np.float16)})
    return out


# ------------------------------------------------------------- painting and PQ sweeps


def topology_guarded_trim(mask: np.ndarray, radius_guard: float = 2.0) -> np.ndarray:
    """Trim over-dilated penumbras on thick structures while strictly protecting thin spines/barbs and connectivity."""
    if not np.any(mask):
        return mask
    mask_u8 = mask.astype(np.uint8)
    n_labels, labels, _, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    dist = cv2.distanceTransform(mask_u8, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    out = np.zeros_like(mask)
    cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    for lbl in range(1, n_labels):
        comp = (labels == lbl)
        max_d = float((dist * comp).max())
        if max_d <= radius_guard:
            # Thin structure (width <= 4px): 100% preserved
            out |= comp
            continue
        eroded = cv2.erode(comp.astype(np.uint8), cross, iterations=1) > 0
        if not np.any(eroded):
            out |= comp
            continue
        n_eroded, _ = cv2.connectedComponents(eroded.astype(np.uint8), connectivity=8)
        if n_eroded == 2:
            # Single component preserved: apply 1px boundary trim on thick structure
            out |= eroded
        else:
            # Erosion would sever a bridge or neck: preserve original component
            out |= comp
    return out


def paint(cands: list[dict], mask_thr: float, min_area: int, max_area: int | None = None, trim: bool = False):
    """Greedy zero-overlap painting in descending score.

    Returns (label map int32 with 0 = background and r + 1 = r-th ranked candidate,
    ranked candidate indices, eligible flags, retained areas, ranked scores). A candidate is
    eligible when it keeps between max(min_area, 1) and max_area pixels after higher-ranked
    candidates claim theirs; ineligible candidates claim nothing.
    """
    order = sorted(range(len(cands)), key=lambda i: -cands[i]["score"])
    lab = np.zeros((IMG, IMG), np.int32)
    n = len(order)
    elig = np.zeros(n, bool)
    area = np.zeros(n, np.int64)
    floor = max(int(min_area), 1)
    ceiling = int(max_area) if max_area else IMG * IMG
    for r, i in enumerate(order):
        c = cands[i]
        y1, y2, x1, x2 = c["win"]
        sub = lab[y1:y2, x1:x2]
        new = (c["prob"] > mask_thr) & (sub == 0)
        if trim and np.any(new):
            new = topology_guarded_trim(new) & (sub == 0)
        a = int(np.count_nonzero(new))
        if floor <= a <= ceiling:
            sub[new] = r + 1
            elig[r] = True
            area[r] = a
    scores = np.asarray([cands[i]["score"] for i in order], np.float64)
    return lab, order, elig, area, scores


def effective_scores(elig: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """The best eligible candidate is always kept (every image gets >= 1 instance)."""
    eff = scores.copy()
    first = np.flatnonzero(elig)
    if first.size:
        eff[first[0]] = np.inf
    return eff


def count_ge(values, grid: np.ndarray, weights=None) -> np.ndarray:
    """out[t] = (weighted) number of values >= grid[t]; grid ascending."""
    v = np.asarray(values, np.float64)
    if v.size == 0:
        return np.zeros(len(grid))
    pos = np.searchsorted(grid, v, side="right")
    h = np.bincount(pos, weights=weights, minlength=len(grid) + 1)
    return np.cumsum(h[::-1])[::-1][1 : len(grid) + 1]


def sweep_image(cands, gt_views, mask_thr: float, min_area: int, grid: np.ndarray, max_area: int | None = None, trim: bool = False) -> np.ndarray:
    """Exact competition PQ statistics (TP, FP, FN, sum IoU) for every score threshold in grid.

    Matching follows the official evaluator: per annotator view, a (GT, prediction) pair is a hit
    when IoU > 0.5; predictions without a hit are FP, GT without a hit are FN. Predictions are
    disjoint, so only the prediction with the largest overlap can hit a GT.
    """
    lab, _, elig, area, scores = paint(cands, mask_thr, min_area, max_area, trim=trim)
    eff = effective_scores(elig, scores)
    flat = lab.ravel()
    n = len(scores)
    out = np.zeros((4, len(grid)))
    for gts in gt_views:
        nh = np.zeros(n, np.int64)
        hit_s, hit_iou = [], []
        for idx in gts:
            if idx.size == 0:
                continue
            cnt = np.bincount(flat[idx], minlength=n + 1)
            cnt[0] = 0
            j = int(cnt.argmax())
            inter = int(cnt[j])
            if inter:
                iou = inter / float(idx.size + area[j - 1] - inter)
                if iou > 0.5:
                    nh[j - 1] += 1
                    hit_s.append(eff[j - 1])
                    hit_iou.append(iou)
        tp = count_ge(hit_s, grid)
        out[0] += tp
        out[1] += count_ge(eff[elig & (nh == 0)], grid)
        out[2] += len(gts) - tp
        out[3] += count_ge(hit_s, grid, weights=np.asarray(hit_iou, np.float64))
    return out


def pq_from(stats) -> dict:
    tp, fp, fn, s = (float(x) for x in stats)
    den = tp + 0.5 * fp + 0.5 * fn
    return {
        "PQ": s / den if den else 0.0,
        "SQ": s / tp if tp else 0.0,
        "RQ": tp / den if den else 0.0,
        "TP": int(round(tp)),
        "FP": int(round(fp)),
        "FN": int(round(fn)),
    }


def final_masks(cands, mask_thr: float, min_area: int, tau: float, max_area: int | None = None, trim: bool = False) -> list[np.ndarray]:
    """Disjoint final masks in rank order; identical selection rule to ``sweep_image``."""
    lab, _, elig, _, scores = paint(cands, mask_thr, min_area, max_area, trim=trim)
    eff = effective_scores(elig, scores)
    keep = np.flatnonzero(elig & (eff >= tau))
    return [lab == r + 1 for r in keep]


# ------------------------------------------------------------------------- submission


def rle_encode(mask: np.ndarray) -> str:
    counts = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))["counts"]
    return counts.decode("ascii") if isinstance(counts, bytes) else str(counts)


def validate_submission(frame, expected_stems, min_area: int) -> dict:
    """Columns, ids, nulls, exact stem coverage, decodable 2048x2048 RLEs, area floor, zero overlap."""
    import pandas as pd  # noqa: F401  (frame is a DataFrame)

    problems = []
    if list(frame.columns) != ["filament_id", "segmentation_rle"]:
        problems.append(f"columns {list(frame.columns)}")
    if frame.isna().to_numpy().any():
        problems.append("null values")
    if frame["filament_id"].duplicated().any():
        problems.append("duplicate filament_id")
    parts = frame["filament_id"].astype(str).str.rsplit("_", n=1)
    stems = parts.str[0]
    if not parts.str[1].fillna("").str.fullmatch(r"[1-9]\d*").all():
        problems.append("instance suffix is not a positive integer")
    unexpected = set(stems) - set(expected_stems)
    if unexpected:
        problems.append(f"stem set mismatch: unexpected stems {unexpected}")
    overlap_px, bad_rle, small, areas = 0, 0, 0, []
    for _, group in frame.groupby(stems, sort=False):
        occupied = np.zeros((IMG, IMG), bool)
        for counts in group["segmentation_rle"].astype(str):
            try:
                m = mask_util.decode({"size": [IMG, IMG], "counts": counts.encode("ascii")}) > 0
            except Exception:
                bad_rle += 1
                continue
            a = int(m.sum())
            areas.append(a)
            small += a < max(int(min_area), 1)
            overlap_px += int(np.count_nonzero(m & occupied))
            occupied |= m
    if bad_rle:
        problems.append(f"{bad_rle} undecodable RLEs")
    if small:
        problems.append(f"{small} masks below {min_area} px")
    if overlap_px:
        problems.append(f"{overlap_px} overlapping pixels")
    per_image = stems.value_counts()
    return {
        "ok": not problems,
        "problems": problems,
        "rows": int(len(frame)),
        "images": int(stems.nunique()),
        "overlap_pixels": overlap_px,
        "undecodable": bad_rle,
        "mean_area": float(np.mean(areas)) if areas else 0.0,
        "median_area": float(np.median(areas)) if areas else 0.0,
        "per_image": {
            "min": int(per_image.min()) if len(per_image) else 0,
            "median": float(per_image.median()) if len(per_image) else 0.0,
            "max": int(per_image.max()) if len(per_image) else 0,
            "mean": float(per_image.mean()) if len(per_image) else 0.0,
        },
    }
