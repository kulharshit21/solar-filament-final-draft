#!/usr/bin/env python
"""Fine-tune the SAM 2.1 (Hiera-L) mask decoder as a box-prompted filament refiner.

The image encoder and prompt encoder stay frozen. Each training annotation becomes a square
context crop resized to 1024, prompted with its jittered box; loss = 20*focal + dice + IoU-head MSE.
Validation = mean IoU on held-out-fold annotations. Run under CUDA_VISIBLE_DEVICES (one GPU).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from pycocotools import mask as mask_util  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from solar.common import TrainAnnotations, load_config, log, paths, read_prep, resolve, split_stems  # noqa: E402
from solar.models import SAM_SIZE, extract_crop, norm3, sam2_forward, sam2_load, square_crop, to_crop_box  # noqa: E402


class CropDataset(Dataset):
    def __init__(self, P, A, stems, cfg, train: bool, limit: int = 0):
        s = cfg["sam2"]
        self.P, self.train = P, train
        self.ctx = tuple(float(v) for v in s["context_train"])
        self.ctx_val = float(s["context_infer"])
        self.min_crop = int(s["min_crop"])
        self.jitter = float(s["box_jitter"])
        self.items = []
        for stem in stems:
            for view in A.stem_views[stem]:
                for ann in A.anns.get(view, []):
                    rle = A.rle(ann, view)
                    x, y, w, h = (float(v) for v in mask_util.toBbox(rle))
                    if w < 1 or h < 1:
                        continue
                    self.items.append((stem, rle, (x, y, x + w, y + h)))
        if limit and len(self.items) > limit:
            self.items = random.Random(0).sample(self.items, limit)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        stem, rle, box = self.items[i]
        gray = read_prep(self.P, "train", stem)
        if self.train:
            shift = (random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1))
            x0, y0, side = square_crop(box, random.uniform(*self.ctx), self.min_crop, shift)
        else:
            x0, y0, side = square_crop(box, self.ctx_val, self.min_crop)
        crop = extract_crop(gray, x0, y0, side)
        # official pycocotools rasterisation, resampled exactly like the image crop
        full = np.ascontiguousarray(mask_util.decode(rle)) * np.uint8(255)
        mask = (extract_crop(full, x0, y0, side) > 127).astype(np.uint8)
        b = np.asarray(to_crop_box(box, x0, y0, side), np.float32)
        if self.train:
            w, h = b[2] - b[0], b[3] - b[1]
            b += np.asarray([w, h, w, h], np.float32) * np.random.normal(0, self.jitter, 4).astype(np.float32)
            if random.random() < 0.5:
                crop, mask = crop[:, ::-1], mask[:, ::-1]
                b = np.asarray([SAM_SIZE - b[2], b[1], SAM_SIZE - b[0], b[3]], np.float32)
            if random.random() < 0.5:
                crop, mask = crop[::-1], mask[::-1]
                b = np.asarray([b[0], SAM_SIZE - b[3], b[2], SAM_SIZE - b[1]], np.float32)
        b = np.clip(b, 0, SAM_SIZE)
        b = np.asarray([min(b[0], b[2] - 1), min(b[1], b[3] - 1), max(b[2], b[0] + 1), max(b[3], b[1] + 1)], np.float32)
        return (torch.from_numpy(norm3(np.ascontiguousarray(crop))), torch.from_numpy(b),
                torch.from_numpy(np.ascontiguousarray(mask).astype(np.float32)))


def losses(logits_lr, iou_pred, target):
    up = F.interpolate(logits_lr[:, None], size=target.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    prob = up.sigmoid()
    ce = F.binary_cross_entropy_with_logits(up, target, reduction="none")
    p_t = prob * target + (1 - prob) * (1 - target)
    focal = (ce * ((1 - p_t) ** 2) * (0.25 * target + 0.75 * (1 - target))).mean()
    inter = (prob * target).flatten(1).sum(1)
    dice = 1 - (2 * inter + 1) / (prob.flatten(1).sum(1) + target.flatten(1).sum(1) + 1)
    with torch.no_grad():
        hard = (up > 0).float()
        i = (hard * target).flatten(1).sum(1)
        u = ((hard + target) > 0).float().flatten(1).sum(1).clamp(min=1)
        iou_true = i / u
    return 20 * focal + dice.mean() + F.mse_loss(iou_pred, iou_true), iou_true


@torch.no_grad()
def evaluate(model, dl, device) -> float:
    model.eval()
    scores = []
    for px, bx, gt in dl:
        logits, iou = sam2_forward(model, px.to(device), bx.to(device), device)
        _, iou_true = losses(logits, iou, gt.to(device))
        scores.append(iou_true.cpu())
    model.train()
    return float(torch.cat(scores).mean()) if scores else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--mode", choices=["fold", "full"], default="fold")
    args = ap.parse_args()
    cfg = load_config(args.config)
    P = paths(cfg)
    s = cfg["sam2"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    A = TrainAnnotations(P["ann"])
    train_stems, val_stems = split_stems(cfg, P, args.mode)
    ds = CropDataset(P, A, train_stems, cfg, train=True)
    vs = CropDataset(P, A, val_stems, cfg, train=False, limit=int(s["val_max_annotations"]))
    nw = int(s["num_workers"])
    dl = DataLoader(ds, batch_size=int(s["batch_size"]), shuffle=True, num_workers=nw, drop_last=True,
                    pin_memory=True, persistent_workers=nw > 0)
    vl = DataLoader(vs, batch_size=int(s["batch_size"]), shuffle=False, num_workers=nw, pin_memory=True)
    model = sam2_load(resolve(s["pretrained"]), device)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=float(s["lr"]), weight_decay=float(s["weight_decay"]))
    total = len(dl) * int(s["epochs"])
    if int(s.get("max_steps") or 0) > 0:
        total = min(total, int(s["max_steps"]))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=float(s["lr"]), total_steps=max(total, 1), pct_start=0.05)
    out = P["models"] / f"sam2_{args.mode}_decoder.pt"
    log(f"[sam2:{args.mode}] {len(ds)} training crops | {len(vs)} validation crops | "
        f"{sum(p.numel() for p in params) / 1e6:.2f}M trainable params | {total} steps")

    base = evaluate(model, vl, device) if len(vs) else 0.0
    log(f"[sam2:{args.mode}] zero-shot validation mean IoU {base:.4f}")
    best, step, t0, history = base, 0, time.time(), [{"epoch": 0, "val_iou": base}]
    torch.save(model.mask_decoder.state_dict(), out)
    budget = float(s.get("time_hours") or 0) * 3600.0
    out_of_time = False
    for epoch in range(1, int(s["epochs"]) + 1):
        run, n = 0.0, 0
        for px, bx, gt in dl:
            if budget and time.time() - t0 > budget:
                out_of_time = True
                log(f"[sam2:{args.mode}] time budget {s['time_hours']} h reached at step {step}")
                break
            logits, iou = sam2_forward(model, px.to(device, non_blocking=True), bx.to(device, non_blocking=True), device)
            loss, _ = losses(logits, iou, gt.to(device, non_blocking=True))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            run, n, step = run + float(loss.detach()), n + 1, step + 1
            if step % 100 == 0:
                log(f"[sam2:{args.mode}] epoch {epoch} step {step}/{total} loss {run / n:.4f} {(time.time() - t0) / 60:.1f} min")
            if step >= total:
                break
        val = evaluate(model, vl, device) if len(vs) else 0.0
        history.append({"epoch": epoch, "step": step, "train_loss": run / max(n, 1), "val_iou": val})
        if val > best:
            best = val
            torch.save(model.mask_decoder.state_dict(), out)
        log(f"[sam2:{args.mode}] epoch {epoch} val mean IoU {val:.4f} (best {best:.4f})")
        (P["logs"] / f"sam2_{args.mode}_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if step >= total or out_of_time:
            break
    log(f"[sam2:{args.mode}] done; best validation mean IoU {best:.4f}; saved {out}")


if __name__ == "__main__":
    main()
