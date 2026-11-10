#!/usr/bin/env python
"""Fine-tune Mask2Former (Swin-L, COCO instance weights) for 1-class filament instances.

One training sample per annotator view; polygons are rasterised directly at the training
resolution. The best checkpoint is chosen by PQ on the held-out fold (training data).
Run under CUDA_VISIBLE_DEVICES with a single visible GPU.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from pycocotools import mask as mask_util  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from solar.common import (  # noqa: E402
    IMG, TrainAnnotations, load_config, log, paths, pq_from, read_prep, resolve, split_stems, sweep_image,
    train_stats,
)
from solar.models import m2f_candidates, m2f_load, norm3  # noqa: E402


class ViewDataset(Dataset):
    """Targets use the official pycocotools rasterisation (cv2.fillPoly is ~11% larger)."""

    def __init__(self, P, A, stems, size, rotate):
        self.P, self.size, self.rotate = P, int(size), float(rotate)
        self.items = []
        for stem in stems:
            for view in A.stem_views[stem]:
                rles = [A.rle(ann, view) for ann in A.anns.get(view, [])]
                if rles:
                    self.items.append((stem, rles))

    def __len__(self):
        return len(self.items)

    def _sample(self, i):
        stem, rles = self.items[i]
        S = self.size
        img = cv2.resize(read_prep(self.P, "train", stem), (S, S), interpolation=cv2.INTER_AREA)
        masks = np.stack([
            cv2.resize(np.ascontiguousarray(mask_util.decode(r), dtype=np.float32), (S, S), interpolation=cv2.INTER_AREA) > 0.5
            for r in rles
        ]).astype(np.uint8)
        if random.random() < 0.5:
            img, masks = img[:, ::-1], masks[:, :, ::-1]
        if random.random() < 0.5:
            img, masks = img[::-1], masks[:, ::-1]
        k = random.randint(0, 3)
        if k:
            img, masks = np.rot90(img, k), np.rot90(masks, k, axes=(1, 2))
        img, masks = np.ascontiguousarray(img), np.ascontiguousarray(masks)
        if self.rotate > 0:
            M = cv2.getRotationMatrix2D((S / 2.0, S / 2.0), random.uniform(-self.rotate, self.rotate), 1.0)
            img = cv2.warpAffine(img, M, (S, S), flags=cv2.INTER_LINEAR, borderValue=0)
            masks = np.stack([cv2.warpAffine(m, M, (S, S), flags=cv2.INTER_NEAREST, borderValue=0) for m in masks])
        gain = random.uniform(0.85, 1.15)
        bias = random.uniform(-12, 12)
        img = np.clip(img.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)
        masks = masks[masks.reshape(len(masks), -1).sum(1) >= 4]
        return img, masks

    def __getitem__(self, i):
        for _ in range(10):
            img, masks = self._sample(i)
            if len(masks):
                return torch.from_numpy(norm3(img)), torch.from_numpy(masks.astype(np.float32))
            i = random.randrange(len(self.items))
        raise RuntimeError("Could not draw a sample with at least one instance")


def collate(batch):
    return torch.stack([b[0] for b in batch]), [b[1] for b in batch]


def evaluate(model, P, A, stems, cfg, device) -> dict:
    model.eval()
    grid = np.round(np.arange(0.05, 0.951, 0.05), 2)
    total = np.zeros((4, len(grid)))
    max_area = int(train_stats(P)["max_area"])
    for stem in stems:
        cands = m2f_candidates(model, read_prep(P, "train", stem), cfg, device, flips=["none"])
        total += sweep_image(cands, A.gt_flat_views(stem), 0.5, 0, grid, max_area)
    pq = [pq_from(total[:, t])["PQ"] for t in range(len(grid))]
    t = int(np.argmax(pq))
    model.train()
    return {"tau": float(grid[t]), **pq_from(total[:, t])}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--mode", choices=["fold", "full"], default="fold")
    args = ap.parse_args()
    cfg = load_config(args.config)
    P = paths(cfg)
    m = cfg["mask2former"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    A = TrainAnnotations(P["ann"])
    train_stems, val_stems = split_stems(cfg, P, args.mode)
    ds = ViewDataset(P, A, train_stems, m["image_size"], m["rotate_degrees"])
    dl = DataLoader(ds, batch_size=int(m["batch_size"]), shuffle=True, num_workers=int(m["num_workers"]),
                    collate_fn=collate, drop_last=True, pin_memory=True, persistent_workers=int(m["num_workers"]) > 0)
    model = m2f_load(resolve(m["pretrained"]), device)
    model.train()

    backbone = [p for n, p in model.named_parameters() if n.startswith("model.pixel_level_module.encoder.")]
    rest = [p for n, p in model.named_parameters() if not n.startswith("model.pixel_level_module.encoder.")]
    opt = torch.optim.AdamW([
        {"params": backbone, "lr": float(m["lr_backbone"])},
        {"params": rest, "lr": float(m["lr_decoder"])},
    ], weight_decay=float(m["weight_decay"]))
    accum = max(1, int(m["grad_accum"]))
    steps_per_epoch = max(1, math.ceil(len(dl) / accum))
    total_steps = steps_per_epoch * int(m["epochs"])
    if int(m.get("max_steps") or 0) > 0:
        total_steps = min(total_steps, int(m["max_steps"]))
    warmup = min(int(m["warmup_steps"]), max(1, total_steps // 10))
    base_lrs = [g["lr"] for g in opt.param_groups]
    budget = float(m.get("time_hours") or 0) * 3600.0

    def set_lr(step: int, elapsed: float) -> None:
        """Linear warmup, then cosine decay over whichever ends first: steps or the time budget."""
        if step < warmup:
            factor = (step + 1) / warmup
        else:
            progress = (step - warmup) / max(1, total_steps - warmup)
            if budget:
                progress = max(progress, elapsed / budget)
            factor = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        for group, base in zip(opt.param_groups, base_lrs):
            group["lr"] = base * factor
    out_dir = P["models"] / f"mask2former_{args.mode}"
    hist_path = P["logs"] / f"mask2former_{args.mode}_history.json"
    log(f"[m2f:{args.mode}] {len(ds)} view samples | {len(train_stems)} train stems | {len(val_stems)} val stems | "
        f"{total_steps} optimizer steps | size {m['image_size']} | batch {m['batch_size']}x{accum}")

    best, history, step, t0 = -1.0, [], 0, time.time()
    epochs = int(m["epochs"])
    out_of_time = False
    for epoch in range(1, epochs + 1):
        running, n = 0.0, 0
        opt.zero_grad(set_to_none=True)
        for it, (px, masks) in enumerate(dl, 1):
            if budget and time.time() - t0 > budget:
                out_of_time = True
                log(f"[m2f:{args.mode}] time budget {m['time_hours']} h reached at step {step}")
                break
            px = px.to(device, non_blocking=True)
            mask_labels = [mk.to(device, non_blocking=True) for mk in masks]
            class_labels = [torch.zeros(len(mk), dtype=torch.long, device=device) for mk in masks]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                loss = model(pixel_values=px, mask_labels=mask_labels, class_labels=class_labels).loss / accum
            loss.backward()
            running += float(loss.detach()) * accum
            n += 1
            if it % accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(m["grad_clip"]))
                set_lr(step, time.time() - t0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 50 == 0:
                    log(f"[m2f:{args.mode}] epoch {epoch} step {step}/{total_steps} loss {running / n:.3f} "
                        f"{(time.time() - t0) / 60:.1f} min")
                if step >= total_steps:
                    break
        rem = n % accum
        if rem != 0 and step < total_steps and not out_of_time:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.data.mul_(accum / rem)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(m["grad_clip"]))
            set_lr(step, time.time() - t0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
        last = epoch == epochs or step >= total_steps or out_of_time
        do_eval = val_stems and (epoch % max(1, int(m["eval_every"])) == 0 or last)
        record = {"epoch": epoch, "step": step, "train_loss": running / max(n, 1), "minutes": (time.time() - t0) / 60}
        if do_eval:
            metrics = evaluate(model, P, A, val_stems, cfg, device)
            record.update(metrics)
            if metrics["PQ"] > best:
                best = metrics["PQ"]
                model.save_pretrained(out_dir)
                (out_dir / "astra_val.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
            log(f"[m2f:{args.mode}] epoch {epoch} val PQ {metrics['PQ']:.4f} SQ {metrics['SQ']:.4f} "
                f"TP {metrics['TP']} FP {metrics['FP']} FN {metrics['FN']} (best {best:.4f})")
        history.append(record)
        hist_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        if step >= total_steps or out_of_time:
            break
    if not (out_dir / "config.json").exists():
        model.save_pretrained(out_dir)
    log(f"[m2f:{args.mode}] done; best held-out PQ {best:.4f}; saved {out_dir}")


if __name__ == "__main__":
    main()
