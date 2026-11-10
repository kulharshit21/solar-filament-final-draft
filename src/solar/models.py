"""Model wrappers (torch): YOLO-seg, Mask2Former and SAM2 inference/training helpers."""

from __future__ import annotations

import math

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .common import IMG, fuse_sources, guard

MEAN = np.asarray([0.485, 0.456, 0.406], np.float32)
STD = np.asarray([0.229, 0.224, 0.225], np.float32)


# -------------------------------------------------------------------------- flips


def flip_np(img: np.ndarray, mode: str) -> np.ndarray:
    if mode == "h":
        return img[..., :, ::-1]
    if mode == "v":
        return img[..., ::-1, :]
    if mode == "hv":
        return img[..., ::-1, ::-1]
    return img


def flip_t(t: torch.Tensor, mode: str) -> torch.Tensor:
    dims = {"h": (-1,), "v": (-2,), "hv": (-2, -1)}.get(mode)
    return torch.flip(t, dims=dims) if dims else t


def unflip_box(box, mode: str, size: int = IMG):
    x1, y1, x2, y2 = (float(v) for v in box)
    if mode in ("h", "hv"):
        x1, x2 = size - x2, size - x1
    if mode in ("v", "hv"):
        y1, y2 = size - y2, size - y1
    return (x1, y1, x2, y2)


def norm3(gray_u8: np.ndarray) -> np.ndarray:
    """uint8 (H, W) -> float32 (3, H, W) ImageNet-normalised."""
    x = gray_u8.astype(np.float32)[..., None] / 255.0
    return ((np.repeat(x, 3, axis=2) - MEAN) / STD).transpose(2, 0, 1).copy()


# ----------------------------------------------------------------- GPU -> candidates


def gpu_candidates(probs: torch.Tensor, scores, src: str, boxes=None, thr: float = 0.05, pad: int = 2) -> list[dict]:
    """probs: (N, 2048, 2048) float/bool on GPU. Crops every map to its support before transfer."""
    out = []
    if probs.numel() == 0:
        return out
    support = probs > thr
    rows_any = support.any(dim=2)
    cols_any = support.any(dim=1)
    for i in range(probs.shape[0]):
        rows = torch.nonzero(rows_any[i]).squeeze(1)
        if rows.numel() == 0:
            continue
        cols = torch.nonzero(cols_any[i]).squeeze(1)
        y1, y2 = max(0, int(rows[0]) - pad), min(IMG, int(rows[-1]) + 1 + pad)
        x1, x2 = max(0, int(cols[0]) - pad), min(IMG, int(cols[-1]) + 1 + pad)
        local = probs[i, y1:y2, x1:x2].float().half().cpu().numpy()
        box = boxes[i] if boxes is not None else (x1, y1, x2, y2)
        out.append({
            "win": (y1, y2, x1, x2),
            "prob": local,
            "score": float(scores[i]),
            "src": src,
            "idx": len(out),
            "box": tuple(float(v) for v in box),
        })
    return out


# ------------------------------------------------------------------------------ YOLO


def yolo_load(path):
    from ultralytics import YOLO

    return YOLO(str(guard(path)))


@torch.inference_mode()
def yolo_candidates(model, gray: np.ndarray, cfg: dict, device: str) -> list[dict]:
    y = cfg["yolo"]
    flips = list(y.get("tta_flips") or ["none"])
    views = []
    for mode in flips:
        img = np.ascontiguousarray(flip_np(gray, mode))
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        res = model.predict(
            source=bgr, imgsz=int(y["infer_imgsz"]), conf=float(y["infer_conf"]), iou=float(y["infer_iou"]),
            max_det=int(y["max_det"]), retina_masks=True, half=str(device) != "cpu", device=device, verbose=False,
        )[0]
        if res.masks is None or res.boxes is None or len(res.boxes) == 0:
            views.append([])
            continue
        masks = flip_t(res.masks.data, mode)
        if masks.shape[-2:] != (IMG, IMG):
            masks = F.interpolate(masks[None].float(), size=(IMG, IMG), mode="nearest")[0]
        scores = res.boxes.conf.float().cpu().numpy()
        boxes = [unflip_box(b, mode) for b in res.boxes.xyxy.float().cpu().numpy()]
        views.append(gpu_candidates(masks.float(), scores, "yolo", boxes=boxes, thr=0.5))
        del masks, res
    return views[0] if len(views) == 1 else fuse_sources(views, [1.0] * len(views), cfg["select"]["fuse_iou"])


# ----------------------------------------------------------------------- Mask2Former


def m2f_load(path, device, num_labels: int = 1):
    from transformers import Mask2FormerForUniversalSegmentation

    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        str(guard(path)), num_labels=num_labels, id2label={0: "filament"}, label2id={"filament": 0},
        ignore_mismatched_sizes=True,
    )
    return model.to(device)


@torch.inference_mode()
def m2f_candidates(model, gray: np.ndarray, cfg: dict, device, flips=None) -> list[dict]:
    m = cfg["mask2former"]
    size = int(m["image_size"])
    flips = list(flips if flips is not None else (m.get("tta_flips") or ["none"]))
    small = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    views = []
    for mode in flips:
        x = torch.from_numpy(norm3(np.ascontiguousarray(flip_np(small, mode))))[None].to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=str(device) != "cpu"):
            out = model(pixel_values=x)
        cls = out.class_queries_logits.float().softmax(-1)[0, :, 0]
        keep = torch.nonzero(cls >= float(m["min_class_prob"])).squeeze(1)
        if keep.numel() > int(m["topk"]):
            keep = keep[cls[keep].topk(int(m["topk"])).indices]
        if keep.numel() == 0:
            views.append([])
            continue
        logits = out.masks_queries_logits[0, keep].float()
        probs = flip_t(F.interpolate(logits[None], size=(IMG, IMG), mode="bilinear", align_corners=False)[0].sigmoid_(), mode)
        binary = probs > 0.5
        fill = (probs * binary).flatten(1).sum(1) / binary.flatten(1).sum(1).clamp(min=1)
        scores = (cls[keep] * fill).cpu().numpy()
        views.append(gpu_candidates(probs, scores, "mask2former", thr=0.05))
        del out, logits, probs, binary
    return views[0] if len(views) == 1 else fuse_sources(views, [1.0] * len(views), cfg["select"]["fuse_iou"])


# ------------------------------------------------------------------------------ SAM2

SAM_SIZE = 1024


def sam2_load(path, device, decoder_state=None):
    from transformers import Sam2Model

    model = Sam2Model.from_pretrained(str(guard(path)))
    if decoder_state is not None:
        model.mask_decoder.load_state_dict(torch.load(guard(decoder_state), map_location="cpu", weights_only=True))
    for name, p in model.named_parameters():
        p.requires_grad_(name.startswith("mask_decoder."))
    return model.to(device)


def square_crop(box, context: float, min_side: int, shift=(0.0, 0.0)):
    x1, y1, x2, y2 = (float(v) for v in box)
    side = int(max(min_side, math.ceil(max(x2 - x1, y2 - y1, 1.0) * context)))
    cx = (x1 + x2) / 2.0 + shift[0] * side
    cy = (y1 + y2) / 2.0 + shift[1] * side
    return int(round(cx - side / 2.0)), int(round(cy - side / 2.0)), side


def extract_crop(gray: np.ndarray, x0: int, y0: int, side: int, out: int = SAM_SIZE) -> np.ndarray:
    """Square crop with zero padding outside the image, resized to out x out."""
    h, w = gray.shape
    canvas = np.zeros((side, side), np.uint8)
    sx1, sy1, sx2, sy2 = max(0, x0), max(0, y0), min(w, x0 + side), min(h, y0 + side)
    if sx2 > sx1 and sy2 > sy1:
        canvas[sy1 - y0 : sy2 - y0, sx1 - x0 : sx2 - x0] = gray[sy1:sy2, sx1:sx2]
    interp = cv2.INTER_AREA if side > out else cv2.INTER_LINEAR
    return cv2.resize(canvas, (out, out), interpolation=interp)


def to_crop_box(box, x0, y0, side, out: int = SAM_SIZE):
    s = out / float(side)
    x1, y1, x2, y2 = (float(v) for v in box)
    return [np.clip((x1 - x0) * s, 0, out - 1), np.clip((y1 - y0) * s, 0, out - 1),
            np.clip((x2 - x0) * s, 1, out), np.clip((y2 - y0) * s, 1, out)]


def sam2_forward(model, pixels: torch.Tensor, boxes: torch.Tensor, device):
    """pixels (B, 3, 1024, 1024), boxes (B, 4) in 1024-pixel crop coordinates."""
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=str(device) != "cpu"):
        emb = model.get_image_embeddings(pixel_values=pixels)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=str(device) != "cpu"):
        out = model(image_embeddings=emb, input_boxes=boxes[:, None, :], multimask_output=False)
    return out.pred_masks[:, 0, 0].float(), out.iou_scores[:, 0, 0].float()


@torch.inference_mode()
def sam2_refine(model, gray: np.ndarray, cands: list[dict], cfg: dict, device, src_name: str) -> list[dict]:
    """Re-segment every candidate inside a context crop around its box (same idx and score)."""
    s = cfg["sam2"]
    out = []
    jobs = []
    for c in cands:
        x0, y0, side = square_crop(c["box"], float(s["context_infer"]), int(s["min_crop"]))
        jobs.append((c, x0, y0, side))
    bs = int(s["infer_batch"])
    for start in range(0, len(jobs), bs):
        chunk = jobs[start : start + bs]
        px = torch.from_numpy(np.stack([norm3(extract_crop(gray, x0, y0, side)) for _, x0, y0, side in chunk])).to(device)
        bx = torch.tensor([to_crop_box(c["box"], x0, y0, side) for c, x0, y0, side in chunk], dtype=torch.float32, device=device)
        logits, iou = sam2_forward(model, px, bx, device)
        for j, (c, x0, y0, side) in enumerate(chunk):
            prob = F.interpolate(logits[j][None, None], size=(side, side), mode="bilinear", align_corners=False)[0, 0].sigmoid()
            sy1, sx1 = max(0, y0), max(0, x0)
            sy2, sx2 = min(IMG, y0 + side), min(IMG, x0 + side)
            local = prob[sy1 - y0 : sy2 - y0, sx1 - x0 : sx2 - x0]
            support = local > 0.05
            rows = torch.nonzero(support.any(1)).squeeze(1)
            if rows.numel() == 0:
                continue
            cols = torch.nonzero(support.any(0)).squeeze(1)
            r0, r1 = int(rows[0]), int(rows[-1]) + 1
            c0, c1 = int(cols[0]), int(cols[-1]) + 1
            out.append({
                "win": (sy1 + r0, sy1 + r1, sx1 + c0, sx1 + c1),
                "prob": local[r0:r1, c0:c1].half().cpu().numpy(),
                "score": float(c["score"]),
                "src": src_name,
                "idx": int(c["idx"]),
                "box": tuple(float(v) for v in c["box"]),
                "sam_iou": float(iou[j]),
            })
        del px, bx, logits, iou
    return out
