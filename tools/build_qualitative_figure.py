"""Create a report-ready overlay from the exact verified 0.40 submission.

Requires local competition test images. It writes only a derived qualitative
figure, never raw test images or prediction caches, into reports/figures.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from pycocotools import mask as mask_utils

ROOT = Path(__file__).resolve().parents[1]
SUBMISSION = Path(r"D:\SOLAR GPT\Submission_040_Recreated_20261111\submission.csv")
IMAGE_DIR = Path(r"D:\SOLAR GPT\recreated_040_submission\data\test_images")
OUTPUT = ROOT / "reports" / "figures" / "qualitative_predictions.png"
STEMS = ("20110120105534Ch", "20110408083214Th", "20110214175414Mh")
COLORS = ((230, 57, 70), (29, 161, 242), (255, 183, 3), (67, 170, 139), (156, 39, 176))


def decode(rle: str) -> np.ndarray:
    encoded = {"size": [2048, 2048], "counts": rle.encode("ascii")}
    return mask_utils.decode(encoded).astype(bool)


def main() -> None:
    by_stem: dict[str, list[str]] = {stem: [] for stem in STEMS}
    with SUBMISSION.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            stem = row["filament_id"].rsplit("_", 1)[0]
            if stem in by_stem:
                by_stem[stem].append(row["segmentation_rle"])

    panels: list[Image.Image] = []
    for stem in STEMS:
        image_path = next(IMAGE_DIR.glob(f"{stem}.*"))
        base = Image.open(image_path).convert("RGB").resize((512, 512))
        arr = np.asarray(base).copy()
        masks = [decode(value) for value in by_stem[stem]]
        for index, full_mask in enumerate(masks):
            mask = np.asarray(Image.fromarray(full_mask).resize((512, 512), Image.Resampling.NEAREST))
            color = np.array(COLORS[index % len(COLORS)], dtype=np.uint8)
            arr[mask] = (0.45 * arr[mask] + 0.55 * color).astype(np.uint8)
        panel = Image.fromarray(arr)
        draw = ImageDraw.Draw(panel)
        draw.rectangle((0, 0, 512, 30), fill=(0, 0, 0))
        draw.text((8, 8), f"{stem}: {len(masks)} predicted instances", fill=(255, 255, 255))
        panels.append(panel)

    canvas = Image.new("RGB", (1536, 512), "white")
    for index, panel in enumerate(panels):
        canvas.paste(panel, (index * 512, 0))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUTPUT, quality=95)
    print(OUTPUT)


if __name__ == "__main__":
    main()
