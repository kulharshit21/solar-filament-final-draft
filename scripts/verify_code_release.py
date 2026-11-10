#!/usr/bin/env python3
"""Verify that this is the exact code-only 0.40 release."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

HISTORICAL_SHA256 = {
    "config.yaml": "02b39b81d8522b20c74babed1af055b2b47fc5014c79c55347de4a028b783e23",
    "requirements.txt": "8ebfcb8f3f74ac5c324c8ca9628c87c26258ca31fc947290bf89cbaff7212f03",
    "reports/validation_report.json": "561fd9bff53d93bb1e2c09f1b6c2e8e662eaa7e506e14202b1e462bc7ec93b4f",
    "scripts/00_verify.py": "cf3c74603d79c356d12935f11f8f260f54a78be7a3b2a2ae96a47ba3909974ff",
    "scripts/01_prepare.py": "80f3b1070fd85d8524403bdfa5d3456abda17fa4980ba40a5dc25eb916bfe92a",
    "scripts/02_train_yolo.py": "ef346a45285b57db89477abf1c0a377e1a68c398ea33c897f5d92964564e4d8b",
    "scripts/03_train_mask2former.py": "90240df218522b883bef2d9bc866fa1f9b5d25b9982c88d39991f3be02ed75b3",
    "scripts/04_train_sam2.py": "78162d00b8f07404ba8ee08f3e0f3fee4739aaa49c7fdbbf508414926f6e8593",
    "scripts/05_predict.py": "795b9cc5ca313ad6cb0e7fb7c4aebfe373f45902dd0b4a06bf11bc69a8c5f328",
    "scripts/06_select_and_submit.py": "706c171594f9359f949683b50c29db538141c54afe5a375cd0c1e0af415f53ab",
    "scripts/generate_w11.py": "1d6fcef1a750baa47ab60febc593025e37f364f6649a852e4677ab321c00747e",
    "scripts/verify_package.py": "066da842b20350b2103c41d2134156393d014951637417c279ea571474467f76",
    "src/solar/__init__.py": "652cfe118118b6300c8a2868b89a28a6095d1d37bc5a2be2bddc717ee7b60c7c",
    "src/solar/common.py": "0d831f53123123e5a2c0c4090da55e995ef51b682ef61bb71f691ec37991c757",
    "src/solar/models.py": "e0adf0281897e7ac25ed5849ae72a0c9d045ccdf0d0389e36535ada0684e7d64",
}

FORBIDDEN_RELEASE_PARTS = {"data", "dataset", "models", "weights", "submissions", "work"}
FORBIDDEN_SUFFIXES = {".pt", ".pth", ".ckpt", ".safetensors"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    problems: list[str] = []

    for relative, expected in HISTORICAL_SHA256.items():
        path = ROOT / relative
        if not path.is_file():
            problems.append(f"missing historical file: {relative}")
            continue
        actual = sha256(path)
        if actual != expected:
            problems.append(f"hash mismatch: {relative}: {actual}")

    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        relative = path.relative_to(ROOT)
        lowered = {part.lower() for part in relative.parts}
        if lowered & FORBIDDEN_RELEASE_PARTS:
            problems.append(f"forbidden release directory: {relative.as_posix()}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            problems.append(f"forbidden model binary: {relative.as_posix()}")
        if path.suffix.lower() == ".csv" and "submission" in path.name.lower():
            problems.append(f"forbidden submission CSV: {relative.as_posix()}")

    if problems:
        for problem in problems:
            print(f"FAIL: {problem}")
        raise SystemExit("VERDICT: FAIL")

    print(f"Verified {len(HISTORICAL_SHA256)} immutable historical source files.")
    print("Verified code-only publication boundary: no weights, dataset, or submission CSV.")
    print("VERDICT: PASS")


if __name__ == "__main__":
    main()

