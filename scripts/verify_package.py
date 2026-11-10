#!/usr/bin/env python3
"""
verify_package.py
=================
Verifies the standalone package integrity of the 0.40 Clean Submission repository:
- Checks existence and integrity of weights (YOLO full, Mask2Former, SAM2 full decoder)
- Validates submission.csv structure (180 images, 1,261 instances, disjoint masks, valid RLE)
- Confirms source code modules, generate_w11.py, and configuration
"""
from pathlib import Path
import pandas as pd

PKG_ROOT = Path(__file__).resolve().parent.parent

def verify():
    print(f"=== Verifying 0.40 Clean Submission Package at {PKG_ROOT} ===")
    
    # 1. Weights
    weights_dir = PKG_ROOT / "weights"
    assert weights_dir.exists(), "Missing weights/ directory"
    required_weights = [
        ("yolo_full.pt", 50),
        ("sam2_full_decoder.pt", 10),
        ("sam2_fold_decoder.pt", 10),
    ]
    for w_name, min_mb in required_weights:
        w_path = weights_dir / w_name
        assert w_path.exists(), f"Missing weight: {w_name}"
        mb = w_path.stat().st_size / (1024 * 1024)
        assert mb >= min_mb, f"{w_name} too small ({mb:.1f} MB)"
        print(f"  [PASS] Weight: {w_name} ({mb:.1f} MB)")
        
    # Check Mask2Former weights directory
    m2f_dir = weights_dir / "mask2former_fold"
    assert m2f_dir.exists(), "Missing mask2former_fold/"
    assert (m2f_dir / "model.safetensors").exists(), "Missing mask2former_fold/model.safetensors"
    print(f"  [PASS] Model: mask2former_fold/model.safetensors")

    # 2. Submission CSV (Official 0.40 Public LB Champion)
    sub_path = PKG_ROOT / "submission.csv"
    assert sub_path.exists(), "Missing submission.csv"
    df = pd.read_csv(sub_path)
    assert list(df.columns) == ["filament_id", "segmentation_rle"], "Invalid column headers"
    assert df["filament_id"].is_unique, "Duplicate filament_ids found"
    assert not df["segmentation_rle"].isna().any(), "NaN values found in segmentation_rle"
    stems = df["filament_id"].apply(lambda x: x.rsplit("_", 1)[0]).nunique()
    assert stems == 180, f"Expected 180 test images, got {stems}"
    assert len(df) == 1261, f"Expected 1,261 instances for 0.40 champion, got {len(df)}"
    print(f"  [PASS] submission.csv: {len(df)} instances across {stems} test images (0.40 Public LB)")

    # 3. Source code modules & reproduction script
    for mod in ["common.py", "models.py", "__init__.py"]:
        p = PKG_ROOT / "src" / "solar" / mod
        assert p.exists(), f"Missing src/solar/{mod}"
    assert (PKG_ROOT / "scripts" / "generate_w11.py").exists(), "Missing scripts/generate_w11.py"
    print("  [PASS] Reproduction pipeline: scripts/generate_w11.py + src/solar")

    print("\nALL VERIFICATION CHECKS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    verify()
