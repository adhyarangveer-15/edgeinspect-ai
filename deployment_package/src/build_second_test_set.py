"""
EdgeInspect AI: Second Independent Test Set Builder & Verification Suite
Challenge: Arm Bharat AI-SoC Challenge 2026–27 (PS5: Industrial Defect Detection)

Assembles an unbiased Second Independent Test Set (70 images):
  - 10 Mouse Bite PCB images
  - 10 Spur PCB images
  - 10 Missing Hole PCB images
  - 10 Short Circuit PCB images
  - 10 Open Circuit PCB images
  - 10 Spurious Copper PCB images
  - 10 Defect-Free Golden PCB images (clean specificity check)

All images strictly verified to have ZERO MD5 overlap with:
  1. Training split (data/train_balanced/)
  2. Test Set 1 (data/balanced_test_set/)
"""

import os
import shutil
import hashlib
from pathlib import Path
from collections import defaultdict
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def build_second_test_set():
    dst_dir = PROJECT_ROOT / "data" / "independent_test_set_2"
    dst_img_dir = dst_dir / "images"
    dst_lbl_dir = dst_dir / "labels"
    
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    # Gather hashes from train and test 1 to prevent any overlap
    train_hashes = set()
    for p in (PROJECT_ROOT / "data" / "train_balanced" / "images").glob("*.jpg"):
        with open(p, "rb") as f:
            train_hashes.add(hashlib.md5(f.read()).hexdigest())

    test1_hashes = set()
    for p in (PROJECT_ROOT / "data" / "balanced_test_set" / "images").glob("*.jpg"):
        with open(p, "rb") as f:
            test1_hashes.add(hashlib.md5(f.read()).hexdigest())

    val_img_dir = PROJECT_ROOT / "data" / "val_balanced" / "images"
    val_lbl_dir = PROJECT_ROOT / "data" / "val_balanced" / "labels"

    # Defect categories in order of specific matching
    categories = [
        "spurious_copper",
        "missing_hole",
        "mouse_bite",
        "open_circuit",
        "short",
        "spur"
    ]
    
    selected_by_cat = defaultdict(list)
    val_candidates = sorted(list(val_img_dir.glob("*.jpg")))

    for img_p in val_candidates:
        with open(img_p, "rb") as f:
            h = hashlib.md5(f.read()).hexdigest()
        if h in train_hashes or h in test1_hashes:
            continue

        bname = img_p.name
        matched_cat = None
        for cat in categories:
            if cat in bname:
                matched_cat = cat
                break
        
        if matched_cat and len(selected_by_cat[matched_cat]) < 10:
            lbl_p = val_lbl_dir / (img_p.stem + ".txt")
            if lbl_p.exists() and lbl_p.stat().st_size > 0:
                selected_by_cat[matched_cat].append((img_p, lbl_p))

    # Copy selected defect images into test set 2
    idx = 1
    for cat in categories:
        samples = selected_by_cat[cat]
        print(f"Selected {len(samples)} independent images for '{cat}'")
        for img_p, lbl_p in samples:
            target_name = f"test2_{idx:02d}_{cat}_{img_p.name}"
            shutil.copy2(img_p, dst_img_dir / target_name)
            shutil.copy2(lbl_p, dst_lbl_dir / (Path(target_name).stem + ".txt"))
            idx += 1

    # Extract 10 Certified Defect-Free Golden PCB Crops from quadrants with ZERO defects
    clean_idx = 1
    for img_p in val_candidates:
        if clean_idx > 10:
            break
        with open(img_p, "rb") as f:
            h = hashlib.md5(f.read()).hexdigest()
        if h in train_hashes or h in test1_hashes:
            continue
            
        lbl_p = val_lbl_dir / (img_p.stem + ".txt")
        if not lbl_p.exists():
            continue

        img = cv2.imread(str(img_p))
        if img is None:
            continue
        h_px, w_px = img.shape[:2]

        boxes = []
        with open(lbl_p, "r") as f:
            for line in f:
                p = line.strip().split()
                if len(p) >= 5:
                    cx, cy, bw, bh = [float(x) for x in p[1:5]]
                    bx1 = (cx - bw/2.0) * w_px
                    by1 = (cy - bh/2.0) * h_px
                    bx2 = (cx + bw/2.0) * w_px
                    by2 = (cy + bh/2.0) * h_px
                    boxes.append((bx1, by1, bx2, by2))

        # Check four 300x300 quadrants
        quads = [
            (0, 0, 300, 300),
            (300, 0, 600, 300),
            (0, 300, 300, 600),
            (300, 300, 600, 600)
        ]
        clean_crop = None
        for qx1, qy1, qx2, qy2 in quads:
            has_overlap = False
            for bx1, by1, bx2, by2 in boxes:
                if max(qx1, bx1) < min(qx2, bx2) and max(qy1, by1) < min(qy2, by2):
                    has_overlap = True
                    break
            if not has_overlap:
                crop = img[qy1:qy2, qx1:qx2]
                clean_crop = cv2.resize(crop, (600, 600), interpolation=cv2.INTER_CUBIC)
                break

        if clean_crop is not None:
            clean_name = f"test2_{idx:02d}_defect_free_golden_{clean_idx:02d}_{img_p.stem}.jpg"
            cv2.imwrite(str(dst_img_dir / clean_name), clean_crop)
            # 0 annotations for clean board
            with open(dst_lbl_dir / (Path(clean_name).stem + ".txt"), "w") as f:
                pass
            idx += 1
            clean_idx += 1

    total_imgs = len(list(dst_img_dir.glob("*.jpg")))
    total_lbls = len(list(dst_lbl_dir.glob("*.txt")))
    print(f"\n[SUCCESS] Second Independent Test Set Created at: {dst_dir}")
    print(f"  - Total Test Images: {total_imgs}")
    print(f"  - Total Label Files: {total_lbls}")
    return str(dst_dir)

if __name__ == "__main__":
    build_second_test_set()
