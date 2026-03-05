"""
validate.py
===========
Standalone two-stage validation program.

Runs Stage 1 detection → Stage 2 segmentation → remaps masks back to
original image space → compares against ground-truth YOLO segmentation
labels → reports Precision, Recall, mAP@50, mAP@50:95 per class and overall.

No dependency on any other pipeline file.

Usage:
    python validate.py \
        --stage1  /path/to/stage1_detector.pt \
        --stage2  /path/to/stage2_seg.pt \
        --dataset /path/to/test/dataset \
        --split   test \
        --imgsz   640 \
        --conf    0.25 \
        --iou     0.45 \
        --workers 4

Dataset must follow standard YOLO structure:
    dataset/
      train/
        images/
        labels/
      val/
        images/
        labels/
      test/
        images/
        labels/

Label format (YOLO segmentation, normalized polygons):
    <class_id> x1 y1 x2 y2 ... xN yN
"""

import argparse
import sys
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from ultralytics import YOLO


# =============================================================================
# Data structures
# =============================================================================

class Instance:
    """One segmentation instance (predicted or GT)."""
    def __init__(
        self,
        class_id: int,
        polygon_abs: np.ndarray,   # (N,2) absolute pixel coords in original image
        confidence: float = 1.0,   # always 1.0 for GT
        mask: Optional[np.ndarray] = None,  # pre-computed boolean mask (img_h, img_w)
    ):
        self.class_id    = class_id
        self.polygon_abs = polygon_abs
        self.confidence  = confidence
        self.mask        = mask


# =============================================================================
# I/O helpers
# =============================================================================

def image_files(split_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    return sorted([p for p in split_dir.rglob("*") if p.suffix.lower() in exts])


def parse_yolo_seg_label(label_path: Path, img_w: int, img_h: int) -> List[Instance]:
    """
    Parse a YOLO segmentation label file.
    Returns instances with polygons in absolute pixel coords.
    """
    instances = []
    if not label_path.exists():
        return instances
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            class_id = int(parts[0])
            coords = np.array(parts[1:], dtype=np.float32).reshape(-1, 2)
            # Denormalize
            coords[:, 0] *= img_w
            coords[:, 1] *= img_h
            instances.append(Instance(class_id=class_id, polygon_abs=coords))
    return instances


def get_label_path(img_path: Path, dataset_root: Path) -> Optional[Path]:
    # img_path is e.g. root/test/images/foo.jpg
    images_dir = img_path.parent
    split_dir  = images_dir.parent          # e.g. root/test
    p = split_dir / "labels" / img_path.with_suffix(".txt").name
    return p if p.exists() else None


# =============================================================================
# Polygon → binary mask rasterization
# =============================================================================

def poly_to_mask(polygon_abs: np.ndarray, img_h: int, img_w: int) -> np.ndarray:
    """
    Rasterize an absolute-pixel polygon to a boolean mask of shape (img_h, img_w).
    """
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    pts  = np.round(polygon_abs).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)


# =============================================================================
# IoU between two binary masks
# =============================================================================

def mask_iou(m1: np.ndarray, m2: np.ndarray) -> float:
    inter = (m1 & m2).sum()
    union = (m1 | m2).sum()
    return float(inter) / float(union) if union > 0 else 0.0


# =============================================================================
# Stage 1: detect + crop
# =============================================================================

def detect_and_crop(
    img_path: Path,
    detector: YOLO,
    conf: float,
    iou: float,
    imgsz: int,
) -> List[Dict]:
    """
    Run Stage 1 detection on one image.
    Returns list of crop dicts:
        {
            "crop_img" : np.ndarray (BGR),
            "int_x1","int_y1","int_x2","int_y2": int  (integer pixel coords used
                                                         to extract the crop),
            "crop_w","crop_h": int  (actual crop pixel dimensions),
            "img_w","img_h": int
        }
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return []
    img_h, img_w = img.shape[:2]

    results = detector.predict(
        source=str(img_path),
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        verbose=False,
    )

    crops = []
    if not results or results[0].boxes is None:
        return crops

    for box in results[0].boxes.xyxy.cpu().numpy():
        # Clamp float coords to image bounds
        x1 = max(0.0, float(box[0]))
        y1 = max(0.0, float(box[1]))
        x2 = min(float(img_w), float(box[2]))
        y2 = min(float(img_h), float(box[3]))

        # --- FIX: use integer coords that match the actual crop slice ---
        int_x1, int_y1 = int(x1), int(y1)
        int_x2, int_y2 = int(x2), int(y2)

        cw = int_x2 - int_x1
        ch = int_y2 - int_y1
        if cw < 2 or ch < 2:
            continue

        crop_img = img[int_y1:int_y2, int_x1:int_x2]

        crops.append({
            "crop_img": crop_img,
            "int_x1": int_x1, "int_y1": int_y1,
            "int_x2": int_x2, "int_y2": int_y2,
            "crop_w": cw, "crop_h": ch,
            "img_w": img_w, "img_h": img_h,
        })
    return crops


# =============================================================================
# Stage 2: segment crop → remap to original image space
# =============================================================================

def segment_and_remap(
    crop_info: Dict,
    segmentor: YOLO,
    conf: float,
    iou: float,
    imgsz: int,
) -> List[Instance]:
    """
    Run Stage 2 segmentation on one crop, then remap predicted masks
    back to the original image coordinate space.

    Uses `r.masks.data` (binary mask at crop resolution) directly to avoid
    the lossy polygon roundtrip that degrades mask IoU at strict thresholds.
    The binary mask is placed into a full-size canvas at the integer crop
    origin.  Polygon coordinates from `r.masks.xy` are still shifted and
    stored for reference but are not used for IoU evaluation.
    """
    crop_img = crop_info["crop_img"]
    # --- FIX: use integer offsets that match the actual crop origin ---
    int_x1 = crop_info["int_x1"]
    int_y1 = crop_info["int_y1"]
    crop_w = crop_info["crop_w"]
    crop_h = crop_info["crop_h"]
    img_w  = crop_info["img_w"]
    img_h  = crop_info["img_h"]

    results = segmentor.predict(
        source=crop_img,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        verbose=False,
    )

    instances = []
    if not results or results[0].masks is None:
        return instances

    r        = results[0]
    boxes    = r.boxes
    masks_xy = r.masks.xy   # list of np arrays, pixel coords in crop space

    for i, poly_crop_px in enumerate(masks_xy):
        if poly_crop_px.shape[0] < 3:
            continue

        class_id   = int(boxes.cls[i].item())
        confidence = float(boxes.conf[i].item())

        # --- FIX: shift by integer crop origin, then clamp to image bounds ---
        abs_poly        = poly_crop_px.copy().astype(np.float32)
        abs_poly[:, 0]  = poly_crop_px[:, 0] + int_x1
        abs_poly[:, 1]  = poly_crop_px[:, 1] + int_y1

        # Clamp to original image boundaries
        abs_poly[:, 0]  = np.clip(abs_poly[:, 0], 0, img_w - 1)
        abs_poly[:, 1]  = np.clip(abs_poly[:, 1], 0, img_h - 1)

        # --- FIX: use binary mask from masks.data for accurate stitching ---
        # Using the binary mask directly avoids the lossy polygon roundtrip
        # (mask → polygon contour → rasterize) that introduces quantization
        # errors at mask boundaries, degrading IoU at strict thresholds.
        crop_mask_raw = r.masks.data[i].cpu().numpy()

        # Ensure mask is at crop resolution (may differ if model returns
        # masks at inference or proto resolution)
        if crop_mask_raw.shape[0] != crop_h or crop_mask_raw.shape[1] != crop_w:
            crop_mask = cv2.resize(
                crop_mask_raw.astype(np.float32), (crop_w, crop_h),
                interpolation=cv2.INTER_LINEAR,
            ) >= 0.5
        else:
            crop_mask = crop_mask_raw >= 0.5

        # Stitch: place crop mask into full-size image at the crop position
        full_mask = np.zeros((img_h, img_w), dtype=bool)
        y_end = min(int_y1 + crop_h, img_h)
        x_end = min(int_x1 + crop_w, img_w)
        paste_h = y_end - int_y1
        paste_w = x_end - int_x1
        full_mask[int_y1:y_end, int_x1:x_end] = crop_mask[:paste_h, :paste_w]

        instances.append(Instance(
            class_id    = class_id,
            polygon_abs = abs_poly,
            confidence  = confidence,
            mask        = full_mask,
        ))

    return instances


# =============================================================================
# Matching: predictions vs GT for one image
# =============================================================================

def match_predictions_to_gt(
    preds: List[Instance],
    gts:   List[Instance],
    img_h: int,
    img_w: int,
    iou_threshold: float,
) -> Tuple[List[Tuple], List[bool]]:
    """
    Match predicted instances to GT instances using mask IoU.

    Returns:
        matches : list of (pred_idx, gt_idx, iou, class_id) for true positives
        gt_matched : bool list, True if GT instance was matched
    """
    if not preds or not gts:
        return [], [False] * len(gts)

    # Rasterize all masks once (use pre-computed mask for predictions if available)
    pred_masks = [
        p.mask if p.mask is not None
        else poly_to_mask(p.polygon_abs, img_h, img_w)
        for p in preds
    ]
    gt_masks   = [poly_to_mask(g.polygon_abs, img_h, img_w) for g in gts]

    gt_matched  = [False] * len(gts)
    matches     = []

    # Sort predictions by confidence descending
    pred_order = sorted(range(len(preds)), key=lambda i: preds[i].confidence, reverse=True)

    for pi in pred_order:
        pred      = preds[pi]
        best_iou  = iou_threshold - 1e-9
        best_gi   = -1

        for gi, gt in enumerate(gts):
            if gt_matched[gi]:
                continue
            if gt.class_id != pred.class_id:
                continue
            iou = mask_iou(pred_masks[pi], gt_masks[gi])
            if iou > best_iou:
                best_iou = iou
                best_gi  = gi

        if best_gi >= 0:
            gt_matched[best_gi] = True
            matches.append((pi, best_gi, best_iou, pred.class_id))

    return matches, gt_matched


# =============================================================================
# AP computation
# =============================================================================

def compute_ap(precisions: np.ndarray, recalls: np.ndarray) -> float:
    """
    Compute AP using the 101-point interpolation (COCO style).
    """
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        mask = recalls >= t
        p    = precisions[mask].max() if mask.any() else 0.0
        ap  += p / 101.0
    return ap


def compute_ap_for_class(
    class_id:      int,
    all_preds:     List[Tuple],   # (confidence, is_tp)  across all images
    total_gt:      int,
    iou_threshold: float,
) -> Tuple[float, float, float]:
    """
    Returns (AP, best_precision, best_recall) for one class at one IoU threshold.
    """
    if total_gt == 0 or not all_preds:
        return 0.0, 0.0, 0.0

    # Sort by confidence descending
    all_preds_sorted = sorted(all_preds, key=lambda x: x[0], reverse=True)
    tp_list = np.array([x[1] for x in all_preds_sorted], dtype=np.float32)

    tp_cum  = np.cumsum(tp_list)
    fp_cum  = np.cumsum(1 - tp_list)

    recalls    = tp_cum / total_gt
    precisions = tp_cum / (tp_cum + fp_cum + 1e-9)

    ap = compute_ap(precisions, recalls)

    best_p = float(precisions[-1]) if len(precisions) else 0.0
    best_r = float(recalls[-1])    if len(recalls)    else 0.0

    return ap, best_p, best_r


# =============================================================================
# Main validation loop
# =============================================================================

def run_validation(args: argparse.Namespace) -> None:

    dataset_root = Path(args.dataset)
    img_dir      = dataset_root / args.split / "images"

    if not img_dir.exists():
        print(f"[ERROR] Image directory not found: {img_dir}")
        sys.exit(1)

    images = image_files(img_dir)
    if not images:
        print(f"[ERROR] No images found in {img_dir}")
        sys.exit(1)

    print(f"Loading Stage 1 detector : {args.stage1}")
    detector  = YOLO(args.stage1)

    print(f"Loading Stage 2 segmentor: {args.stage2}")
    segmentor = YOLO(args.stage2)

    print(f"Dataset split            : {args.split}  ({len(images)} images)")
    print(f"IoU match thresholds     : 0.50 … 0.95 (step 0.05)")
    print()

    # Collect class names from Stage 2 model
    class_names: Dict[int, str] = segmentor.names  # {int: str}
    all_class_ids = sorted(class_names.keys())

    # Storage:
    # per_class_preds[iou_thresh][class_id] = [(confidence, is_tp), ...]
    # per_class_gt_counts[class_id]         = int
    # per_class_ious[class_id]              = [iou, ...] (TP IoUs at iou_t == 0.50)
    iou_thresholds  = [round(t, 2) for t in np.arange(0.50, 1.00, 0.05)]
    per_class_preds = {t: {c: [] for c in all_class_ids} for t in iou_thresholds}
    per_class_gt    = {c: 0 for c in all_class_ids}
    per_class_ious: Dict[int, List[float]] = {c: [] for c in all_class_ids}

    n_images_with_gt    = 0
    n_images_no_det     = 0
    n_images_no_label   = 0

    for img_idx, img_path in enumerate(images):
        print(f"\r  Processing {img_idx+1}/{len(images)} ...", end="", flush=True)

        # ------------------------------------------------------------------ #
        # Load GT
        # ------------------------------------------------------------------ #
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_h, img_w = img.shape[:2]

        label_path = get_label_path(img_path, dataset_root)
        if label_path is None:
            n_images_no_label += 1
            continue

        gt_instances = parse_yolo_seg_label(label_path, img_w, img_h)
        if not gt_instances:
            continue

        n_images_with_gt += 1
        for gt in gt_instances:
            if gt.class_id in per_class_gt:
                per_class_gt[gt.class_id] += 1

        # ------------------------------------------------------------------ #
        # Stage 1: detect crops
        # ------------------------------------------------------------------ #
        crops = detect_and_crop(img_path, detector,
                                conf=args.conf, iou=args.iou, imgsz=args.imgsz)
        if not crops:
            n_images_no_det += 1
            # All GT instances are FN — no predictions to log, gt counts already added
            continue

        # ------------------------------------------------------------------ #
        # Stage 2: segment all crops, collect remapped predictions
        # ------------------------------------------------------------------ #
        all_preds: List[Instance] = []
        for crop_info in crops:
            preds = segment_and_remap(crop_info, segmentor,
                                      conf=args.conf, iou=args.iou, imgsz=args.imgsz)
            all_preds.extend(preds)

        # ------------------------------------------------------------------ #
        # Match predictions vs GT at each IoU threshold
        # ------------------------------------------------------------------ #
        for iou_t in iou_thresholds:
            matches, _ = match_predictions_to_gt(
                all_preds, gt_instances, img_h, img_w, iou_t
            )
            matched_pred_indices = {m[0] for m in matches}

            # Collect TP IoU values at the primary threshold (0.50) for mIoU
            if iou_t == 0.50:
                for _, _, match_iou, match_cls in matches:
                    if match_cls in per_class_ious:
                        per_class_ious[match_cls].append(match_iou)

            for pi, pred in enumerate(all_preds):
                if pred.class_id not in per_class_preds[iou_t]:
                    continue
                is_tp = 1.0 if pi in matched_pred_indices else 0.0
                per_class_preds[iou_t][pred.class_id].append(
                    (pred.confidence, is_tp)
                )

    print(f"\r  Processed {len(images)}/{len(images)} images.          ")

    # -----------------------------------------------------------------------
    # Compute per-class AP at each IoU threshold
    # -----------------------------------------------------------------------
    # ap_table[class_id][iou_thresh] = ap value
    ap_table: Dict[int, Dict[float, float]] = {c: {} for c in all_class_ids}
    pr_table: Dict[int, Dict[float, Tuple]] = {c: {} for c in all_class_ids}

    for iou_t in iou_thresholds:
        for c in all_class_ids:
            ap, best_p, best_r = compute_ap_for_class(
                c,
                per_class_preds[iou_t][c],
                per_class_gt[c],
                iou_t,
            )
            ap_table[c][iou_t] = ap
            pr_table[c][iou_t] = (best_p, best_r)

    # -----------------------------------------------------------------------
    # Aggregate metrics
    # -----------------------------------------------------------------------
    def mean_ap_at(iou_t: float) -> float:
        vals = [ap_table[c][iou_t] for c in all_class_ids if per_class_gt[c] > 0]
        return float(np.mean(vals)) if vals else 0.0

    def mean_ap_range(t_start: float, t_end: float, step: float) -> float:
        thresholds = [round(t, 2) for t in np.arange(t_start, t_end + 1e-9, step)]
        vals = [mean_ap_at(t) for t in thresholds if t in iou_thresholds]
        return float(np.mean(vals)) if vals else 0.0

    map50    = mean_ap_at(0.50)
    map5095  = mean_ap_range(0.50, 0.95, 0.05)

    # Per-class mean IoU (from TP matches at IoU threshold 0.50)
    per_class_miou: Dict[int, float] = {}
    for c in all_class_ids:
        if per_class_ious[c]:
            per_class_miou[c] = float(np.mean(per_class_ious[c]))
        else:
            per_class_miou[c] = 0.0

    classes_with_gt = [c for c in all_class_ids if per_class_gt[c] > 0]
    overall_miou = float(np.mean([per_class_miou[c] for c in classes_with_gt])) if classes_with_gt else 0.0

    # -----------------------------------------------------------------------
    # Print results
    # -----------------------------------------------------------------------
    SEP  = "=" * 80
    sep2 = "-" * 80

    print()
    print(SEP)
    print("  VALIDATION RESULTS")
    print(SEP)
    print(f"  Split            : {args.split}")
    print(f"  Total images     : {len(images)}")
    print(f"  Images with GT   : {n_images_with_gt}")
    print(f"  Images no label  : {n_images_no_label}")
    print(f"  Images no detect : {n_images_no_det}")
    print(sep2)

    # Per-class table
    print(f"  {'Class':<20} {'GT':>6} {'P@50':>8} {'R@50':>8} {'AP@50':>8} {'AP@50:95':>10} {'mIoU':>8}")
    print(sep2)

    class_aps50   = []
    class_aps5095 = []

    for c in all_class_ids:
        if per_class_gt[c] == 0:
            continue
        name    = class_names.get(c, str(c))
        gt_n    = per_class_gt[c]
        p50, r50 = pr_table[c][0.50]
        ap50    = ap_table[c][0.50]
        ap5095  = float(np.mean([ap_table[c][t] for t in iou_thresholds]))
        miou_c  = per_class_miou[c]

        class_aps50.append(ap50)
        class_aps5095.append(ap5095)

        print(
            f"  {name:<20} {gt_n:>6} "
            f"{p50:>8.4f} {r50:>8.4f} "
            f"{ap50:>8.4f} {ap5095:>10.4f} {miou_c:>8.4f}"
        )

    print(sep2)

    # Overall row
    overall_p50 = float(np.mean([pr_table[c][0.50][0] for c in all_class_ids if per_class_gt[c] > 0])) if class_aps50 else 0.0
    overall_r50 = float(np.mean([pr_table[c][0.50][1] for c in all_class_ids if per_class_gt[c] > 0])) if class_aps50 else 0.0
    total_gt    = sum(per_class_gt.values())

    print(
        f"  {'ALL':<20} {total_gt:>6} "
        f"{overall_p50:>8.4f} {overall_r50:>8.4f} "
        f"{map50:>8.4f} {map5095:>10.4f} {overall_miou:>8.4f}"
    )
    print(sep2)
    print(f"  mAP@50       : {map50:.4f}")
    print(f"  mAP@50:95    : {map5095:.4f}")
    print(f"  Mean IoU     : {overall_miou:.4f}")
    print(SEP)
    print()

    # Per-threshold breakdown
    print("  AP per IoU threshold (all classes mean):")
    print(f"  {'IoU':<10} {'mAP':>8}")
    print("  " + "-" * 20)
    for t in iou_thresholds:
        print(f"  {t:<10.2f} {mean_ap_at(t):>8.4f}")
    print()


# =============================================================================
# Entry point
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Two-stage YOLO validation: detection → segmentation → remap → metrics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--stage1", required=True,
        help="Path to pretrained Stage 1 YOLO detection model (.pt)",
    )
    parser.add_argument(
        "--stage2", required=True,
        help="Path to trained Stage 2 YOLO segmentation model (.pt)",
    )
    parser.add_argument(
        "--dataset", required=True,
        help="Root path of the YOLO-format test dataset (must contain images/ and labels/)",
    )
    parser.add_argument(
        "--split", default="test",
        help="Dataset split to evaluate on (train / val / test)",
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="Inference image size for both Stage 1 and Stage 2",
    )
    parser.add_argument(
        "--conf", type=float, default=0.25,
        help="Confidence threshold for both Stage 1 and Stage 2",
    )
    parser.add_argument(
        "--iou", type=float, default=0.45,
        help="NMS IoU threshold for both Stage 1 and Stage 2",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Number of dataloader workers (currently informational)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_validation(args)
