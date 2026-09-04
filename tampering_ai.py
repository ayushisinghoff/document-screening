"""
============================================================
 AI-BASED DOCUMENT TAMPERING DETECTION — SINGLE-FILE MODULE
============================================================
Module 3 of the Border Checkpoint Document Screening System.

Covers all four tampering use cases in one file:
  1. Photo Replacement    -> ELA + Copy-Move
  2. Text Manipulation    -> ELA + Text Consistency
  3. Stamp Forgery        -> ELA + Copy-Move
  4. Image Metadata       -> Metadata Analysis

Sections in this file (search for "# ==="):
  1. METADATA ANALYSIS
  2. ERROR LEVEL ANALYSIS (ELA)
  3. COPY-MOVE (CLONE) DETECTION
  4. TEXT / FIELD CONSISTENCY ANALYSIS
  5. AGGREGATE RISK SCORER
  6. DEMO / CLI ENTRY POINT

Usage:
    pip install -r requirements.txt   # opencv-python, numpy, Pillow, scikit-image, scipy
    python tampering_ai.py path/to/document.jpg
    python tampering_ai.py path/to/document.jpg --panel output_panel.png

Integration with the rest of the pipeline:
    from tampering_ai import assess_document
    field_boxes = ocr_module.get_field_boxes(image_path)   # from Module 1
    report = assess_document(image_path, field_boxes)
    report["risk_level"]     # "LOW" / "MEDIUM" / "HIGH"
    report["overall_score"]  # 0.0 - 1.0
    report["all_flags"]      # human-readable reasons for the officer's UI
    report["details"]        # full audit trail per sub-module, for storage
"""

import io
import sys
import json
import argparse

import cv2
import numpy as np
from PIL import Image, ImageChops
from PIL.ExifTags import TAGS
from datetime import datetime


# ============================================================
# 1. METADATA ANALYSIS
# ============================================================
# Flags documents whose EXIF/metadata suggests digital editing: software
# tags left behind by editors, missing EXIF on a photo claiming to be a
# camera/scanner capture, or timestamp mismatches between capture and
# modification. This is a *signal*, not proof — many legitimate scans
# have no EXIF, and many forgeries strip metadata deliberately. Treat it
# as one input into the aggregate score, never a standalone verdict.

SUSPICIOUS_SOFTWARE_KEYWORDS = [
    "photoshop", "gimp", "paint.net", "pixlr", "affinity photo",
    "lightroom", "snapseed", "picsart", "canva", "inkscape"
]


def extract_exif(image_path: str) -> dict:
    """Return a plain dict of EXIF tag_name -> value. Empty dict if none."""
    exif_data = {}
    try:
        img = Image.open(image_path)
        raw_exif = img.getexif()
        if not raw_exif:
            return exif_data
        for tag_id, value in raw_exif.items():
            tag_name = TAGS.get(tag_id, tag_id)
            exif_data[tag_name] = value
    except Exception:
        pass
    return exif_data


def _parse_exif_datetime(value):
    try:
        return datetime.strptime(str(value), "%Y:%m:%d %H:%M:%S")
    except Exception:
        return None


def analyze_metadata(image_path: str) -> dict:
    """
    Returns:
        {
            "score": float 0-1  (higher = more suspicious),
            "flags": [str, ...],
            "exif_present": bool,
            "software_tag": str | None
        }
    """
    exif = extract_exif(image_path)
    flags = []
    score = 0.0
    exif_present = len(exif) > 0

    # 1. Editing-software signature left in metadata
    software_tag = exif.get("Software")
    if software_tag:
        sw_lower = str(software_tag).lower()
        if any(kw in sw_lower for kw in SUSPICIOUS_SOFTWARE_KEYWORDS):
            flags.append(f"Editing software detected in metadata: '{software_tag}'")
            score += 0.5

    # 2. Timestamp inconsistency (DateTimeOriginal vs DateTime / ModifyDate)
    dt_original = _parse_exif_datetime(exif.get("DateTimeOriginal"))
    dt_modified = _parse_exif_datetime(exif.get("DateTime"))
    if dt_original and dt_modified and dt_modified > dt_original:
        delta = (dt_modified - dt_original).total_seconds()
        if delta > 60:  # more than a minute gap = re-saved after capture
            flags.append(
                f"Modify time is {int(delta)}s after original capture time — file was re-saved"
            )
            score += 0.3

    # 3. No EXIF at all — moderate flag only (scans legitimately lack EXIF,
    #    so this alone should never dominate the score)
    if not exif_present:
        flags.append("No EXIF metadata present (could be a scan, screenshot, or stripped file)")
        score += 0.1

    # 4. Missing DateTimeOriginal but has other EXIF (common in re-exported images)
    if exif_present and "DateTimeOriginal" not in exif and "DateTime" in exif:
        flags.append("Original capture timestamp missing while other EXIF is present")
        score += 0.15

    score = min(score, 1.0)
    return {
        "score": round(score, 3),
        "flags": flags,
        "exif_present": exif_present,
        "software_tag": str(software_tag) if software_tag else None,
    }


# ============================================================
# 2. ERROR LEVEL ANALYSIS (ELA)
# ============================================================
# The workhorse technique for Photo Replacement, Text Manipulation, and
# Stamp Forgery in a single pass: all three share a common fingerprint —
# a region that was pasted/edited was compressed at a *different*
# JPEG quality/generation than the rest of the document.
#
# 1. Re-save the image at a known JPEG quality (e.g. 90).
# 2. Diff the original against the resave, pixel-wise.
# 3. Untouched regions (compressed together, same generation) show a
#    low, uniform error level. Edited/pasted regions show a distinctly
#    different error level.
# 4. Amplify and threshold the difference map to get suspicious regions.
#
# Works best on JPEG; PNG/scanned docs have a weaker ELA signal, so the
# aggregate scorer leans more on copy-move + metadata in that case.

def compute_ela_image(image_path: str, quality: int = 90, scale: int = 15):
    """
    Returns:
        ela_np: np.ndarray (H, W, 3) uint8 — amplified difference map for visualization
        diff_gray: np.ndarray (H, W) float — raw per-pixel error magnitude (0-255)
    """
    original = Image.open(image_path).convert("RGB")

    buffer = io.BytesIO()
    original.save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    resaved = Image.open(buffer)

    diff = ImageChops.difference(original, resaved)
    diff_np = np.array(diff).astype(np.float32)
    diff_gray = diff_np.mean(axis=2)  # collapse channels

    max_diff = diff_np.max() if diff_np.max() > 0 else 1
    ela_amplified = (diff_np * (255.0 * scale / max_diff)).clip(0, 255).astype(np.uint8)

    return ela_amplified, diff_gray


def find_suspicious_regions(diff_gray: np.ndarray, block_size: int = 16,
                             std_multiplier: float = 2.5, min_area_ratio: float = 0.001):
    """
    Slides a block window over the ELA error map, flags blocks whose mean
    error deviates sharply from the document's overall error distribution,
    then groups them into bounding boxes.

    Returns:
        {
            "suspicious_ratio": float 0-1 (fraction of document area flagged),
            "regions": [ (x, y, w, h), ... ],
            "global_mean": float, "global_std": float
        }
    """
    h, w = diff_gray.shape
    global_mean = float(diff_gray.mean())
    global_std = float(diff_gray.std()) + 1e-6
    threshold = global_mean + std_multiplier * global_std

    mask = np.zeros_like(diff_gray, dtype=np.uint8)
    for y in range(0, h - block_size, block_size):
        for x in range(0, w - block_size, block_size):
            block = diff_gray[y:y + block_size, x:x + block_size]
            if block.mean() > threshold:
                mask[y:y + block_size, x:x + block_size] = 255

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (block_size, block_size))
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions = []
    total_area = h * w
    flagged_area = 0
    min_area = min_area_ratio * total_area

    for c in contours:
        area = cv2.contourArea(c)
        if area >= min_area:
            x, y, bw, bh = cv2.boundingRect(c)
            regions.append((x, y, bw, bh))
            flagged_area += bw * bh

    return {
        "suspicious_ratio": round(flagged_area / total_area, 4),
        "regions": regions,
        "global_mean": round(global_mean, 3),
        "global_std": round(global_std, 3),
    }


def analyze_ela(image_path: str) -> dict:
    """Full ELA pipeline -> tampering score."""
    _, diff_gray = compute_ela_image(image_path)
    result = find_suspicious_regions(diff_gray)

    flags = []
    ratio = result["suspicious_ratio"]
    n_regions = len(result["regions"])

    if n_regions == 0:
        score = 0.0
    else:
        score = min(1.0, ratio * 8 + min(n_regions, 5) * 0.08)
        flags.append(
            f"{n_regions} region(s) show abnormal JPEG error levels "
            f"covering {ratio*100:.2f}% of the document"
        )

    return {
        "score": round(score, 3),
        "suspicious_ratio": ratio,
        "regions": result["regions"],
        "flags": flags,
    }


def save_ela_heatmap(image_path: str, output_path: str, quality: int = 90, scale: int = 15):
    """Utility: save the visual ELA heatmap for human review / UI overlay."""
    ela_np, _ = compute_ela_image(image_path, quality, scale)
    Image.fromarray(ela_np).save(output_path)


# ============================================================
# 3. COPY-MOVE (CLONE) DETECTION
# ============================================================
# Catches a region of the SAME image copied and pasted elsewhere in the
# SAME image — a duplicated stamp/seal, cloned background texture over
# an erased date/number, or a copied signature loop.
#
# Approach: ORB keypoint detection + self-matching. A cluster of strong
# self-matches at a consistent, nontrivial spatial offset is a strong
# tamper signal. This is a fast, dependency-light CPU baseline — swap for
# a trained copy-move CNN (e.g. BusterNet) once you have labeled data.

def detect_copy_move(image_path: str, match_ratio: float = 0.75,
                      min_distance_px: int = 40, min_matches: int = 8) -> dict:
    """
    Returns:
        {
            "score": float 0-1,
            "match_count": int,
            "flags": [str],
            "matched_points": [ ((x1,y1), (x2,y2)), ... ]
        }
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return {"score": 0.0, "match_count": 0, "flags": ["Could not read image"], "matched_points": []}

    orb = cv2.ORB_create(nfeatures=3000)
    keypoints, descriptors = orb.detectAndCompute(img, None)

    if descriptors is None or len(keypoints) < 10:
        return {"score": 0.0, "match_count": 0, "flags": ["Not enough features to analyze"], "matched_points": []}

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(descriptors, descriptors, k=3)

    good_pairs = []
    for m_list in matches:
        for m in m_list[1:]:  # m_list[0] is always the point matched to itself
            if m.distance < match_ratio * 64:  # empirical Hamming threshold for ORB
                p1 = keypoints[m.queryIdx].pt
                p2 = keypoints[m.trainIdx].pt
                dist = np.hypot(p1[0] - p2[0], p1[1] - p2[1])
                if dist > min_distance_px:
                    good_pairs.append((p1, p2))

    seen = set()
    unique_pairs = []
    for p1, p2 in good_pairs:
        key = tuple(sorted([tuple(np.round(p1, 0)), tuple(np.round(p2, 0))]))
        if key not in seen:
            seen.add(key)
            unique_pairs.append((p1, p2))

    match_count = len(unique_pairs)
    flags = []
    score = 0.0

    if match_count >= min_matches:
        offsets = np.array([(p2[0] - p1[0], p2[1] - p1[1]) for p1, p2 in unique_pairs])
        offset_std = np.std(offsets, axis=0).mean()

        if offset_std < 25:  # consistent translation -> strong signal
            flags.append(
                f"{match_count} self-similar regions found with a consistent spatial "
                f"offset — indicates a cloned/pasted region (e.g. duplicated stamp or texture)"
            )
            score = min(1.0, 0.4 + match_count * 0.03)
        else:
            flags.append(
                f"{match_count} repeated patterns found but with inconsistent offsets — "
                f"likely natural repeating texture (e.g. patterned background), lower confidence"
            )
            score = min(0.4, match_count * 0.01)

    return {
        "score": round(score, 3),
        "match_count": match_count,
        "flags": flags,
        "matched_points": [((float(p1[0]), float(p1[1])), (float(p2[0]), float(p2[1])))
                            for p1, p2 in unique_pairs[:50]],
    }


# ============================================================
# 4. TEXT / FIELD CONSISTENCY ANALYSIS
# ============================================================
# Catches Text Manipulation — altered DOB, passport numbers, names,
# expiry dates — by checking whether OCR-located text fields are
# visually consistent with the rest of the document: baseline
# alignment, stroke-width/font consistency, and edge "halo"
# discontinuities typical of "erase and retype" edits.
#
# Expects `field_boxes` from Module 1 (OCR): a list of dicts like
# {"field": "date_of_birth", "box": (x, y, w, h)}.

def _stroke_width_variance(gray_crop: np.ndarray) -> float:
    """Approximate stroke width consistency via distance transform on the
    binarized text mask. High variance = inconsistent font/weight."""
    if gray_crop.size == 0:
        return 0.0
    _, binary = cv2.threshold(gray_crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
    stroke_pixels = dist[dist > 0]
    if len(stroke_pixels) < 10:
        return 0.0
    return float(np.std(stroke_pixels))


def _edge_halo_score(gray: np.ndarray, box: tuple, pad: int = 6) -> float:
    """Checks for a blur/contrast discontinuity ring right around a field
    box — a common artifact of 'erase, patch, retype' edits."""
    x, y, w, h = box
    H, W = gray.shape
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
    outer = gray[y0:y1, x0:x1]
    inner = gray[y:y + h, x:x + w]
    if outer.size == 0 or inner.size == 0:
        return 0.0
    outer_var = float(np.var(outer))
    inner_var = float(np.var(inner))
    if outer_var == 0:
        return 0.0
    return abs(inner_var - outer_var) / (outer_var + 1e-6)


def analyze_text_field(image_path: str, field_boxes: list) -> dict:
    """
    Args:
        image_path: path to the document image
        field_boxes: [{"field": "date_of_birth", "box": (x, y, w, h)}, ...]

    Returns:
        {
            "score": float 0-1,
            "field_results": [{"field", "stroke_variance", "halo_score", "suspicious"}],
            "flags": [str]
        }
    """
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None or not field_boxes:
        return {"score": 0.0, "field_results": [], "flags": ["No image or field boxes provided"]}

    raw_stats = []
    for f in field_boxes:
        x, y, w, h = f["box"]
        crop = gray[y:y + h, x:x + w]
        sv = _stroke_width_variance(crop)
        halo = _edge_halo_score(gray, f["box"])
        raw_stats.append({"field": f["field"], "stroke_variance": sv, "halo_score": halo})

    sv_values = np.array([s["stroke_variance"] for s in raw_stats])
    baseline_mean, baseline_std = sv_values.mean(), sv_values.std() + 1e-6

    field_results = []
    flags = []
    suspicious_count = 0

    for s in raw_stats:
        z = (s["stroke_variance"] - baseline_mean) / baseline_std
        is_suspicious = bool(abs(z) > 2.0 or s["halo_score"] > 0.6)
        if is_suspicious:
            suspicious_count += 1
            flags.append(
                f"Field '{s['field']}' shows font/stroke inconsistency "
                f"(z={z:.2f}) and/or edge halo (halo={s['halo_score']:.2f}) "
                f"relative to the rest of the document"
            )
        field_results.append({**s, "suspicious": is_suspicious})

    score = min(1.0, suspicious_count / max(1, len(field_boxes)) * 1.2)
    return {"score": round(score, 3), "field_results": field_results, "flags": flags}


# ============================================================
# 5. AGGREGATE RISK SCORER
# ============================================================
# Combines all sub-signals into one explainable risk score. Weighted
# sum, NOT a black-box classifier — every number in the output traces
# back to a specific, nameable signal, which matters for a tool where an
# officer needs to justify a decision. Once real labeled data (genuine
# vs. tampered document pairs) is available, swap the weighted sum for a
# small logistic regression / gradient-boosted model trained on these
# same four sub-scores as features — the sub-score functions above don't
# need to change at all.

WEIGHTS = {
    "ela": 0.35,
    "copy_move": 0.30,
    "metadata": 0.15,
    "text_consistency": 0.20,
}

RISK_BANDS = [
    (0.0, 0.25, "LOW"),
    (0.25, 0.55, "MEDIUM"),
    (0.55, 1.01, "HIGH"),
]


def _band_for(score: float) -> str:
    for lo, hi, label in RISK_BANDS:
        if lo <= score < hi:
            return label
    return "HIGH"


def assess_document(image_path: str, field_boxes: list = None) -> dict:
    """
    Runs all tampering sub-checks and returns a single aggregate report.

    Args:
        image_path: path to the document image
        field_boxes: optional OCR field boxes (from Module 1) for text
                     consistency checking. If omitted, that signal is
                     skipped and its weight is redistributed.

    Returns:
        {
            "overall_score": float 0-1,
            "risk_level": "LOW" | "MEDIUM" | "HIGH",
            "sub_scores": {...},
            "weights_used": {...},
            "all_flags": [str, ...],
            "details": {...}   # raw sub-module outputs for the audit trail
        }
    """
    ela_result = analyze_ela(image_path)
    copy_move_result = detect_copy_move(image_path)
    metadata_result = analyze_metadata(image_path)

    weights = dict(WEIGHTS)
    text_result = None
    if field_boxes:
        text_result = analyze_text_field(image_path, field_boxes)
    else:
        dropped = weights.pop("text_consistency")
        total_remaining = sum(weights.values())
        for k in weights:
            weights[k] += dropped * (weights[k] / total_remaining)

    sub_scores = {
        "ela": ela_result["score"],
        "copy_move": copy_move_result["score"],
        "metadata": metadata_result["score"],
    }
    if text_result:
        sub_scores["text_consistency"] = text_result["score"]

    overall = sum(sub_scores[k] * weights[k] for k in sub_scores)
    overall = round(min(overall, 1.0), 3)

    all_flags = []
    all_flags.extend(ela_result["flags"])
    all_flags.extend(copy_move_result["flags"])
    all_flags.extend(metadata_result["flags"])
    if text_result:
        all_flags.extend(text_result["flags"])

    return {
        "overall_score": overall,
        "risk_level": _band_for(overall),
        "sub_scores": sub_scores,
        "weights_used": weights,
        "all_flags": all_flags,
        "details": {
            "ela": ela_result,
            "copy_move": copy_move_result,
            "metadata": metadata_result,
            "text_consistency": text_result,
        },
    }


# ============================================================
# 6. DEMO / CLI ENTRY POINT
# ============================================================
# Run this file directly to get a risk report printed to the console,
# and optionally a side-by-side visualization panel (original / ELA
# heatmap / copy-move overlay) for a presentation slide.

def build_demo_panel(image_path: str, output_path: str):
    """Side-by-side panel: original | ELA heatmap | copy-move overlay."""
    original_bgr = cv2.imread(image_path)
    h, w = original_bgr.shape[:2]

    ela_np, _ = compute_ela_image(image_path)
    ela_bgr = cv2.cvtColor(ela_np, cv2.COLOR_RGB2BGR)
    ela_bgr = cv2.resize(ela_bgr, (w, h))

    cm_result = detect_copy_move(image_path)
    overlay = original_bgr.copy()
    for p1, p2 in cm_result["matched_points"]:
        pt1, pt2 = tuple(map(int, p1)), tuple(map(int, p2))
        cv2.line(overlay, pt1, pt2, (0, 0, 255), 1)
        cv2.circle(overlay, pt1, 3, (0, 255, 0), -1)
        cv2.circle(overlay, pt2, 3, (0, 255, 0), -1)

    def label(img, text):
        img = img.copy()
        cv2.rectangle(img, (0, 0), (w, 25), (0, 0, 0), -1)
        cv2.putText(img, text, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        return img

    panel = np.hstack([
        label(original_bgr, "Original"),
        label(ela_bgr, "ELA Heatmap"),
        label(overlay, f"Copy-Move ({cm_result['match_count']} matches)"),
    ])
    cv2.imwrite(output_path, panel)


def main():
    parser = argparse.ArgumentParser(description="Document tampering risk assessment")
    parser.add_argument("image_path", help="Path to the document image")
    parser.add_argument("--panel", help="Optional path to save a visual demo panel (PNG)")
    args = parser.parse_args()

    report = assess_document(args.image_path)
    print(json.dumps({k: v for k, v in report.items() if k != "details"}, indent=2))

    if args.panel:
        build_demo_panel(args.image_path, args.panel)
        print(f"\nDemo panel saved to {args.panel}")


if __name__ == "__main__":
    main()



