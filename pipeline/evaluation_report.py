"""Compare predicted polygon labels against ground truth and emit per-model evaluation reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from skimage.draw import polygon as skpolygon

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def load_image_size(image_path: Path) -> Tuple[int, int]:
    from PIL import Image

    with Image.open(image_path) as img:
        return img.width, img.height


def find_image_by_stem(images_dir: Path, stem: str) -> Optional[Path]:
    if images_dir.is_file() and images_dir.stem == stem:
        return images_dir
    if images_dir.is_dir():
        for ext in SUPPORTED_EXTENSIONS:
            candidate = images_dir / f"{stem}{ext}"
            if candidate.exists():
                return candidate
    return None


def image_filename_for_stem(images_dir: Path, stem: str) -> str:
    found = find_image_by_stem(images_dir, stem)
    return found.name if found else f"{stem}.jpg"


def load_polygon_labels(label_path: Path) -> List[Dict[str, Any]]:
    instances: List[Dict[str, Any]] = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7 or (len(parts) - 1) % 2 != 0:
                continue
            label_id = int(parts[0])
            polygon = [float(x) for x in parts[1:]]
            instances.append({"label_id": label_id, "polygon": polygon})
    return instances


def load_prediction_metadata(label_path: Path) -> List[Optional[float]]:
    """Optional per-instance confidence from a sidecar JSON next to the label txt."""
    json_path = label_path.with_suffix(".json")
    if not json_path.exists():
        return []
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    instances = data.get("instances", data) if isinstance(data, dict) else data
    if not isinstance(instances, list):
        return []
    confidences: List[Optional[float]] = []
    for inst in instances:
        conf = inst.get("confidence") if isinstance(inst, dict) else None
        confidences.append(float(conf) if conf is not None else None)
    return confidences


def polygon_to_mask(polygon: List[float], image_size: Tuple[int, int]) -> np.ndarray:
    width, height = image_size
    xs = np.asarray(polygon[0::2], dtype=np.float32) * width
    ys = np.asarray(polygon[1::2], dtype=np.float32) * height
    if xs.size < 3 or ys.size < 3:
        return np.zeros((height, width), dtype=np.uint8)
    rr, cc = skpolygon(ys, xs, shape=(height, width))
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[rr, cc] = 1
    return mask


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(intersection) / float(union) if union > 0 else 0.0


def mask_dice(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = np.logical_and(mask_a, mask_b).sum()
    total = mask_a.sum() + mask_b.sum()
    return float(2 * intersection) / float(total) if total > 0 else 0.0


def merge_instance_masks(masks: List[np.ndarray], image_size: Tuple[int, int]) -> np.ndarray:
    height, width = image_size[1], image_size[0]
    merged = np.zeros((height, width), dtype=np.uint8)
    for mask in masks:
        merged = np.logical_or(merged, mask).astype(np.uint8)
    return merged


def dilate_mask(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    from skimage.morphology import dilation, disk

    return dilation(mask.astype(bool), disk(radius))


def mask_boundary(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    mask_bool = mask.astype(bool)
    dilated = dilate_mask(mask_bool, radius=radius)
    return np.logical_and(dilated, np.logical_not(mask_bool))


def boundary_error(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Symmetric mask disagreement normalized by combined boundary length."""
    xor_area = np.logical_xor(mask_a, mask_b).sum()
    boundary_a = mask_boundary(mask_a)
    boundary_b = mask_boundary(mask_b)
    perimeter = boundary_a.sum() + boundary_b.sum()
    if perimeter == 0:
        return 0.0
    return float(xor_area) / float(perimeter)


def count_duplicate_masks(pred_masks: List[np.ndarray], duplicate_iou: float = 0.75) -> int:
    duplicates = 0
    for i in range(len(pred_masks)):
        for j in range(i + 1, len(pred_masks)):
            if mask_iou(pred_masks[i], pred_masks[j]) >= duplicate_iou:
                duplicates += 1
    return duplicates


def evaluate_image(
    gt_instances: List[Dict[str, Any]],
    pred_instances: List[Dict[str, Any]],
    image_size: Tuple[int, int],
    pred_confidences: Optional[List[Optional[float]]] = None,
    iou_threshold: float = 0.5,
    poor_localization_min: float = 0.1,
    duplicate_iou: float = 0.75,
) -> Dict[str, Any]:
    if not gt_instances and not pred_instances:
        return _empty_image_eval()

    gt_masks = [polygon_to_mask(gt["polygon"], image_size) for gt in gt_instances]
    pred_masks = [polygon_to_mask(pred["polygon"], image_size) for pred in pred_instances]

    iou_matrix: List[List[float]] = []
    class_matrix: List[List[bool]] = []
    for gt_idx, gt in enumerate(gt_instances):
        row: List[float] = []
        class_row: List[bool] = []
        for pred_idx, pred in enumerate(pred_instances):
            same_class = gt["label_id"] == pred["label_id"]
            class_row.append(same_class)
            row.append(mask_iou(gt_masks[gt_idx], pred_masks[pred_idx]) if same_class else 0.0)
        iou_matrix.append(row)
        class_matrix.append(class_row)

    gt_best = [max(row) if row else 0.0 for row in iou_matrix]
    pred_best = [
        max((iou_matrix[r][c] for r in range(len(gt_instances))), default=0.0)
        for c in range(len(pred_instances))
    ]

    false_negatives = sum(1 for val in gt_best if val < iou_threshold)
    false_positives = sum(1 for val in pred_best if val < iou_threshold)
    under_segmentation = sum(
        1
        for c in range(len(pred_instances))
        if sum(1 for r in range(len(gt_instances)) if iou_matrix[r][c] >= iou_threshold) > 1
    )
    over_segmentation = sum(
        1
        for r in range(len(gt_instances))
        if sum(1 for c in range(len(pred_instances)) if iou_matrix[r][c] >= iou_threshold) > 1
    )

    matches: List[Dict[str, Any]] = []
    matched_gt = set()
    matched_pred = set()
    possible_matches = [
        (iou_matrix[r][c], r, c)
        for r in range(len(gt_instances))
        for c in range(len(pred_instances))
        if iou_matrix[r][c] >= iou_threshold
    ]
    possible_matches.sort(reverse=True)
    for iou_val, gt_idx, pred_idx in possible_matches:
        if gt_idx in matched_gt or pred_idx in matched_pred:
            continue
        matched_gt.add(gt_idx)
        matched_pred.add(pred_idx)
        matches.append(
            {
                "gt_index": gt_idx,
                "pred_index": pred_idx,
                "iou": iou_val,
                "dice": mask_dice(gt_masks[gt_idx], pred_masks[pred_idx]),
                "boundary_error": max(
                    0.0, 1.0 - mask_dice(gt_masks[gt_idx], pred_masks[pred_idx])
                ),
                "label_id": gt_instances[gt_idx]["label_id"],
            }
        )

    average_iou = float(sum(match["iou"] for match in matches) / len(matches)) if matches else 0.0
    average_dice = float(sum(match["dice"] for match in matches) / len(matches)) if matches else 0.0
    match_boundary_errors = [
        max(0.0, 1.0 - match["dice"]) for match in matches
    ]
    average_boundary_error = (
        float(sum(match_boundary_errors) / len(match_boundary_errors))
        if match_boundary_errors
        else 0.0
    )

    gt_union = merge_instance_masks(gt_masks, image_size)
    pred_union = merge_instance_masks(pred_masks, image_size)
    match_pixel_accuracies: List[float] = []
    for gt_idx, pred_idx in (
        (match["gt_index"], match["pred_index"]) for match in matches
    ):
        gt_mask = gt_masks[gt_idx]
        pred_mask = pred_masks[pred_idx]
        foreground = np.logical_or(gt_mask, pred_mask)
        if foreground.any():
            match_pixel_accuracies.append(
                float(np.sum(gt_mask[foreground] == pred_mask[foreground])) / float(foreground.sum())
            )
    if match_pixel_accuracies:
        pixel_accuracy = float(sum(match_pixel_accuracies) / len(match_pixel_accuracies))
    else:
        foreground = np.logical_or(gt_union, pred_union)
        if foreground.any():
            pixel_accuracy = float(np.sum(gt_union[foreground] == pred_union[foreground])) / float(
                foreground.sum()
            )
        else:
            pixel_accuracy = 1.0
    union_dice = mask_dice(gt_union, pred_union)
    union_iou = mask_iou(gt_union, pred_union)
    union_boundary = max(0.0, 1.0 - union_dice)

    matched_confidences: List[float] = []
    if pred_confidences:
        for match in matches:
            pred_idx = match["pred_index"]
            if pred_idx < len(pred_confidences) and pred_confidences[pred_idx] is not None:
                matched_confidences.append(float(pred_confidences[pred_idx]))

    classification_correct = (
        false_negatives == 0
        and false_positives == 0
        and len(matches) == len(gt_instances) == len(pred_instances)
        and all(class_matrix[match["gt_index"]][match["pred_index"]] for match in matches)
    )

    return {
        "gt_count": len(gt_instances),
        "pred_count": len(pred_instances),
        "true_positives": len(matches),
        "false_negatives": false_negatives,
        "false_positives": false_positives,
        "under_segmentation": under_segmentation,
        "over_segmentation": over_segmentation,
        "duplicate_masks": count_duplicate_masks(pred_masks, duplicate_iou=duplicate_iou),
        "average_iou": average_iou,
        "union_iou": union_iou,
        "dice": average_dice if matches else union_dice,
        "union_dice": union_dice,
        "pixel_accuracy": pixel_accuracy,
        "boundary_error": average_boundary_error if matches else union_boundary,
        "classification_correct": classification_correct,
        "confidence": (
            float(sum(matched_confidences) / len(matched_confidences)) if matched_confidences else None
        ),
        "matches": matches,
    }


def _empty_image_eval() -> Dict[str, Any]:
    return {
        "gt_count": 0,
        "pred_count": 0,
        "true_positives": 0,
        "false_negatives": 0,
        "false_positives": 0,
        "under_segmentation": 0,
        "over_segmentation": 0,
        "duplicate_masks": 0,
        "average_iou": 0.0,
        "union_iou": 0.0,
        "dice": 0.0,
        "union_dice": 0.0,
        "pixel_accuracy": 1.0,
        "boundary_error": 0.0,
        "classification_correct": True,
        "confidence": None,
        "matches": [],
    }


def format_image_record(stem: str, images_dir: Path, image_eval: Dict[str, Any]) -> Dict[str, Any]:
    record = {
        "image": image_filename_for_stem(images_dir, stem),
        "gt_count": int(image_eval["gt_count"]),
        "pred_count": int(image_eval["pred_count"]),
        "iou": round(float(image_eval["average_iou"]), 4),
        "dice": round(float(image_eval["dice"]), 4),
        "pixel_accuracy": round(float(image_eval["pixel_accuracy"]), 4),
        "false_negative": int(image_eval["false_negatives"]),
        "false_positive": int(image_eval["false_positives"]),
        "under_segmentation": int(image_eval["under_segmentation"]),
        "over_segmentation": int(image_eval["over_segmentation"]),
        "boundary_error": round(float(image_eval["boundary_error"]), 4),
        "duplicate_masks": int(image_eval["duplicate_masks"]),
        "classification_correct": bool(image_eval["classification_correct"]),
    }
    if image_eval["confidence"] is not None:
        record["confidence"] = round(float(image_eval["confidence"]), 4)
    return record


def evaluate_predictions(
    predicted_labels_dir: Path,
    groundtruth_dir: Path,
    images_dir: Path,
    iou_threshold: float = 0.5,
    poor_localization_min: float = 0.1,
    duplicate_iou: float = 0.75,
    include_missing: bool = True,
) -> Dict[str, Any]:
    predicted_files = {p.stem: p for p in predicted_labels_dir.glob("*.txt")}
    gt_files = {p.stem: p for p in Path(groundtruth_dir).rglob("*.txt")}
    common_stems = sorted(set(predicted_files) & set(gt_files))
    missing_groundtruth = sorted(set(predicted_files) - set(gt_files))
    missing_predictions = sorted(set(gt_files) - set(predicted_files))

    candidate_stems = list(common_stems)
    if include_missing:
        candidate_stems.extend(missing_predictions)
        candidate_stems.extend(missing_groundtruth)
    stems_to_evaluate = sorted(
        {stem for stem in set(candidate_stems) if find_image_by_stem(images_dir, stem) is not None}
    )
    skipped_stems = sorted(set(candidate_stems) - set(stems_to_evaluate))

    summary = {
        "total_images": len(stems_to_evaluate),
        "total_gt_instances": 0,
        "total_pred_instances": 0,
        "total_true_positives": 0,
        "total_false_negatives": 0,
        "total_false_positives": 0,
        "total_under_segmentation": 0,
        "total_over_segmentation": 0,
        "total_duplicate_masks": 0,
        "total_average_iou": 0.0,
        "total_dice": 0.0,
        "total_pixel_accuracy": 0.0,
        "total_boundary_error": 0.0,
        "images_evaluated_with_gt": len(common_stems) + len(missing_predictions),
        "images_with_predictions_only": len(missing_groundtruth),
        "classification_correct_images": 0,
        "images": [],
        "missing_groundtruth": missing_groundtruth,
        "missing_predictions": missing_predictions,
        "skipped_no_image": skipped_stems,
    }

    for stem in stems_to_evaluate:
        has_gt = stem in gt_files
        has_pred = stem in predicted_files

        if has_gt and has_pred:
            image_path = find_image_by_stem(images_dir, stem)
            if image_path is None:
                raise FileNotFoundError(f"Cannot find image file for stem {stem} in {images_dir}")
            image_size = load_image_size(image_path)
            gt_instances = load_polygon_labels(gt_files[stem])
            pred_instances = load_polygon_labels(predicted_files[stem])
            pred_confidences = load_prediction_metadata(predicted_files[stem])
            image_eval = evaluate_image(
                gt_instances,
                pred_instances,
                image_size,
                pred_confidences=pred_confidences or None,
                iou_threshold=iou_threshold,
                poor_localization_min=poor_localization_min,
                duplicate_iou=duplicate_iou,
            )
        elif has_gt:
            image_path = find_image_by_stem(images_dir, stem)
            if image_path is None:
                raise FileNotFoundError(f"Cannot find image file for stem {stem} in {images_dir}")
            image_size = load_image_size(image_path)
            gt_instances = load_polygon_labels(gt_files[stem])
            image_eval = evaluate_image(gt_instances, [], image_size, iou_threshold=iou_threshold)
            image_eval["false_negatives"] = image_eval["gt_count"]
            image_eval["classification_correct"] = False
        else:
            image_path = find_image_by_stem(images_dir, stem)
            if image_path is None:
                raise FileNotFoundError(f"Cannot find image file for stem {stem} in {images_dir}")
            image_size = load_image_size(image_path)
            pred_instances = load_polygon_labels(predicted_files[stem])
            pred_confidences = load_prediction_metadata(predicted_files[stem])
            image_eval = evaluate_image(
                [],
                pred_instances,
                image_size,
                pred_confidences=pred_confidences or None,
                iou_threshold=iou_threshold,
            )
            image_eval["false_positives"] = image_eval["pred_count"]
            image_eval["classification_correct"] = False

        _accumulate_summary(summary, image_eval)
        summary["images"].append(format_image_record(stem, images_dir, image_eval))

    evaluated_count = summary["total_images"] or 1
    summary["average_iou_across_images"] = float(summary["total_average_iou"] / evaluated_count)
    summary["average_dice_across_images"] = float(summary["total_dice"] / evaluated_count)
    summary["average_pixel_accuracy"] = float(summary["total_pixel_accuracy"] / evaluated_count)
    summary["average_boundary_error"] = float(summary["total_boundary_error"] / evaluated_count)
    summary["classification_accuracy"] = float(
        summary["classification_correct_images"] / evaluated_count
    )
    return summary


def _accumulate_summary(summary: Dict[str, Any], image_eval: Dict[str, Any]) -> None:
    summary["total_gt_instances"] += image_eval["gt_count"]
    summary["total_pred_instances"] += image_eval["pred_count"]
    summary["total_true_positives"] += image_eval["true_positives"]
    summary["total_false_negatives"] += image_eval["false_negatives"]
    summary["total_false_positives"] += image_eval["false_positives"]
    summary["total_under_segmentation"] += image_eval["under_segmentation"]
    summary["total_over_segmentation"] += image_eval["over_segmentation"]
    summary["total_duplicate_masks"] += image_eval["duplicate_masks"]
    summary["total_average_iou"] += image_eval["average_iou"]
    summary["total_dice"] += image_eval["dice"]
    summary["total_pixel_accuracy"] += image_eval["pixel_accuracy"]
    summary["total_boundary_error"] += image_eval["boundary_error"]
    if image_eval["classification_correct"]:
        summary["classification_correct_images"] += 1


def build_evaluation_document(model_name: str, summary: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "model": model_name,
        "summary": {
            "total_images": summary["total_images"],
            "total_gt_instances": summary["total_gt_instances"],
            "total_pred_instances": summary["total_pred_instances"],
            "true_positives": summary["total_true_positives"],
            "false_negatives": summary["total_false_negatives"],
            "false_positives": summary["total_false_positives"],
            "under_segmentation": summary["total_under_segmentation"],
            "over_segmentation": summary["total_over_segmentation"],
            "duplicate_masks": summary["total_duplicate_masks"],
            "average_iou": round(summary["average_iou_across_images"], 4),
            "average_dice": round(summary["average_dice_across_images"], 4),
            "average_pixel_accuracy": round(summary["average_pixel_accuracy"], 4),
            "average_boundary_error": round(summary["average_boundary_error"], 4),
            "classification_accuracy": round(summary["classification_accuracy"], 4),
            "missing_groundtruth_count": len(summary["missing_groundtruth"]),
            "missing_predictions_count": len(summary["missing_predictions"]),
        },
        "images": summary["images"],
        "missing_groundtruth": summary["missing_groundtruth"],
        "missing_predictions": summary["missing_predictions"],
        "skipped_no_image": summary.get("skipped_no_image", []),
    }


def format_evaluation_report(summary: Dict[str, Any]) -> str:
    lines = [
        f"Total images evaluated: {summary['total_images']}",
        f"Total groundtruth instances: {summary['total_gt_instances']}",
        f"Total predicted instances: {summary['total_pred_instances']}",
        f"True positives: {summary['total_true_positives']}",
        f"False negatives: {summary['total_false_negatives']}",
        f"False positives: {summary['total_false_positives']}",
        f"Under segmentation: {summary['total_under_segmentation']}",
        f"Over segmentation: {summary['total_over_segmentation']}",
        f"Duplicate masks: {summary['total_duplicate_masks']}",
        f"Average IoU across images: {summary['average_iou_across_images']:.4f}",
        f"Average Dice across images: {summary['average_dice_across_images']:.4f}",
        f"Average pixel accuracy: {summary['average_pixel_accuracy']:.4f}",
        f"Average boundary error: {summary['average_boundary_error']:.4f}",
        f"Classification accuracy (images): {summary['classification_accuracy']:.4f}",
        "",
        "Image-level results:",
    ]
    for image_record in summary["images"]:
        lines.append(
            f"{image_record['image']}: iou={image_record['iou']:.4f} dice={image_record['dice']:.4f} "
            f"px_acc={image_record['pixel_accuracy']:.4f} fn={image_record['false_negative']} "
            f"fp={image_record['false_positive']} under={image_record['under_segmentation']} "
            f"over={image_record['over_segmentation']} boundary={image_record['boundary_error']:.4f} "
            f"class_ok={image_record['classification_correct']}"
        )
    if summary["missing_groundtruth"]:
        lines.append("")
        lines.append("Predicted labels with no GT:")
        lines.extend(summary["missing_groundtruth"])
    if summary["missing_predictions"]:
        lines.append("")
        lines.append("GT labels with no predictions:")
        lines.extend(summary["missing_predictions"])
    return "\n".join(lines)


def save_evaluation_reports(
    summary: Dict[str, Any],
    output_dir: Path,
    model_name: str,
) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "evaluation.json"
    txt_path = output_dir / "evaluation.txt"
    document = build_evaluation_document(model_name, summary)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(document, f, indent=2)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(format_evaluation_report(summary))
    return json_path, txt_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate predicted polygon labels against ground truth labels."
    )
    parser.add_argument("--predicted-labels", required=True, help="Folder with generated .txt labels")
    parser.add_argument("--groundtruth-labels", required=True, help="Folder with ground truth .txt labels")
    parser.add_argument("--images", required=True, help="Folder with source images")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Model results folder where evaluation.json and evaluation.txt are written",
    )
    parser.add_argument("--model-name", default=None, help="Model name stored in evaluation.json")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--duplicate-iou", type=float, default=0.75)
    args = parser.parse_args()

    predicted_labels = Path(args.predicted_labels).expanduser().resolve()
    groundtruth_labels = Path(args.groundtruth_labels).expanduser().resolve()
    images_dir = Path(args.images).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    model_name = args.model_name or output_dir.name

    summary = evaluate_predictions(
        predicted_labels_dir=predicted_labels,
        groundtruth_dir=groundtruth_labels,
        images_dir=images_dir,
        iou_threshold=args.iou_threshold,
        duplicate_iou=args.duplicate_iou,
    )
    json_path, txt_path = save_evaluation_reports(summary, output_dir, model_name)
    print(f"Saved evaluation report: {json_path}")
    print(f"Saved evaluation summary: {txt_path}")
    print(f"Images evaluated: {summary['total_images']}")
    print(f"Average IoU: {summary['average_iou_across_images']:.4f}")

    try:
        from observation_report import build_observation_report, save_observation_reports

        observation = build_observation_report(
            build_evaluation_document(model_name, summary),
            predicted_labels_dir=predicted_labels,
            groundtruth_dir=groundtruth_labels,
            results_dir=output_dir,
        )
        obs_json, obs_html = save_observation_reports(observation, output_dir, observation.get("summary"))
        print(f"Saved observation report: {obs_json}")
        print(f"Saved observation HTML: {obs_html}")
    except Exception as exc:
        print(f"Observation report skipped: {exc}")


if __name__ == "__main__":
    main()
