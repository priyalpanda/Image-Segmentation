import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS


def find_image_by_stem(images_dir: Path, stem: str) -> Path:
    for ext in SUPPORTED_EXTENSIONS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Cannot find image file for stem {stem} in {images_dir}")


def parse_polygon_label_file(path: Path) -> List[Dict[str, Any]]:
    instances: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            label_id = int(parts[0])
            coords = [float(x) for x in parts[1:]]
            if len(coords) < 6:
                # fallback for bbox label in x_center y_center w h format
                if len(coords) == 4:
                    x1, y1, x2, y2 = coords
                    coords = [x1, y1, x2, y1, x2, y2, x1, y2]
                else:
                    continue
            instances.append({"label_id": label_id, "coords": coords})
    return instances


def polygon_to_mask(coords: List[float], width: int, height: int) -> np.ndarray:
    pts = np.array(coords, dtype=np.float32).reshape(-1, 2)
    if pts.size == 0:
        return np.zeros((height, width), dtype=np.uint8)

    if pts.max() <= 1.0:
        pts[:, 0] *= width
        pts[:, 1] *= height

    pts_int = np.clip(np.round(pts), 0, np.array([width - 1, height - 1])).astype(np.int32)
    if len(pts_int) < 3:
        return np.zeros((height, width), dtype=np.uint8)

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [pts_int], 1)
    return mask


def compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    if mask_a.shape != mask_b.shape:
        raise ValueError("Mask shapes must match to compute IoU")
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(intersection / union) if union > 0 else 0.0


def evaluate_instance_masks(
    gt_masks: List[np.ndarray], pred_masks: List[np.ndarray], iou_threshold: float = 0.5
) -> Dict[str, Any]:
    if not gt_masks and not pred_masks:
        return {
            "gt_count": 0,
            "pred_count": 0,
            "false_negatives": 0,
            "false_positives": 0,
            "matches": [],
            "average_iou": 0.0,
        }

    iou_matrix = [[compute_iou(gt, pred) for pred in pred_masks] for gt in gt_masks]

    gt_best = [max(row) if row else 0.0 for row in iou_matrix]
    pred_best = [max((iou_matrix[r][c] for r in range(len(gt_masks))), default=0.0) for c in range(len(pred_masks))]

    false_negatives = sum(1 for iou in gt_best if iou < iou_threshold)
    false_positives = sum(1 for iou in pred_best if iou < iou_threshold)
    matched_ious = [iou for iou in gt_best if iou >= iou_threshold]
    average_iou = float(sum(matched_ious) / len(matched_ious)) if matched_ious else 0.0

    matches = []
    for gt_idx, row in enumerate(iou_matrix):
        best_pred = None
        best_iou = 0.0
        for pred_idx, iou_value in enumerate(row):
            if iou_value > best_iou:
                best_iou = iou_value
                best_pred = pred_idx
        matches.append({"gt_index": gt_idx, "pred_index": best_pred, "iou": best_iou})

    return {
        "gt_count": len(gt_masks),
        "pred_count": len(pred_masks),
        "false_negatives": false_negatives,
        "false_positives": false_positives,
        "matches": matches,
        "average_iou": average_iou,
    }


def evaluate_image_polygons(
    gt_instances: List[Dict[str, Any]],
    pred_instances: List[Dict[str, Any]],
    image_size: Tuple[int, int],
    iou_threshold: float = 0.5,
) -> Dict[str, Any]:
    width, height = image_size
    gt_masks = [polygon_to_mask(inst["coords"], width, height) for inst in gt_instances]
    pred_masks = [polygon_to_mask(inst["coords"], width, height) for inst in pred_instances]
    out = evaluate_instance_masks(gt_masks, pred_masks, iou_threshold=iou_threshold)
    out["gt_instances"] = gt_instances
    out["pred_instances"] = pred_instances
    return out


def evaluate_polygon_predictions(
    predicted_dir: Path,
    groundtruth_dir: Path,
    images_dir: Path,
    iou_threshold: float = 0.5,
) -> Dict[str, Any]:
    predicted_files = {p.stem: p for p in predicted_dir.glob("*.txt")}
    groundtruth_files = {p.stem: p for p in groundtruth_dir.glob("*.txt")}
    stems = sorted(set(predicted_files) & set(groundtruth_files))

    summary = {
        "total_images": len(stems),
        "total_gt_instances": 0,
        "total_pred_instances": 0,
        "total_false_negatives": 0,
        "total_false_positives": 0,
        "total_average_iou": 0.0,
        "images": [],
        "missing_groundtruth": sorted(set(predicted_files) - set(groundtruth_files)),
        "missing_predictions": sorted(set(groundtruth_files) - set(predicted_files)),
    }

    for stem in stems:
        image_path = find_image_by_stem(images_dir, stem)
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Unable to read image {image_path}")
        height, width = image.shape[:2]

        gt_instances = parse_polygon_label_file(groundtruth_files[stem])
        pred_instances = parse_polygon_label_file(predicted_files[stem])

        image_eval = evaluate_image_polygons(
            gt_instances,
            pred_instances,
            (width, height),
            iou_threshold=iou_threshold,
        )

        summary["total_gt_instances"] += image_eval["gt_count"]
        summary["total_pred_instances"] += image_eval["pred_count"]
        summary["total_false_negatives"] += image_eval["false_negatives"]
        summary["total_false_positives"] += image_eval["false_positives"]
        summary["total_average_iou"] += image_eval["average_iou"]

        summary["images"].append({"image": stem, **image_eval})

    summary["average_iou_across_images"] = (
        float(summary["total_average_iou"] / summary["total_images"]) if summary["total_images"] else 0.0
    )
    summary["false_negative_rate"] = (
        float(summary["total_false_negatives"]) / summary["total_gt_instances"] if summary["total_gt_instances"] else 0.0
    )
    summary["false_positive_rate"] = (
        float(summary["total_false_positives"]) / summary["total_pred_instances"] if summary["total_pred_instances"] else 0.0
    )
    summary["error_rate"] = (
        float(summary["total_false_negatives"] + summary["total_false_positives"]) / summary["total_gt_instances"]
        if summary["total_gt_instances"]
        else 0.0
    )
    return summary


def save_summary_files(summary: Dict[str, Any], output_dir: Path, model_name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{model_name}_polygon_evaluation.json"
    csv_path = output_dir / f"{model_name}_polygon_evaluation.csv"
    txt_path = output_dir / f"{model_name}_polygon_evaluation.txt"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Total images: {summary['total_images']}\n")
        f.write(f"Total GT instances: {summary['total_gt_instances']}\n")
        f.write(f"Total predicted instances: {summary['total_pred_instances']}\n")
        f.write(f"Total false negatives: {summary['total_false_negatives']}\n")
        f.write(f"Total false positives: {summary['total_false_positives']}\n")
        f.write(f"Average IoU across images: {summary['average_iou_across_images']:.4f}\n")
        f.write(f"False negative rate: {summary['false_negative_rate']:.4f}\n")
        f.write(f"False positive rate: {summary['false_positive_rate']:.4f}\n")
        f.write(f"Error rate: {summary['error_rate']:.4f}\n")
        f.write(f"Missing groundtruth files: {summary['missing_groundtruth']}\n")
        f.write(f"Missing prediction files: {summary['missing_predictions']}\n")

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image",
            "gt_count",
            "pred_count",
            "false_negatives",
            "false_positives",
            "average_iou",
        ])
        for image_data in summary["images"]:
            writer.writerow([
                image_data["image"],
                image_data["gt_count"],
                image_data["pred_count"],
                image_data["false_negatives"],
                image_data["false_positives"],
                f"{image_data['average_iou']:.4f}",
            ])


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate polygon label predictions against groundtruth.")
    parser.add_argument("--predicted-labels", required=True, help="Folder containing predicted polygon .txt labels")
    parser.add_argument("--groundtruth-labels", required=True, help="Folder containing groundtruth polygon .txt labels")
    parser.add_argument("--images", required=True, help="Folder containing source images")
    parser.add_argument("--output-dir", default="results", help="Directory to save polygon evaluation reports")
    parser.add_argument("--model-name", default="model", help="Name of the model for report filenames")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="IoU threshold for matches")
    args = parser.parse_args()

    predicted_labels = Path(args.predicted_labels).expanduser().resolve()
    groundtruth_labels = Path(args.groundtruth_labels).expanduser().resolve()
    images_dir = Path(args.images).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not predicted_labels.exists() or not groundtruth_labels.exists() or not images_dir.exists():
        raise FileNotFoundError("One or more input directories do not exist")

    summary = evaluate_polygon_predictions(
        predicted_dir=predicted_labels,
        groundtruth_dir=groundtruth_labels,
        images_dir=images_dir,
        iou_threshold=args.iou_threshold,
    )
    save_summary_files(summary, output_dir, args.model_name)
    print(f"Saved polygon evaluation reports to {output_dir}")


if __name__ == "__main__":
    main()
