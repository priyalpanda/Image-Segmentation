import argparse
import json
import subprocess
import sys
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

SCRIPT_DIR = Path(__file__).resolve().parent

from evaluation_report import evaluate_predictions, save_evaluation_reports


def run_preprocess(input_path: Path, output_path: Path) -> None:
    cmd = [sys.executable, str(SCRIPT_DIR / "preprocess.py"), "--input", str(input_path), "--output", str(output_path)]
    subprocess.run(cmd, check=True)


def run_segment(
    input_path: Path,
    images_output: Path,
    labels_output: Path,
    model_name: str = "maskrcnn",
    yolo_weights: Optional[str] = None,
) -> Dict[str, Any]:
    """Run segmentation and return results data."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("segment", SCRIPT_DIR / "segment.py")
    segment_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(segment_module)

    return segment_module.segment_folder(
        input_path=input_path,
        images_output=images_output,
        labels_output=labels_output,
        model_name=model_name,
        yolo_weights=yolo_weights,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Full image/instance segmentation pipeline.")
    parser.add_argument("--input", required=True, help="Input image file or folder")
    parser.add_argument("--workdir", default="pipeline_output", help="Working directory for outputs")
    parser.add_argument("--results-dir", default="results", help="Final results directory")
    parser.add_argument(
        "--model",
        default="maskrcnn",
        help="Segmentation model to run: maskrcnn, yolo, or path to YOLO weights",
    )
    parser.add_argument(
        "--yolo-weights",
        default=None,
        help="Optional path to a YOLO segmentation weight file (e.g. yolov11-seg.pt)",
    )
    parser.add_argument(
        "--groundtruth-labels",
        default=None,
        help="Folder containing YOLO-format groundtruth .txt labels for evaluation",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold used to decide true positives",
    )
    parser.add_argument(
        "--poor-localization-min",
        type=float,
        default=0.1,
        help="Minimum IoU for poor localization when detection exists but is weak",
    )

    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    workdir = Path(args.workdir).expanduser().resolve()
    results_dir = Path(args.results_dir).expanduser().resolve()
    model_arg = args.model
    model_name = model_arg.lower()
    yolo_weights = args.yolo_weights
    groundtruth_labels = args.groundtruth_labels

    if Path(model_arg).suffix.lower() in {".pt", ".pth", ".onnx"} and Path(model_arg).exists():
        yolo_weights = model_arg
        model_name = "yolo"
        output_folder_name = Path(model_arg).stem
    else:
        output_folder_name = Path(yolo_weights).stem if yolo_weights else model_name

    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    workdir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    preprocessed_path = workdir / "_preprocessed_temp"
    print("[1/3] Preprocessing images...")
    run_preprocess(input_path, preprocessed_path)

    model_results_dir = results_dir / output_folder_name
    print(f"[2/3] Running segmentation using {model_name}...")
    run_segment(
        input_path=preprocessed_path,
        images_output=model_results_dir / "images",
        labels_output=model_results_dir / "labels",
        model_name=model_name,
        yolo_weights=yolo_weights,
    )

    evaluation_report = None
    if groundtruth_labels:
        groundtruth_path = Path(groundtruth_labels).expanduser().resolve()
        if not groundtruth_path.exists():
            raise FileNotFoundError(f"Groundtruth labels path not found: {groundtruth_path}")
        print(f"[3/4] Evaluating predictions for {output_folder_name}...")
        evaluation_report = evaluate_predictions(
            predicted_labels_dir=model_results_dir / "labels",
            groundtruth_dir=groundtruth_path,
            images_dir=input_path,
            iou_threshold=args.iou_threshold,
            poor_localization_min=args.poor_localization_min,
        )
        json_path, txt_path = save_evaluation_reports(
            evaluation_report,
            model_results_dir,
            output_folder_name,
        )
        print(f"Saved evaluation report: {json_path}")
        print(f"Saved evaluation summary: {txt_path}")
        try:
            from observation_report import build_observation_report, save_observation_reports

            with open(json_path, "r", encoding="utf-8") as f:
                evaluation_document = json.load(f)
            observation = build_observation_report(
                evaluation_document,
                predicted_labels_dir=model_results_dir / "labels",
                groundtruth_dir=groundtruth_path,
                results_dir=model_results_dir,
            )
            obs_json, obs_html = save_observation_reports(
                observation, model_results_dir, evaluation_document.get("summary")
            )
            print(f"Saved observation report: {obs_json}")
            print(f"Saved observation HTML: {obs_html}")
        except Exception as exc:
            print(f"Observation report skipped: {exc}")

    if preprocessed_path.exists():
        shutil.rmtree(preprocessed_path)

    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)
    print(f"Model: {output_folder_name}")
    print(f"Output folder: {model_results_dir}")
    print(f"Segmented images: {model_results_dir / 'images'}")
    print(f"Segmentation labels: {model_results_dir / 'labels'}")
    if evaluation_report is not None:
        print(f"Evaluation JSON: {model_results_dir / 'evaluation.json'}")
        print(f"Evaluation summary: {model_results_dir / 'evaluation.txt'}")
        print(f"False negatives: {evaluation_report['total_false_negatives']}")
        print(f"False positives: {evaluation_report['total_false_positives']}")
        print(f"Under segmentation: {evaluation_report['total_under_segmentation']}")
        print(f"Over segmentation: {evaluation_report['total_over_segmentation']}")
        print(f"Average IoU: {evaluation_report['average_iou_across_images']:.4f}")
        print(f"Average Dice: {evaluation_report['average_dice_across_images']:.4f}")
        print(f"Classification accuracy: {evaluation_report['classification_accuracy']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
