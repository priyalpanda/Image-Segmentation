from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from skimage import measure
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights, maskrcnn_resnet50_fpn

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS


def get_image_paths(input_path: Path) -> List[Path]:
    if input_path.is_dir():
        return sorted([p for p in input_path.iterdir() if is_image_file(p)])
    if is_image_file(input_path):
        return [input_path]
    return []


def create_colored_overlay(image: Image.Image, masks: List[np.ndarray]) -> Image.Image:
    """Create overlay with all instances colored (like reference image)."""
    overlay_array = np.array(image, dtype=np.uint8)
    colors = [
        (255, 0, 0),
        (0, 0, 255),
        (0, 255, 255),
        (0, 255, 0),
        (255, 255, 0),
        (255, 0, 255),
        (255, 128, 0),
        (128, 0, 255),
        (0, 128, 255),
        (255, 192, 203),
    ]

    for idx, mask_array in enumerate(masks):
        color = colors[idx % len(colors)]
        if mask_array.shape != overlay_array.shape[:2]:
            mask_image = Image.fromarray((mask_array > 0).astype(np.uint8) * 255)
            mask_image = mask_image.resize((image.width, image.height), resample=Image.NEAREST)
            mask_bool = np.asarray(mask_image, dtype=np.uint8) > 0
        else:
            mask_bool = mask_array > 0

        for c in range(3):
            overlay_array[mask_bool, c] = color[c]

    return Image.fromarray(overlay_array)


def flatten_polygon_coords(poly: Any) -> List[float]:
    coords = np.asarray(poly)
    if coords.ndim == 2 and coords.shape[1] == 2:
        coords = coords.reshape(-1)
    return [float(x) for x in coords]


def mask_to_polygon(mask: np.ndarray, image_size: Tuple[int, int]) -> List[float]:
    contours = measure.find_contours(mask.astype(np.uint8), 0.5)
    if not contours:
        return []
    contour = max(contours, key=len)
    width, height = image_size
    return [float(coord) for point in contour for coord in (point[1] / width, point[0] / height)]


def draw_polygon_overlay(image: Image.Image, polygons: List[List[float]]) -> Image.Image:
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    for polygon in polygons:
        if len(polygon) < 6:
            continue
        points = [(polygon[i] * image.width, polygon[i + 1] * image.height) for i in range(0, len(polygon), 2)]
        if len(points) < 3:
            continue
        draw.line(points + [points[0]], fill=(255, 0, 0), width=3)
    return overlay


def load_model(device: torch.device):
    """Load Mask R-CNN with pre-trained weights."""
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn(weights=weights)
    model.to(device)
    model.eval()
    return model, weights


def load_yolo_model(weights_path: Optional[str] = None):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("ultralytics package is required for YOLO segmentation") from exc

    if weights_path:
        weights_file = Path(weights_path)
        if not weights_file.exists():
            raise FileNotFoundError(f"YOLO weights file not found: {weights_path}")
        return YOLO(str(weights_file))

    if Path("yolov11-seg.pt").exists():
        return YOLO("yolov11-seg.pt")
    if Path("yolov8l-seg.pt").exists():
        return YOLO("yolov8l-seg.pt")
    if Path("yolov8n-seg.pt").exists():
        return YOLO("yolov8n-seg.pt")
    return YOLO("yolov8l-seg")


def segment_image_maskrcnn(
    model,
    transform,
    device: torch.device,
    image_path: Path,
    images_output: Path,
    labels_output: Path,
    score_threshold: float = 0.5,
) -> Dict[str, Any]:
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image)

    with torch.no_grad():
        prediction = model([image_tensor.to(device)])[0]

    scores = prediction["scores"].cpu().numpy()
    masks = prediction["masks"].cpu().numpy()
    boxes = prediction["boxes"].cpu().numpy()
    labels = prediction["labels"].cpu().numpy()

    images_output.mkdir(parents=True, exist_ok=True)
    labels_output.mkdir(parents=True, exist_ok=True)

    instance_count = 0
    masks_list: List[np.ndarray] = []
    instance_data: List[Dict[str, Any]] = []

    for idx, score in enumerate(scores):
        if score < score_threshold:
            continue

        instance_count += 1
        mask_array = (masks[idx, 0] > 0.5).astype(np.uint8)
        masks_list.append(mask_array)

        box = boxes[idx]
        polygon = mask_to_polygon(mask_array, (image.width, image.height))
        instance_data.append(
            {
                "instance_id": instance_count,
                "confidence": float(score),
                "bbox": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                "label_id": int(labels[idx]),
                "polygon": polygon,
            }
        )

    if instance_data:
        output_filename = images_output / f"{image_path.stem}_segmented.png"
        overlay = create_colored_overlay(image, masks_list)
        overlay.save(output_filename)

    polygon_txt_filename = labels_output / f"{image_path.stem}.txt"
    with open(polygon_txt_filename, "w", encoding="utf-8") as f:
        for inst in instance_data:
            if not inst["polygon"]:
                continue
            coords = " ".join(f"{coord:.6f}" for coord in inst["polygon"])
            f.write(f"{inst['label_id']} {coords}\n")

    print(f"Mask R-CNN segmented {image_path.name}: {instance_count} instance(s)")
    return {"filename": image_path.name, "instances": instance_count}


def segment_folder_maskrcnn(
    input_path: Path,
    images_output: Path,
    labels_output: Path,
) -> Dict[str, Any]:
    image_paths = get_image_paths(input_path)
    if not image_paths:
        raise FileNotFoundError(f"No supported images found in {input_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading Mask R-CNN model (device: {device})...")
    model, weights = load_model(device)
    transform = weights.transforms()

    total_instances = 0
    results_data = {
        "model": "Mask R-CNN (ResNet50-FPN)",
        "total_images": len(image_paths),
        "total_instances": 0,
        "images": [],
    }

    for idx, image_path in enumerate(image_paths, 1):
        result = segment_image_maskrcnn(model, transform, device, image_path, images_output, labels_output)
        total_instances += result["instances"]
        results_data["images"].append(result)
        print(f"[{idx}/{len(image_paths)}] Processed")

    results_data["total_instances"] = total_instances
    print(f"\nSegmentation complete: {total_instances} instances from {len(image_paths)} images")
    return results_data


def segment_folder_yolo(
    input_path: Path,
    images_output: Path,
    labels_output: Path,
    yolo_weights: Optional[str] = None,
    conf_threshold: float = 0.25,
) -> Dict[str, Any]:
    image_paths = get_image_paths(input_path)
    if not image_paths:
        raise FileNotFoundError(f"No supported images found in {input_path}")

    model = load_yolo_model(yolo_weights)

    images_output.mkdir(parents=True, exist_ok=True)
    labels_output.mkdir(parents=True, exist_ok=True)

    results_data = {
        "model": "YOLO",
        "total_images": len(image_paths),
        "total_instances": 0,
        "images": [],
    }

    total_instances = 0
    for idx, image_path in enumerate(image_paths, 1):
        with Image.open(image_path) as image_obj:
            width, height = image_obj.width, image_obj.height
            predictions = model.predict(str(image_path), conf=conf_threshold, save=False)
            instance_data: List[Dict[str, Any]] = []
            instance_count = 0
            mask_arrays: List[np.ndarray] = []

            for result in predictions:
                if result.masks is None:
                    continue

                boxes = result.boxes
                masks = result.masks
                for i in range(len(masks.data)):
                    instance_count += 1
                    bbox = boxes.xyxy[i].cpu().numpy().tolist()
                    confidence = float(boxes.conf[i].cpu().item()) if hasattr(boxes, "conf") else None
                    label_id = int(boxes.cls[i].cpu().item()) if hasattr(boxes, "cls") else None
                    mask_array = (masks.data[i] > 0.5).cpu().numpy().astype(np.uint8)
                    polygon = mask_to_polygon(mask_array, (width, height))
                    instance_data.append(
                        {
                            "instance_id": instance_count,
                            "confidence": confidence,
                            "bbox": bbox,
                            "label_id": label_id,
                            "polygon": polygon,
                        }
                    )
                    if mask_array.size:
                        mask_arrays.append(mask_array)

            if mask_arrays:
                overlay = create_colored_overlay(image_obj, mask_arrays)
                output_path = images_output / f"{image_path.stem}_segmented.png"
                overlay.save(output_path)

        polygon_txt_path = labels_output / f"{image_path.stem}.txt"
        with open(polygon_txt_path, "w", encoding="utf-8") as f:
            for inst in instance_data:
                if not inst["polygon"]:
                    continue
                coords = " ".join(f"{coord:.6f}" for coord in inst["polygon"])
                f.write(f"{inst['label_id']} {coords}\n")

        total_instances += instance_count
        results_data["images"].append({"filename": image_path.name, "instances": instance_count})
        print(f"YOLO segmented {image_path.name}: {instance_count} instance(s)")

    results_data["total_instances"] = total_instances
    print(f"\nSegmentation complete: {total_instances} instances from {len(image_paths)} images")
    return results_data


def segment_folder(
    input_path: Path,
    images_output: Path,
    labels_output: Path,
    model_name: str = "maskrcnn",
    yolo_weights: Optional[str] = None,
) -> Dict[str, Any]:
    model_key = model_name.lower()
    if model_key in {"maskrcnn", "mask_rcnn", "mask-rcnn"}:
        return segment_folder_maskrcnn(input_path, images_output, labels_output)
    if model_key in {"yolo", "yolov8", "yolov8l", "yolo11", "yolov11", "yolo11seg", "yolov11seg"}:
        return segment_folder_yolo(input_path, images_output, labels_output, yolo_weights=yolo_weights)
    raise ValueError(f"Unsupported model: {model_name}. Choose maskrcnn or yolo.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Instance segmentation using Mask R-CNN or YOLO.")
    parser.add_argument("--input", required=True, help="Input image file or folder")
    parser.add_argument("--images", required=True, help="Output folder for mask images")
    parser.add_argument("--labels", required=True, help="Output folder for labels")
    parser.add_argument(
        "--model",
        default="maskrcnn",
        choices=["maskrcnn", "yolo"],
        help="Segmentation model to run",
    )
    parser.add_argument(
        "--yolo-weights",
        default=None,
        help="Optional path to YOLO weights file",
    )
    args = parser.parse_args()

    segment_folder(
        Path(args.input),
        Path(args.images),
        Path(args.labels),
        model_name=args.model,
        yolo_weights=args.yolo_weights,
    )
