"""
Unsupervised computer vision on issue_eval/images only.

Standard workflow (no human bias):
  1. Feature extraction — CLIP vision encoder (default) or CNN outputs embedding vectors per image.
  2. Clustering — K-Means groups vectors by distance; t-SNE/UMAP plots groups in 2D.
"""

from __future__ import annotations

import argparse
import html
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

# Official COCO 80 classes used by YOLOv8-seg / YOLO11-seg (Ultralytics, MS COCO)
COCO_SEG_CLASSES: List[str] = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def coco_class_name(class_id: int) -> str:
    if 0 <= class_id < len(COCO_SEG_CLASSES):
        return COCO_SEG_CLASSES[class_id]
    return f"unknown_class_{class_id}"


def list_images(images_dir: Path) -> List[Path]:
    files = [
        p
        for p in sorted(images_dir.iterdir())
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    if not files:
        raise FileNotFoundError(f"No images found in {images_dir}")
    return files


def load_label_class_ids(label_path: Path) -> List[int]:
    if not label_path.exists():
        return []
    ids: List[int] = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 7 and (len(parts) - 1) % 2 == 0:
                ids.append(int(parts[0]))
    return ids


def per_image_label_stats(
    stems: List[str],
    groundtruth_dir: Optional[Path],
    predicted_dir: Optional[Path],
) -> Dict[str, Dict[str, Any]]:
    stats: Dict[str, Dict[str, Any]] = {}
    for stem in stems:
        gt_ids = load_label_class_ids(groundtruth_dir / f"{stem}.txt") if groundtruth_dir else []
        pred_ids = load_label_class_ids(predicted_dir / f"{stem}.txt") if predicted_dir else []
        gt_set = set(gt_ids)
        pred_set = set(pred_ids)
        missed_classes = sorted(gt_set - pred_set)
        extra_classes = sorted(pred_set - gt_set)
        stats[stem] = {
            "gt_instance_count": len(gt_ids),
            "pred_instance_count": len(pred_ids),
            "gt_class_ids": gt_ids,
            "pred_class_ids": pred_ids,
            "missed_class_names": [coco_class_name(i) for i in missed_classes],
            "extra_class_names": [coco_class_name(i) for i in extra_classes],
        }
    return stats


def analyze_classes_global(
    stems: List[str],
    groundtruth_dir: Optional[Path],
    predicted_dir: Optional[Path],
) -> Dict[str, Any]:
    gt_counter: Counter = Counter()
    pred_counter: Counter = Counter()
    miss_counter: Counter = Counter()

    for stem in stems:
        gt_ids = load_label_class_ids(groundtruth_dir / f"{stem}.txt") if groundtruth_dir else []
        pred_ids = load_label_class_ids(predicted_dir / f"{stem}.txt") if predicted_dir else []
        gt_counter.update(gt_ids)
        pred_counter.update(pred_ids)
        for cid in set(gt_ids) - set(pred_ids):
            miss_counter[cid] += 1

    total_gt = sum(gt_counter.values()) or 1

    def ranked(counter: Counter, top_n: int = 20) -> List[Dict[str, Any]]:
        return [
            {
                "class_id": cid,
                "coco_name": coco_class_name(cid),
                "instance_count": count,
                "share_of_gt": round(count / total_gt, 4),
            }
            for cid, count in counter.most_common(top_n)
        ]

    never_predicted = [
        {
            "class_id": cid,
            "coco_name": coco_class_name(cid),
            "gt_instances": gt_counter[cid],
            "images_where_missed": miss_counter[cid],
        }
        for cid in sorted(gt_counter)
        if pred_counter[cid] == 0
    ]
    never_predicted.sort(key=lambda x: (-x["gt_instances"], -x["images_where_missed"]))

    return {
        "coco_dataset": "MS COCO (80 classes, YOLOv8-seg / YOLO11-seg)",
        "total_gt_instances": sum(gt_counter.values()),
        "total_pred_instances": sum(pred_counter.values()),
        "top_classes_in_issue_images": ranked(gt_counter),
        "top_predicted_classes": ranked(pred_counter),
        "classes_never_detected": never_predicted[:30],
        "classes_most_often_completely_missed": sorted(
            [
                {
                    "coco_name": coco_class_name(cid),
                    "images_affected": miss_counter[cid],
                    "gt_instances": gt_counter[cid],
                }
                for cid in miss_counter
            ],
            key=lambda x: (-x["images_affected"], -x["gt_instances"]),
        )[:20],
    }


def build_clip_encoder(
    clip_model: str = "ViT-B-32",
    clip_pretrained: str = "openai",
):
    """CLIP vision branch — image embeddings aligned with text (no labels used at inference)."""
    import open_clip
    import torch

    model, _, preprocess = open_clip.create_model_and_transforms(
        clip_model, pretrained=clip_pretrained
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    dim = int(getattr(model.visual, "output_dim", 512))
    label = f"CLIP {clip_model} ({clip_pretrained})"
    return "clip", model, preprocess, device, dim, label


def build_cnn_encoder(backbone: str):
    """Pre-trained CNN without the classification head — returns embedding vectors only."""
    import torch
    import torch.nn as nn
    from torchvision import models

    name = backbone.lower()
    if name in {"resnet50", "resnet"}:
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        model = models.resnet50(weights=weights)
        model.fc = nn.Identity()
        transform = weights.transforms()
        dim = 2048
        label = "ResNet50 (ImageNet)"
    elif name == "vgg16":
        weights = models.VGG16_Weights.IMAGENET1K_V1
        model = models.vgg16(weights=weights)
        model.classifier = nn.Sequential(nn.Flatten())
        transform = weights.transforms()
        dim = 25088
        label = "VGG16 (ImageNet)"
    else:
        raise ValueError(f"Unsupported CNN backbone '{backbone}'.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    return "cnn", model, transform, device, dim, label


def load_feature_encoder(
    backbone: str = "clip",
    clip_model: str = "ViT-B-32",
    clip_pretrained: str = "openai",
):
    name = backbone.lower()
    if name in {"clip", "openclip", "open_clip"}:
        return build_clip_encoder(clip_model, clip_pretrained)
    return build_cnn_encoder(name)


def extract_embedding_vectors(
    image_paths: List[Path],
    backbone: str = "clip",
    batch_size: int = 16,
    clip_model: str = "ViT-B-32",
    clip_pretrained: str = "openai",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Step 1: each image → one numeric embedding vector (shape: n_images × embedding_dim)."""
    import torch
    import torch.nn.functional as F

    encoder_type, model, transform, device, _dim, label = load_feature_encoder(
        backbone, clip_model=clip_model, clip_pretrained=clip_pretrained
    )
    if encoder_type == "clip":
        batch_size = min(batch_size, 8)

    chunks: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start : start + batch_size]
            tensors = []
            for path in batch_paths:
                img = Image.open(path).convert("RGB")
                tensors.append(transform(img))
            batch = torch.stack(tensors).to(device)
            if encoder_type == "clip":
                feats = model.encode_image(batch)
                feats = F.normalize(feats, dim=-1)
            else:
                feats = model(batch)
            chunks.append(feats.cpu().numpy())

    matrix = np.vstack(chunks)
    if encoder_type == "clip":
        desc = (
            "Each row is one CLIP image embedding (L2-normalized). "
            "Captures semantic visual similarity (scene type, objects, layout) "
            "without using your segmentation labels."
        )
    else:
        desc = (
            "Each row is one CNN image embedding. "
            "Encodes low-level visual patterns without class names."
        )
    meta = {
        "backbone": label,
        "encoder_type": encoder_type,
        "embedding_dim": int(matrix.shape[1]),
        "num_images": int(matrix.shape[0]),
        "normalized": encoder_type == "clip",
        "description": desc,
    }
    return matrix, meta


def pick_best_k(features_scaled: np.ndarray, min_k: int = 2, max_k: int = 8) -> Tuple[int, List[Dict[str, Any]]]:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    n = len(features_scaled)
    max_k = min(max_k, n - 1)
    if max_k < min_k:
        return 1, [{"k": 1, "silhouette": 0.0}]

    scores: List[Dict[str, Any]] = []
    best_k = min_k
    best_score = -1.0
    for k in range(min_k, max_k + 1):
        labels = KMeans(n_clusters=k, random_state=42, n_init=15).fit_predict(features_scaled)
        sil = float(silhouette_score(features_scaled, labels)) if k > 1 else 0.0
        scores.append({"k": k, "silhouette": round(sil, 4)})
        if sil > best_score:
            best_score = sil
            best_k = k
    return best_k, scores


def kmeans_cluster_embeddings(features_scaled: np.ndarray, n_clusters: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Step 2a: K-Means — assign each image to exactly one of K visual groups."""
    from sklearn.cluster import KMeans

    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=15)
    labels = model.fit_predict(features_scaled)
    return labels, {
        "algorithm": "K-Means",
        "n_clusters": n_clusters,
        "description": (
            "Images in the same cluster have similar embedding vectors "
            "(mathematically close in high-dimensional space)."
        ),
    }


def compute_tsne_2d(features_scaled: np.ndarray) -> Tuple[np.ndarray, Dict[str, str]]:
    from sklearn.manifold import TSNE

    n = len(features_scaled)
    perp = min(30.0, max(5.0, (n - 1) / 3))
    coords = TSNE(
        n_components=2,
        random_state=42,
        perplexity=perp,
        init="pca",
        learning_rate="auto",
    ).fit_transform(features_scaled)
    return coords, {
        "method": "t-SNE",
        "description": "2D map: points close together = visually similar photos (for human review).",
    }


def compute_umap_2d(features_scaled: np.ndarray) -> Tuple[Optional[np.ndarray], Dict[str, str]]:
    try:
        import umap

        n = len(features_scaled)
        n_neighbors = min(15, max(2, n - 1))
        coords = umap.UMAP(
            n_components=2,
            random_state=42,
            n_neighbors=n_neighbors,
            min_dist=0.1,
        ).fit_transform(features_scaled)
        return coords, {
            "method": "UMAP",
            "description": "Alternative 2D layout; often preserves global structure better than t-SNE.",
        }
    except ImportError:
        return None, {
            "method": "UMAP",
            "description": "UMAP not installed (pip install umap-learn). t-SNE map used instead.",
        }


def layman_issue_for_image(stats: Dict[str, Any]) -> str:
    gt_n = stats["gt_instance_count"]
    pred_n = stats["pred_instance_count"]
    missed = stats["missed_class_names"]
    if gt_n == 0 and pred_n == 0:
        return "No labeled objects to compare — the model did not produce usable masks for scoring."
    if pred_n == 0:
        return (
            f"The model found nothing, but {gt_n} object(s) should be there. "
            "The scene was effectively skipped."
        )
    if pred_n < gt_n * 0.5:
        return (
            f"The scene is crowded ({gt_n} objects labeled). The model only found about {pred_n}, "
            "so many things were missed — often smaller items or overlapping objects."
        )
    if missed:
        names = ", ".join(missed[:4])
        extra = "…" if len(missed) > 4 else ""
        return (
            f"The model often misses entire categories here (e.g. {names}{extra}). "
            f"Labeled {gt_n} objects vs {pred_n} detected."
        )
    return (
        f"Some objects were found ({pred_n} of {gt_n}), but masks may still be wrong shape or class. "
        "This image is in the hard set because overall evaluation flagged it."
    )


def layman_cluster_title(
    cluster_class_counts: Counter,
    avg_gt: float,
) -> str:
    top = [coco_class_name(c) for c, _ in cluster_class_counts.most_common(3)]
    top_str = ", ".join(top) if top else "mixed content"
    if avg_gt >= 12:
        return f"Busy scenes — lots of objects ({top_str})"
    if avg_gt >= 6:
        return f"Moderately crowded scenes ({top_str})"
    if "person" in top[:2]:
        return f"People-focused scenes ({top_str})"
    if any(x in top for x in ("car", "bus", "truck", "bicycle")):
        return f"Street / vehicles ({top_str})"
    if any(x in top for x in ("chair", "dining table", "cup", "bowl", "bottle")):
        return f"Indoor / table items ({top_str})"
    return f"Mixed outdoor or indoor ({top_str})"


def layman_cluster_why(cluster_title: str, avg_gt: float, avg_miss_rate: float) -> str:
    if "Busy" in cluster_title or avg_gt >= 12:
        return (
            "These photos have many things in one frame. The model tends to catch the big, obvious "
            "objects and miss smaller or overlapping ones — like a busy street or a table full of items."
        )
    if avg_miss_rate >= 0.4:
        return (
            "In this group the model regularly skips whole types of objects. Training needs more "
            "examples of these scenes and those missing categories."
        )
    if "People" in cluster_title:
        return (
            "People appear often here. Crowds, partial bodies, and overlap make it hard to draw a "
            "clean outline around each person."
        )
    if "Street" in cluster_title or "vehicles" in cluster_title:
        return (
            "Cars and outdoor scenes dominate. Objects overlap and vary in size, which confuses "
            "instance masks."
        )
    if "Indoor" in cluster_title or "table" in cluster_title:
        return (
            "Indoor scenes with dishes, furniture, and small items. Small objects (fork, cup, etc.) "
            "are easy to miss next to larger ones."
        )
    return (
        "Visually similar images group together. They share hard segmentation patterns such as "
        "overlap, small objects, or missing detections."
    )


def training_recommendations(class_analysis: Dict[str, Any], clusters: List[Dict[str, Any]]) -> Dict[str, Any]:
    top = class_analysis.get("top_classes_in_issue_images", [])[:8]
    never = class_analysis.get("classes_never_detected", [])[:10]
    top_names = [c["coco_name"] for c in top]
    never_names = [c["coco_name"] for c in never]

    collect_data = [
        "More photos like the issue set: many objects per image (10+ labeled instances).",
        "Scenes with overlap (people in crowds, items on tables, parked cars close together).",
        f"Extra images containing: {', '.join(top_names[:5])} — these appear most in failed cases.",
        f"Targeted shots for rarely detected classes: {', '.join(never_names) or 'small COCO objects'}.",
        "Small objects at medium distance (utensils, sports gear, handheld items).",
        "Same lighting and camera style as your deployment images (not only clean studio shots).",
    ]

    training_actions = [
        "Fine-tune YOLOv8-seg (or your seg model) on COCO-format labels with polygon masks.",
        "Use mosaic / copy-paste augmentation to simulate crowded scenes.",
        "Increase max detections per image and slightly lower confidence at inference for dense scenes.",
        "Oversample training batches from high-FN images listed in this report.",
        "Add class-focused epochs or loss weight for person and other top failure classes.",
        "Validate on a held-out set of issue_eval-style crowded images, not only easy single-object images.",
    ]

    metrics_to_track = [
        "Recall per class (especially person, car, chair, cup, bowl).",
        "Mask IoU on images with 10+ instances.",
        "False negative rate = missed GT instances / total GT instances.",
    ]

    return {
        "data_to_collect": collect_data,
        "training_actions": training_actions,
        "metrics_to_track": metrics_to_track,
        "model_reference": "YOLOv8-seg / COCO 80-class segmentation (Ultralytics)",
    }


def save_2d_embedding_plot(
    coords: np.ndarray,
    labels: np.ndarray,
    names: List[str],
    output_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10", alpha=0.85, s=70)
    if len(names) <= 35:
        for i, name in enumerate(names):
            plt.annotate(Path(name).stem[-8:], (coords[i, 0], coords[i, 1]), fontsize=6, alpha=0.75)
    plt.colorbar(scatter, label="K-Means cluster ID")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_combined_maps(
    tsne_coords: np.ndarray,
    umap_coords: Optional[np.ndarray],
    labels: np.ndarray,
    output_path: Path,
    backbone_label: str,
) -> None:
    import matplotlib.pyplot as plt

    cols = 2 if umap_coords is not None else 1
    fig, axes = plt.subplots(1, cols, figsize=(7 * cols, 6))
    if cols == 1:
        axes = [axes]
    axes[0].scatter(tsne_coords[:, 0], tsne_coords[:, 1], c=labels, cmap="tab10", alpha=0.85, s=60)
    axes[0].set_title(f"t-SNE — {backbone_label}")
    axes[0].set_xlabel("t-SNE 1")
    axes[0].set_ylabel("t-SNE 2")
    if umap_coords is not None:
        axes[1].scatter(umap_coords[:, 0], umap_coords[:, 1], c=labels, cmap="tab10", alpha=0.85, s=60)
        axes[1].set_title(f"UMAP — {backbone_label}")
        axes[1].set_xlabel("UMAP 1")
        axes[1].set_ylabel("UMAP 2")
    fig.suptitle("Unsupervised grouping: image embeddings + K-Means (colors = cluster)", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def render_html_report(report: Dict[str, Any]) -> str:
    train = report.get("training_guide", {})
    clusters_html = []
    for cluster in report.get("visual_groups", []):
        imgs = "".join(f"<li><code>{html.escape(i)}</code></li>" for i in cluster.get("images", []))
        classes = ", ".join(
            f"{c['coco_name']} ({c['instance_count']})"
            for c in cluster.get("common_coco_classes", [])[:6]
        )
        clusters_html.append(
            f"""
            <section class="cluster">
              <h3>{html.escape(cluster['layman_title'])} — {cluster['image_count']} images</h3>
              <p><strong>In plain English:</strong> {html.escape(cluster['layman_why'])}</p>
              <p><strong>Common COCO objects in this group:</strong> {html.escape(classes or 'n/a')}</p>
              <p>Avg labeled objects per image: {cluster.get('avg_objects_per_image', 0):.1f}</p>
              <ul>{imgs}</ul>
            </section>
            """
        )

    class_rows = "".join(
        f"<tr><td>{c['class_id']}</td><td>{html.escape(c['coco_name'])}</td>"
        f"<td>{c['instance_count']}</td><td>{c['share_of_gt']:.1%}</td></tr>"
        for c in report.get("class_analysis", {}).get("top_classes_in_issue_images", [])[:12]
    )
    never_rows = "".join(
        f"<tr><td>{html.escape(c['coco_name'])}</td><td>{c['gt_instances']}</td>"
        f"<td>{c.get('images_where_missed', '')}</td></tr>"
        for c in report.get("class_analysis", {}).get("classes_never_detected", [])[:12]
    )

    collect_li = "".join(f"<li>{html.escape(x)}</li>" for x in train.get("data_to_collect", []))
    action_li = "".join(f"<li>{html.escape(x)}</li>" for x in train.get("training_actions", []))
    metric_li = "".join(f"<li>{html.escape(x)}</li>" for x in train.get("metrics_to_track", []))

    per_image_rows = "".join(
        f"<tr><td>{html.escape(r['image'])}</td><td>{r['kmeans_cluster']}</td>"
        f"<td>{html.escape(r['layman_issue'][:120])}...</td></tr>"
        for r in report.get("per_image", [])[:25]
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Issue image pattern analysis</title>
  <style>
    body {{ font-family: Segoe UI, Arial, sans-serif; margin: 24px; max-width: 960px; color: #222; line-height: 1.5; }}
    h1 {{ margin-bottom: 0.2em; }}
    .cluster {{ background: #f6f8fa; border-radius: 8px; padding: 16px; margin-bottom: 16px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 14px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; vertical-align: top; }}
    th {{ background: #2c3e50; color: #fff; }}
    img.plot {{ max-width: 100%; border-radius: 8px; }}
    .box {{ background: #e8f4fc; padding: 14px; border-radius: 8px; margin: 16px 0; }}
  </style>
</head>
<body>
  <h1>Unsupervised pattern analysis — issue images</h1>
  <p><strong>{report['image_count']}</strong> original photos from <code>issue_eval/images</code> only.
  No manual sorting — pure math.</p>

  <h2>Method (standard unsupervised computer vision)</h2>
  <ol>
    <li><strong>Feature extraction:</strong> {html.escape(report['unsupervised_workflow']['step1_feature_extraction']['technique'])} — {html.escape(report['unsupervised_workflow']['step1_feature_extraction']['summary'])}</li>
    <li><strong>Clustering (K-Means):</strong> {html.escape(report['unsupervised_workflow']['step2_kmeans']['summary'])}</li>
    <li><strong>2D maps (t-SNE / UMAP):</strong> {html.escape(report['unsupervised_workflow']['step3_visualization']['summary'])}</li>
  </ol>
  <p>Backbone: <strong>{html.escape(report['embedding']['backbone'])}</strong> ·
  Vector size: <strong>{report['embedding']['embedding_dim']}</strong> numbers per image ·
  K = <strong>{report['clustering']['chosen_k']}</strong> clusters ·
  Silhouette: {report['clustering']['silhouette_score']}</p>

  <img class="plot" src="embedding_maps.png" alt="t-SNE and UMAP maps" />
  <p><img class="plot" src="tsne_clusters.png" alt="t-SNE" style="max-width:48%" />
  {'<img class="plot" src="umap_clusters.png" alt="UMAP" style="max-width:48%" />' if report.get('umap_available') else ''}</p>

  <h2>K-Means groups (every image in exactly one cluster)</h2>
  <p>{html.escape(report['clustering']['k_selection_note'])}</p>
  {''.join(clusters_html)}

  <h2>COCO classes that show up most in these hard images</h2>
  <table><thead><tr><th>ID</th><th>COCO class</th><th>Labeled instances</th><th>Share</th></tr></thead>
  <tbody>{class_rows}</tbody></table>

  <h2>COCO classes the model often never detects here</h2>
  <table><thead><tr><th>Class</th><th>GT instances</th><th>Images affected</th></tr></thead>
  <tbody>{never_rows}</tbody></table>

  <div class="box">
    <h2>What to collect &amp; how to train better</h2>
    <h3>Data to collect</h3>
    <ul>{collect_li}</ul>
    <h3>Training steps</h3>
    <ul>{action_li}</ul>
    <h3>Metrics to watch</h3>
    <ul>{metric_li}</ul>
  </div>

  <h2>Per-image plain-English notes (sample)</h2>
  <table><thead><tr><th>Image</th><th>Visual group</th><th>What goes wrong</th></tr></thead>
  <tbody>{per_image_rows}</tbody></table>
</body>
</html>
"""


def run_analysis(
    images_dir: Path,
    output_dir: Path,
    groundtruth_dir: Optional[Path] = None,
    predicted_dir: Optional[Path] = None,
    n_clusters: Optional[int] = None,
    backbone: str = "clip",
    clip_model: str = "ViT-B-32",
    clip_pretrained: str = "openai",
) -> Dict[str, Any]:
    from sklearn.preprocessing import StandardScaler

    image_paths = list_images(images_dir)
    stems = [p.stem for p in image_paths]
    n_images = len(image_paths)

    label_stats = per_image_label_stats(stems, groundtruth_dir, predicted_dir)
    class_analysis = analyze_classes_global(stems, groundtruth_dir, predicted_dir)

    embeddings, embedding_meta = extract_embedding_vectors(
        image_paths,
        backbone=backbone,
        clip_model=clip_model,
        clip_pretrained=clip_pretrained,
    )
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(embeddings)

    if n_clusters is None:
        best_k, k_scores = pick_best_k(features_scaled)
        best_sil = next((s["silhouette"] for s in k_scores if s["k"] == best_k), 0.0)
        # Weak separation → use more groups so every bucket is easier to act on
        if n_images >= 40 and (best_sil < 0.15 or best_k < 4):
            chosen_k = min(6, max(4, round(n_images / 12)))
            k_note = (
                f"Auto-selected k={chosen_k} for clearer groups "
                f"(best silhouette was only {best_sil:.3f} at k={best_k})."
            )
        else:
            chosen_k = best_k
            k_note = f"Selected k={chosen_k} with highest silhouette ({best_sil:.3f})."
    else:
        chosen_k = min(max(1, n_clusters), n_images)
        k_scores = [{"k": chosen_k, "silhouette": None}]
        k_note = f"Manual k={chosen_k}."

    labels, kmeans_meta = kmeans_cluster_embeddings(features_scaled, chosen_k)
    assert len(labels) == n_images, "Every image must have a cluster label"
    assert len(set(labels)) <= chosen_k

    sil = 0.0
    if chosen_k > 1:
        from sklearn.metrics import silhouette_score

        sil = float(silhouette_score(features_scaled, labels))

    tsne_coords, tsne_meta = compute_tsne_2d(features_scaled)
    umap_coords, umap_meta = compute_umap_2d(features_scaled)

    cluster_members: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for idx, path in enumerate(image_paths):
        stem = path.stem
        st = label_stats.get(stem, {})
        cluster_id = int(labels[idx])
        miss_rate = 0.0
        if st.get("gt_instance_count", 0) > 0:
            miss_rate = max(0, st["gt_instance_count"] - st.get("pred_instance_count", 0)) / st["gt_instance_count"]
        member = {
            "image": path.name,
            "stem": stem,
            "kmeans_cluster": cluster_id,
            "tsne_x": round(float(tsne_coords[idx, 0]), 4),
            "tsne_y": round(float(tsne_coords[idx, 1]), 4),
            "gt_instance_count": st.get("gt_instance_count", 0),
            "pred_instance_count": st.get("pred_instance_count", 0),
            "missed_coco_classes": st.get("missed_class_names", []),
            "layman_issue": layman_issue_for_image(st),
        }
        if umap_coords is not None:
            member["umap_x"] = round(float(umap_coords[idx, 0]), 4)
            member["umap_y"] = round(float(umap_coords[idx, 1]), 4)
        cluster_members[cluster_id].append(member)

    visual_groups: List[Dict[str, Any]] = []
    cluster_titles: Dict[int, str] = {}
    for cluster_id in sorted(cluster_members):
        members = cluster_members[cluster_id]
        class_counter: Counter = Counter()
        for m in members:
            stem = m["stem"]
            for cid in label_stats.get(stem, {}).get("gt_class_ids", []):
                class_counter[cid] += 1
        avg_gt = float(np.mean([m["gt_instance_count"] for m in members]))
        avg_miss = float(np.mean([
            max(0, m["gt_instance_count"] - m["pred_instance_count"]) / m["gt_instance_count"]
            if m["gt_instance_count"] else 0
            for m in members
        ]))
        title = layman_cluster_title(class_counter, avg_gt)
        cluster_titles[cluster_id] = title
        visual_groups.append(
            {
                "group_id": cluster_id,
                "layman_title": title,
                "layman_why": layman_cluster_why(title, avg_gt, avg_miss),
                "image_count": len(members),
                "images": [m["image"] for m in members],
                "avg_objects_per_image": round(avg_gt, 2),
                "common_coco_classes": [
                    {
                        "class_id": cid,
                        "coco_name": coco_class_name(cid),
                        "instance_count": cnt,
                    }
                    for cid, cnt in class_counter.most_common(8)
                ],
                "members": members,
            }
        )

    per_image = sorted(
        [m for members in cluster_members.values() for m in members],
        key=lambda x: (x["kmeans_cluster"], x["image"]),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    names = [p.name for p in image_paths]
    save_2d_embedding_plot(
        tsne_coords,
        labels,
        names,
        output_dir / "tsne_clusters.png",
        f"t-SNE map ({embedding_meta['backbone']})",
        "t-SNE dimension 1",
        "t-SNE dimension 2",
    )
    umap_available = umap_coords is not None
    if umap_available:
        save_2d_embedding_plot(
            umap_coords,
            labels,
            names,
            output_dir / "umap_clusters.png",
            f"UMAP map ({embedding_meta['backbone']})",
            "UMAP dimension 1",
            "UMAP dimension 2",
        )
    save_combined_maps(tsne_coords, umap_coords, labels, output_dir / "embedding_maps.png", embedding_meta["backbone"])

    report = {
        "source": "issue_eval/images only (original photos)",
        "image_count": n_images,
        "all_images_grouped": True,
        "unsupervised_workflow": {
            "step1_feature_extraction": {
                "technique": (
                    "CLIP vision encoder (L2-normalized image embeddings)"
                    if embedding_meta.get("encoder_type") == "clip"
                    else "Pre-trained CNN (no classification head)"
                ),
                "summary": embedding_meta["description"],
                "backbone": embedding_meta["backbone"],
                "embedding_dim": embedding_meta["embedding_dim"],
            },
            "step2_kmeans": {
                "technique": "K-Means clustering",
                "summary": kmeans_meta["description"],
                "chosen_k": chosen_k,
                "silhouette_score": round(sil, 4),
                "k_selection_note": k_note,
                "k_search_scores": k_scores,
            },
            "step3_visualization": {
                "technique": "t-SNE and UMAP (2D)",
                "summary": "Reduce high-dimensional embeddings to a scatter plot so you can see which photos sit near each other.",
                "tsne": tsne_meta,
                "umap": umap_meta,
            },
        },
        "embedding": embedding_meta,
        "clustering": {
            "algorithm": "K-Means on standardized image embeddings",
            "chosen_k": chosen_k,
            "k_selection_note": k_note,
            "k_search_scores": k_scores,
            "silhouette_score": round(sil, 4),
        },
        "umap_available": umap_available,
        "class_analysis": class_analysis,
        "visual_groups": visual_groups,
        "per_image": per_image,
        "training_guide": training_recommendations(class_analysis, visual_groups),
    }

    json_path = output_dir / "issue_cluster_analysis.json"
    html_path = output_dir / "issue_cluster_analysis.html"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(render_html_report(report))

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cluster issue_eval/images with CLIP/CNN embeddings, K-Means, t-SNE/UMAP, and COCO class analysis."
    )
    parser.add_argument(
        "--images-dir",
        default="issue_eval/images",
        help="Folder with original issue images only",
    )
    parser.add_argument(
        "--output-dir",
        default="issue_eval/reports",
        help="Where to write JSON, HTML, and t-SNE plot",
    )
    parser.add_argument(
        "--groundtruth-labels",
        default="labels/train2017",
        help="Optional GT labels for COCO class stats",
    )
    parser.add_argument(
        "--predicted-labels",
        default="results/yolo/labels",
        help="Optional predicted labels for miss analysis",
    )
    parser.add_argument("--n-clusters", type=int, default=None)
    parser.add_argument(
        "--backbone",
        default="clip",
        choices=["clip", "resnet50", "vgg16"],
        help="Feature extractor: clip (default), resnet50, or vgg16",
    )
    parser.add_argument(
        "--clip-model",
        default="ViT-B-32",
        help="OpenCLIP architecture when --backbone clip (e.g. ViT-B-32, ViT-L-14)",
    )
    parser.add_argument(
        "--clip-pretrained",
        default="openai",
        help="OpenCLIP weights tag (e.g. openai, laion2b_s34b_b79k)",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    images_dir = (root / args.images_dir).resolve()
    output_dir = (root / args.output_dir).resolve()
    gt_dir = (root / args.groundtruth_labels).resolve()
    pred_dir = (root / args.predicted_labels).resolve()

    report = run_analysis(
        images_dir=images_dir,
        output_dir=output_dir,
        groundtruth_dir=gt_dir if gt_dir.exists() else None,
        predicted_dir=pred_dir if pred_dir.exists() else None,
        n_clusters=args.n_clusters,
        backbone=args.backbone,
        clip_model=args.clip_model,
        clip_pretrained=args.clip_pretrained,
    )
    print(f"Analyzed {report['image_count']} images from {images_dir}")
    print(f"Groups: {report['clustering']['chosen_k']} (silhouette {report['clustering']['silhouette_score']})")
    print(f"JSON: {output_dir / 'issue_cluster_analysis.json'}")
    print(f"HTML: {output_dir / 'issue_cluster_analysis.html'}")


if __name__ == "__main__":
    main()
