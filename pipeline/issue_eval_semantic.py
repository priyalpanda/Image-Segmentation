"""
Semantic grouping of issue_eval images using CLIP zero-shot classification.

Unlike K-Means on embeddings (geometry-only clusters), this report assigns each image
to a named scene category by comparing its embedding to CLIP text prompts.
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

from issue_eval_analysis import (
    analyze_classes_global,
    build_clip_encoder,
    compute_tsne_2d,
    layman_issue_for_image,
    list_images,
    per_image_label_stats,
    save_2d_embedding_plot,
)

# Scene categories: multiple text prompts per category (prompt ensembling).
SEMANTIC_CATEGORIES: Dict[str, Dict[str, Any]] = {
    "food_and_dining": {
        "title": "Food & dining",
        "why": "Meals, restaurants, kitchens, and table-focused scenes.",
        "prompts": [
            "a photo of food on a plate or bowl",
            "people eating at a restaurant",
            "a buffet or table full of many dishes",
            "a close-up of pizza or a cooked meal",
            "a packed lunchbox or bento with food items",
        ],
    },
    "aviation_aircraft": {
        "title": "Aviation & aircraft",
        "why": "Planes, airports, runways, and aerial views of aircraft.",
        "prompts": [
            "an airplane on a runway or taxiway",
            "a photograph of an aircraft in the sky",
            "an airport scene with airplanes",
            "a vintage historical photograph of a biplane",
            "an aerial view of a large passenger jet at an airport",
        ],
    },
    "rail_trains": {
        "title": "Rail & trains",
        "why": "Trains, tracks, stations, and railway yards.",
        "prompts": [
            "a passenger train on railway tracks",
            "a train at a station or rail yard",
            "multiple railroad tracks with a train",
        ],
    },
    "street_traffic": {
        "title": "Street & traffic",
        "why": "Roads, cars, buses, intersections, and urban traffic scenes.",
        "prompts": [
            "a busy city street with cars and traffic",
            "vehicles at a road intersection with traffic lights",
            "cars and buses on an urban road",
            "a parking lot full of cars",
        ],
    },
    "indoor_people_crowd": {
        "title": "Indoor people & crowds",
        "why": "Rooms with many people, gatherings, offices, or living spaces.",
        "prompts": [
            "a crowded indoor room with many people",
            "people gathered inside a building",
            "an indoor living room or lounge with people",
            "a group of people at an indoor event",
        ],
    },
    "sports_recreation": {
        "title": "Sports & recreation",
        "why": "Athletes, playing fields, courts, and outdoor sports.",
        "prompts": [
            "people playing sports outdoors",
            "a sports field or court with athletes",
            "skateboarding or snowboarding action",
            "a baseball or soccer game scene",
        ],
    },
    "animals": {
        "title": "Animals",
        "why": "Pets, wildlife, or animal-focused photos.",
        "prompts": [
            "a photo of a dog or cat",
            "farm animals or wildlife in a scene",
            "a person with a horse or riding animals",
        ],
    },
    "home_interior_objects": {
        "title": "Home interior & objects",
        "why": "Furniture, appliances, bathrooms, bedrooms — not primarily food.",
        "prompts": [
            "a living room with sofa and furniture",
            "a bathroom with toilet or sink",
            "a bedroom with bed and furniture",
            "a kitchen interior with appliances but not a meal close-up",
        ],
    },
    "retail_shop": {
        "title": "Retail & shops",
        "why": "Stores, shelves, products, and shopping environments.",
        "prompts": [
            "products on shelves inside a store",
            "a shop or supermarket aisle",
            "a market stall with goods for sale",
        ],
    },
    "nature_outdoor": {
        "title": "Nature & outdoor landscape",
        "why": "Parks, beaches, mountains, sky — not dominated by vehicles or food.",
        "prompts": [
            "a natural outdoor landscape with trees or grass",
            "a beach or ocean shoreline scene",
            "a park or garden without heavy urban traffic",
        ],
    },
    "water_boats": {
        "title": "Water & boats",
        "why": "Boats, harbors, and water-centric scenes.",
        "prompts": [
            "a boat on water or at a dock",
            "a harbor or marina with boats",
            "people on a beach near the ocean",
        ],
    },
}


def encode_clip_text_prompts(
    model,
    tokenizer,
    device,
    categories: Dict[str, Dict[str, Any]],
) -> Tuple[List[str], np.ndarray, Dict[str, slice]]:
    """Returns flat prompt list, normalized text embeddings, and category → row slice."""
    import torch
    import torch.nn.functional as F

    flat_prompts: List[str] = []
    cat_slices: Dict[str, slice] = {}
    offset = 0
    for cat_id, spec in categories.items():
        prompts = spec["prompts"]
        n = len(prompts)
        flat_prompts.extend(prompts)
        cat_slices[cat_id] = slice(offset, offset + n)
        offset += n

    tokens = tokenizer(flat_prompts).to(device)
    with torch.no_grad():
        text_feats = model.encode_text(tokens)
        text_feats = F.normalize(text_feats, dim=-1)
    return flat_prompts, text_feats.cpu().numpy(), cat_slices


def classify_images_semantically(
    model,
    preprocess,
    device,
    image_paths: List[Path],
    categories: Dict[str, Dict[str, Any]],
    clip_model: str = "ViT-B-32",
    batch_size: int = 8,
    temperature: float = 100.0,
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """
    Zero-shot semantic labels: each image → best-matching text category via CLIP similarity.
    """
    import open_clip
    import torch
    import torch.nn.functional as F

    tokenizer = open_clip.get_tokenizer(clip_model)
    _, text_emb, cat_slices = encode_clip_text_prompts(model, tokenizer, device, categories)
    text_t = torch.from_numpy(text_emb).to(device)

    cat_ids = list(categories.keys())
    per_image: List[Dict[str, Any]] = []
    image_emb_chunks: List[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start : start + batch_size]
            tensors = []
            for path in batch_paths:
                img = Image.open(path).convert("RGB")
                tensors.append(preprocess(img))
            batch = torch.stack(tensors).to(device)
            img_feats = model.encode_image(batch)
            img_feats = F.normalize(img_feats, dim=-1)
            image_emb_chunks.append(img_feats.cpu().numpy())

            # Cosine similarity per prompt, then mean per category; softmax over categories only.
            sims = (img_feats @ text_t.T).cpu().numpy()
            for i, path in enumerate(batch_paths):
                cat_scores: Dict[str, float] = {}
                for cat_id, sl in cat_slices.items():
                    cat_scores[cat_id] = float(sims[i, sl].mean())

                cat_ids_order = list(categories.keys())
                score_vec = np.array([cat_scores[c] for c in cat_ids_order], dtype=np.float64)
                exp_scores = np.exp(score_vec * temperature)
                cat_probs = exp_scores / exp_scores.sum()

                ranked_idx = np.argsort(-cat_probs)
                primary_id = cat_ids_order[int(ranked_idx[0])]
                primary_score = float(cat_probs[int(ranked_idx[0])])
                second_id = cat_ids_order[int(ranked_idx[1])] if len(ranked_idx) > 1 else ""
                second_score = float(cat_probs[int(ranked_idx[1])]) if second_id else 0.0
                margin = primary_score - second_score

                per_image.append(
                    {
                        "image": path.name,
                        "stem": path.stem,
                        "semantic_category": primary_id,
                        "semantic_title": categories[primary_id]["title"],
                        "confidence": round(primary_score, 4),
                        "margin_vs_runner_up": round(margin, 4),
                        "runner_up_category": second_id,
                        "runner_up_title": categories[second_id]["title"] if second_id else "",
                        "all_category_scores": {
                            cat_ids_order[j]: round(float(cat_probs[j]), 4)
                            for j in range(len(cat_ids_order))
                        },
                        "raw_cosine_scores": {
                            k: round(v, 4) for k, v in sorted(cat_scores.items(), key=lambda x: -x[1])
                        },
                    }
                )

    image_matrix = np.vstack(image_emb_chunks)
    return per_image, image_matrix


def semantic_category_to_id(categories: Dict[str, Dict[str, Any]]) -> Dict[str, int]:
    return {cat_id: idx for idx, cat_id in enumerate(categories)}


def render_semantic_html(report: Dict[str, Any]) -> str:
    groups_html = []
    for grp in report.get("semantic_groups", []):
        imgs = "".join(f"<li><code>{html.escape(i)}</code></li>" for i in grp.get("images", []))
        groups_html.append(
            f"""
            <section class="group">
              <h3>{html.escape(grp['title'])} — {grp['image_count']} images</h3>
              <p>{html.escape(grp['why'])}</p>
              <p><em>Avg CLIP confidence in this group: {grp.get('avg_confidence', 0):.1%}</em></p>
              <ul>{imgs}</ul>
            </section>
            """
        )

    rows = "".join(
        f"<tr><td>{html.escape(r['image'])}</td>"
        f"<td>{html.escape(r['semantic_title'])}</td>"
        f"<td>{r['confidence']:.1%}</td>"
        f"<td>{html.escape(r['runner_up_title'])}</td>"
        f"<td>{html.escape(r['layman_issue'][:100])}…</td></tr>"
        for r in sorted(report.get("per_image", []), key=lambda x: x["semantic_category"])
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Semantic CLIP report — issue images</title>
  <style>
    body {{ font-family: system-ui, sans-serif; max-width: 960px; margin: 2rem auto; padding: 0 1rem; }}
    .group {{ border: 1px solid #ddd; border-radius: 8px; padding: 1rem; margin: 1rem 0; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 0.9rem; }}
    th, td {{ border: 1px solid #ccc; padding: 0.4rem 0.6rem; text-align: left; }}
    img.plot {{ max-width: 100%; }}
    .note {{ background: #f5f8ff; padding: 1rem; border-radius: 8px; }}
  </style>
</head>
<body>
  <h1>Semantic grouping (CLIP zero-shot)</h1>
  <div class="note">
    <p><strong>{report['image_count']}</strong> images from <code>issue_eval/images</code>.</p>
    <p>{html.escape(report['method']['summary'])}</p>
    <p><strong>Not the same as the K-Means report:</strong> {html.escape(report['comparison_to_kmeans'])}</p>
  </div>

  <h2>t-SNE colored by semantic category</h2>
  <img class="plot" src="tsne_by_semantics.png" alt="t-SNE by semantics"/>

  <h2>Groups by meaning (scene type)</h2>
  {''.join(groups_html)}

  <h2>Every image — assigned category</h2>
  <table>
    <thead><tr><th>Image</th><th>Semantic group</th><th>CLIP confidence</th><th>Runner-up</th><th>Segmentation issue</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
</body>
</html>
"""


def run_semantic_analysis(
    images_dir: Path,
    output_dir: Path,
    groundtruth_dir: Optional[Path] = None,
    predicted_dir: Optional[Path] = None,
    clip_model: str = "ViT-B-32",
    clip_pretrained: str = "openai",
) -> Dict[str, Any]:
    image_paths = list_images(images_dir)
    stems = [p.stem for p in image_paths]
    n_images = len(image_paths)

    label_stats = per_image_label_stats(stems, groundtruth_dir, predicted_dir)
    class_analysis = analyze_classes_global(stems, groundtruth_dir, predicted_dir)

    _, model, preprocess, device, dim, backbone_label = build_clip_encoder(clip_model, clip_pretrained)

    semantic_rows, image_embeddings = classify_images_semantically(
        model,
        preprocess,
        device,
        image_paths,
        SEMANTIC_CATEGORIES,
        clip_model=clip_model,
    )

    cat_to_int = semantic_category_to_id(SEMANTIC_CATEGORIES)
    color_labels = np.array([cat_to_int[r["semantic_category"]] for r in semantic_rows])

    from sklearn.preprocessing import StandardScaler

    features_scaled = StandardScaler().fit_transform(image_embeddings)
    tsne_coords, _ = compute_tsne_2d(features_scaled)

    members_by_cat: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for idx, row in enumerate(semantic_rows):
        stem = row["stem"]
        st = label_stats.get(stem, {})
        row["tsne_x"] = round(float(tsne_coords[idx, 0]), 4)
        row["tsne_y"] = round(float(tsne_coords[idx, 1]), 4)
        row["gt_instance_count"] = st.get("gt_instance_count", 0)
        row["pred_instance_count"] = st.get("pred_instance_count", 0)
        row["missed_coco_classes"] = st.get("missed_class_names", [])
        row["layman_issue"] = layman_issue_for_image(st)
        members_by_cat[row["semantic_category"]].append(row)

    semantic_groups: List[Dict[str, Any]] = []
    for cat_id in SEMANTIC_CATEGORIES:
        members = members_by_cat.get(cat_id, [])
        if not members:
            continue
        spec = SEMANTIC_CATEGORIES[cat_id]
        confs = [m["confidence"] for m in members]
        semantic_groups.append(
            {
                "category_id": cat_id,
                "title": spec["title"],
                "why": spec["why"],
                "prompts_used": spec["prompts"],
                "image_count": len(members),
                "avg_confidence": round(float(np.mean(confs)), 4),
                "images": [m["image"] for m in members],
                "members": members,
            }
        )
    semantic_groups.sort(key=lambda g: -g["image_count"])

    output_dir.mkdir(parents=True, exist_ok=True)
    names = [p.name for p in image_paths]
    save_2d_embedding_plot(
        tsne_coords,
        color_labels,
        names,
        output_dir / "tsne_by_semantics.png",
        f"t-SNE by semantic category ({backbone_label})",
        "t-SNE dimension 1",
        "t-SNE dimension 2",
    )

    report = {
        "report_type": "semantic_clip_zero_shot",
        "source": "issue_eval/images only (original photos)",
        "image_count": n_images,
        "comparison_to_kmeans": (
            "The file issue_cluster_analysis.json groups by K-Means on embedding distance "
            "(numeric clusters 0–4, not named themes). This report groups by CLIP text–image "
            "similarity to explicit scene descriptions (food, aviation, street, etc.)."
        ),
        "method": {
            "technique": "CLIP zero-shot semantic classification",
            "backbone": backbone_label,
            "embedding_dim": dim,
            "summary": (
                "Each image is compared to natural-language scene prompts. The highest-scoring "
                "category is the semantic group. Prompts are averaged per category (prompt ensembling)."
            ),
            "temperature": 100.0,
            "categories": {
                cat_id: {"title": spec["title"], "prompt_count": len(spec["prompts"])}
                for cat_id, spec in SEMANTIC_CATEGORIES.items()
            },
        },
        "semantic_groups": semantic_groups,
        "per_image": semantic_rows,
        "class_analysis": class_analysis,
    }

    json_path = output_dir / "semantic_cluster_analysis.json"
    html_path = output_dir / "semantic_cluster_analysis.html"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    html_path.write_text(render_semantic_html(report), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Semantic CLIP zero-shot grouping for issue_eval/images (separate from K-Means report)."
    )
    parser.add_argument("--images-dir", default="issue_eval/images")
    parser.add_argument(
        "--output-dir",
        default="issue_eval/reports/semantic",
        help="Writes semantic_cluster_analysis.json/html and tsne_by_semantics.png",
    )
    parser.add_argument("--groundtruth-labels", default="labels/train2017")
    parser.add_argument("--predicted-labels", default="results/yolo/labels")
    parser.add_argument("--clip-model", default="ViT-B-32")
    parser.add_argument("--clip-pretrained", default="openai")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    images_dir = (root / args.images_dir).resolve()
    output_dir = (root / args.output_dir).resolve()
    gt_dir = (root / args.groundtruth_labels).resolve()
    pred_dir = (root / args.predicted_labels).resolve()

    report = run_semantic_analysis(
        images_dir=images_dir,
        output_dir=output_dir,
        groundtruth_dir=gt_dir if gt_dir.exists() else None,
        predicted_dir=pred_dir if pred_dir.exists() else None,
        clip_model=args.clip_model,
        clip_pretrained=args.clip_pretrained,
    )

    print(f"Semantic report: {len(report['semantic_groups'])} categories, {report['image_count']} images")
    for g in report["semantic_groups"]:
        print(f"  {g['title']}: {g['image_count']} (avg conf {g['avg_confidence']:.1%})")
    out = (root / args.output_dir).resolve()
    print(f"JSON: {out / 'semantic_cluster_analysis.json'}")
    print(f"HTML: {out / 'semantic_cluster_analysis.html'}")


if __name__ == "__main__":
    main()
