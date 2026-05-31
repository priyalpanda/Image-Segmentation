"""Build struggle/observation reports from a model evaluation.json file."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


def stem_from_image_name(image_name: str) -> str:
    return Path(image_name).stem


def count_label_instances(label_path: Path) -> int:
    if not label_path.exists():
        return 0
    count = 0
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 7 and (len(parts) - 1) % 2 == 0:
                count += 1
    return count


def load_label_counts(
    predicted_labels_dir: Optional[Path],
    groundtruth_dir: Optional[Path],
) -> Tuple[Dict[str, int], Dict[str, int]]:
    gt_counts: Dict[str, int] = {}
    pred_counts: Dict[str, int] = {}
    if groundtruth_dir and groundtruth_dir.exists():
        for path in groundtruth_dir.rglob("*.txt"):
            gt_counts[path.stem] = count_label_instances(path)
    if predicted_labels_dir and predicted_labels_dir.exists():
        for path in predicted_labels_dir.glob("*.txt"):
            pred_counts[path.stem] = count_label_instances(path)
    return gt_counts, pred_counts


def is_empty_evaluation_pattern(metrics: Dict[str, Any]) -> bool:
    return (
        metrics.get("iou", -1) == 0.0
        and metrics.get("dice", -1) == 0.0
        and metrics.get("false_negative", -1) == 0
        and metrics.get("false_positive", -1) == 0
        and metrics.get("pixel_accuracy", -1) == 1.0
    )


STRUGGLE_CATEGORIES: Dict[str, Dict[str, str]] = {
    "no_segmentation_output": {
        "title": "No segmentation output",
        "summary": "The model produced no masks or no label file for these images.",
        "why": (
            "Without predictions there is nothing to compare to ground truth. "
            "Common causes: confidence threshold too high, image content unlike training data, "
            "or pipeline did not write labels."
        ),
    },
    "unevaluable_no_groundtruth": {
        "title": "Cannot evaluate (no ground truth)",
        "summary": "Predictions exist but there is no matching ground-truth label file.",
        "why": (
            "Metrics are not meaningful without GT. These cases are flagged so they are not "
            "confused with model performance on labeled data."
        ),
    },
    "cluttered_scene_low_iou": {
        "title": "Heavily cluttered — low mask overlap",
        "summary": "Many objects in the scene and low average IoU on matched instances.",
        "why": (
            "Crowded scenes with overlapping instances make instance boundaries hard to separate. "
            "The model may detect some objects but mask shapes overlap poorly with ground truth, "
            "driving IoU down."
        ),
    },
    "cluttered_scene_missed_instances": {
        "title": "Heavily cluttered — many missed objects",
        "summary": "Dense scenes where the model misses a large share of ground-truth instances.",
        "why": (
            "High instance count increases occlusion and competition between detections. "
            "The model often finds the salient objects but fails to recall smaller or partially "
            "hidden instances (high false negatives)."
        ),
    },
    "cluttered_scene_false_alarms": {
        "title": "Heavily cluttered — extra detections",
        "summary": "Dense scenes with many spurious or duplicate predictions.",
        "why": (
            "In clutter, the model may fire multiple boxes on the same object or on background "
            "texture, producing false positives alongside true objects."
        ),
    },
    "total_mask_mismatch": {
        "title": "Detections present but masks do not match",
        "summary": "Both GT and predictions exist, but matched IoU is zero.",
        "why": (
            "The model sees something in the image but segment boundaries or classes do not align "
            "with any GT instance at the IoU threshold — wrong object, wrong region, or systematic "
            "coordinate/scale mismatch."
        ),
    },
    "severe_under_detection": {
        "title": "Severe under-detection",
        "summary": "Very high false negatives relative to the number of objects.",
        "why": (
            "The model captures only a fraction of labeled instances. Often seen in busy images, "
            "small objects, or classes the weights handle poorly."
        ),
    },
    "missed_instances": {
        "title": "Moderate missed detections",
        "summary": "Several ground-truth instances were not matched.",
        "why": (
            "Recall is incomplete: some objects are detected with usable IoU, but others are "
            "skipped entirely or below the confidence cutoff."
        ),
    },
    "over_segmentation": {
        "title": "Over-segmentation / false alarms",
        "summary": "More predictions than ground-truth matches; extra instances.",
        "why": (
            "The model splits one object into multiple masks, detects background as objects, "
            "or assigns wrong classes — inflating false positives."
        ),
    },
    "poor_mask_quality": {
        "title": "Poor mask alignment",
        "summary": "Objects are found but average IoU is below threshold with few total misses.",
        "why": (
            "Localization is weak: coarse polygons, boundary shrinkage, or class-correct boxes "
            "with imprecise mask edges reduce IoU without always increasing false negatives."
        ),
    },
    "partial_miss_and_localization": {
        "title": "Partial miss with weak localization",
        "summary": "Mixed errors: some missed instances and modest IoU.",
        "why": (
            "Combination of recall and precision issues — typical when scenes are moderately "
            "complex or objects vary in scale."
        ),
    },
    "count_or_class_mismatch": {
        "title": "Count or class mismatch",
        "summary": "Instance counts or classes do not line up with ground truth.",
        "why": (
            "Matched masks may exist but wrong COCO class IDs or unequal GT vs prediction counts "
            "prevent a clean per-image match."
        ),
    },
}


def compute_clutter_threshold(gt_counts: List[int], default: int = 8) -> int:
    positive = sorted(c for c in gt_counts if c > 0)
    if len(positive) < 4:
        return default
    index = int(0.75 * (len(positive) - 1))
    return max(default, positive[index])


def assign_struggle_category(
    obs: Dict[str, Any],
    clutter_gt_threshold: int,
    low_iou_threshold: float,
) -> Tuple[str, str]:
    gt_count = int(obs.get("gt_count", 0))
    pred_count = int(obs.get("pred_count", 0))
    metrics = obs.get("metrics", {})
    issues = obs.get("issues", [])
    iou = float(metrics.get("iou", 0.0))
    fn = int(metrics.get("false_negative", 0))
    fp = int(metrics.get("false_positive", 0))
    miss_rate = fn / gt_count if gt_count else 0.0
    is_cluttered = gt_count >= clutter_gt_threshold

    if "no_segmentation" in issues:
        return (
            "no_segmentation_output",
            f"No prediction instances (GT={gt_count}, pred={pred_count}). {obs.get('notes', '')}",
        )
    if "no_groundtruth" in issues and gt_count == 0:
        return (
            "unevaluable_no_groundtruth",
            "No ground-truth labels available for this image; segmentation output cannot be scored.",
        )

    if is_cluttered:
        if iou < low_iou_threshold:
            return (
                "cluttered_scene_low_iou",
                f"{gt_count} GT instances (cluttered, ≥{clutter_gt_threshold}); average IoU {iou:.2f} — "
                "dense overlap likely hurts mask quality.",
            )
        if fn >= 5 or miss_rate >= 0.35:
            return (
                "cluttered_scene_missed_instances",
                f"{gt_count} GT instances (cluttered); {fn} missed ({miss_rate:.0%} of GT) — "
                "model recalls only part of the crowded scene.",
            )
        if fp >= 5:
            return (
                "cluttered_scene_false_alarms",
                f"{gt_count} GT instances (cluttered); {fp} extra predictions — "
                "duplicate or spurious masks in a busy image.",
            )

    if iou == 0.0 and gt_count > 0 and pred_count > 0:
        return (
            "total_mask_mismatch",
            f"GT={gt_count}, pred={pred_count}, but zero IoU on matches — "
            "detections do not align with any GT mask.",
        )

    if fn >= 10 or (gt_count >= 5 and miss_rate >= 0.5):
        return (
            "severe_under_detection",
            f"{fn} of {gt_count} instances missed ({miss_rate:.0%}) — severe recall failure.",
        )

    if fn >= 2 and pred_count < gt_count:
        return (
            "missed_instances",
            f"{fn} missed of {gt_count} GT objects (pred={pred_count}) — incomplete detection.",
        )

    if fp >= 3 and fp > fn:
        return (
            "over_segmentation",
            f"{fp} false positives vs {fn} false negatives — model over-predicts instances.",
        )

    if 0.0 < iou < low_iou_threshold and fn <= 1:
        return (
            "poor_mask_quality",
            f"IoU {iou:.2f} with at most {fn} miss — objects found but masks are imprecise.",
        )

    if fn > 0 and iou < low_iou_threshold:
        return (
            "partial_miss_and_localization",
            f"IoU {iou:.2f}, {fn} missed, {fp} extra — mixed recall and boundary errors.",
        )

    if fn > 0:
        return (
            "missed_instances",
            f"{fn} missed of {gt_count} GT instance(s).",
        )

    if fp > 0:
        return ("over_segmentation", f"{fp} unmatched prediction(s).")

    return (
        "count_or_class_mismatch",
        "Prediction counts or classes do not fully agree with ground truth.",
    )


def classify_image_issues(
    metrics: Dict[str, Any],
    stem: str,
    missing_predictions: Set[str],
    missing_groundtruth: Set[str],
    gt_count: int,
    pred_count: int,
    low_iou_threshold: float,
) -> Tuple[List[str], str, str]:
    issues: List[str] = []
    notes: List[str] = []

    if stem in missing_predictions:
        issues.append("no_segmentation")
        notes.append("Ground-truth labels exist but no prediction label file was produced.")
    elif pred_count == 0 and gt_count > 0:
        issues.append("no_segmentation")
        notes.append(f"Model produced no instances while ground truth has {gt_count} instance(s).")
    elif pred_count == 0 and is_empty_evaluation_pattern(metrics):
        issues.append("no_segmentation")
        if stem in missing_groundtruth:
            notes.append(
                "No segmentation output (empty labels). Listed under missing_groundtruth — "
                "no ground-truth file to compare against."
            )
        else:
            notes.append("No instances in prediction labels (empty segmentation).")
    elif pred_count == 0 and gt_count == 0 and stem in missing_groundtruth:
        issues.append("no_groundtruth")
        notes.append("Prediction label file exists but there is no matching ground-truth file.")

    if stem in missing_groundtruth and "no_groundtruth" not in issues:
        issues.append("no_groundtruth")
        notes.append("No ground-truth label file for this image; metrics are not fully comparable.")

    fn = int(metrics.get("false_negative", 0))
    fp = int(metrics.get("false_positive", 0))
    iou = float(metrics.get("iou", 0.0))

    if fn > 0 and "no_segmentation" not in issues:
        issues.append("false_negative")
        notes.append(f"{fn} ground-truth instance(s) were missed (false negative).")

    if iou == 0.0 and (fn > 0 or fp > 0) and "low_iou" not in issues:
        issues.append("low_iou")
        notes.append("Zero average IoU on matched instances — masks do not overlap ground truth.")

    if 0.0 < iou < low_iou_threshold:
        issues.append("low_iou")
        notes.append(f"Average IoU {iou:.4f} is below threshold {low_iou_threshold:.2f}.")

    if metrics.get("classification_correct") is False:
        if fn > 0 or fp > 0 or iou < low_iou_threshold or pred_count != gt_count:
            if "classification_error" not in issues:
                issues.append("classification_error")
            notes.append("Instance count or class alignment does not match ground truth.")

    if fp > 0 and "no_segmentation" not in issues:
        issues.append("false_positive")
        notes.append(f"{fp} extra predicted instance(s) not matched to ground truth.")

    # Deduplicate while preserving order
    seen: Set[str] = set()
    ordered_issues: List[str] = []
    for issue in issues:
        if issue not in seen:
            seen.add(issue)
            ordered_issues.append(issue)

    if not ordered_issues:
        return [], "none", ""

    severity = "medium"
    if (
        "no_segmentation" in ordered_issues
        or stem in missing_predictions
        or (pred_count == 0 and is_empty_evaluation_pattern(metrics))
    ):
        severity = "critical"
    elif fn >= 5 or iou == 0.0 or (fn >= 3 and iou < low_iou_threshold):
        severity = "high"
    elif fn >= 1 or iou < low_iou_threshold:
        severity = "medium"

    return ordered_issues, severity, " ".join(notes)


def build_observation_report(
    evaluation: Dict[str, Any],
    predicted_labels_dir: Optional[Path] = None,
    groundtruth_dir: Optional[Path] = None,
    results_dir: Optional[Path] = None,
    low_iou_threshold: float = 0.5,
) -> Dict[str, Any]:
    missing_predictions = set(evaluation.get("missing_predictions", []))
    missing_groundtruth = set(evaluation.get("missing_groundtruth", []))
    gt_counts, pred_counts = load_label_counts(predicted_labels_dir, groundtruth_dir)

    all_gt_counts = list(gt_counts.values())
    clutter_gt_threshold = compute_clutter_threshold(all_gt_counts)

    observations: List[Dict[str, Any]] = []
    by_issue: Dict[str, List[str]] = {
        "low_iou": [],
        "false_negative": [],
        "classification_error": [],
        "no_segmentation": [],
        "no_groundtruth": [],
        "false_positive": [],
    }

    for entry in evaluation.get("images", []):
        image_name = entry["image"]
        stem = stem_from_image_name(image_name)
        gt_count = entry.get("gt_count", gt_counts.get(stem, 0))
        pred_count = entry.get("pred_count", pred_counts.get(stem, 0))

        issues, severity, notes = classify_image_issues(
            entry,
            stem,
            missing_predictions,
            missing_groundtruth,
            gt_count,
            pred_count,
            low_iou_threshold,
        )
        if not issues:
            continue

        record = {
            "image": image_name,
            "stem": stem,
            "severity": severity,
            "issues": issues,
            "notes": notes,
            "metrics": dict(entry),
            "gt_count": gt_count,
            "pred_count": pred_count,
        }
        if results_dir:
            segmented = results_dir / "images" / f"{stem}_segmented.png"
            if segmented.exists():
                record["segmented_image"] = str(segmented.relative_to(results_dir)).replace("\\", "/")

        category_id, category_reason = assign_struggle_category(
            record, clutter_gt_threshold, low_iou_threshold
        )
        record["struggle_category"] = category_id
        record["struggle_category_title"] = STRUGGLE_CATEGORIES[category_id]["title"]
        record["struggle_reason"] = category_reason

        observations.append(record)
        for issue in issues:
            by_issue.setdefault(issue, []).append(image_name)

    observations = [o for o in observations if o["severity"] in {"critical", "high"}]
    by_issue = {
        issue: [name for name in names if any(o["image"] == name for o in observations)]
        for issue, names in by_issue.items()
    }

    categories = build_category_groups(observations, clutter_gt_threshold)

    severity_rank = {"critical": 0, "high": 1}
    observations.sort(
        key=lambda row: (
            severity_rank.get(row["severity"], 9),
            row.get("struggle_category", ""),
            -row["metrics"].get("false_negative", 0),
            row["image"],
        )
    )

    return {
        "model": evaluation.get("model", "unknown"),
        "source_evaluation": "evaluation.json",
        "thresholds": {
            "low_iou": low_iou_threshold,
            "cluttered_min_gt_instances": clutter_gt_threshold,
        },
        "summary": {
            "total_images_in_evaluation": len(evaluation.get("images", [])),
            "struggling_images_count": len(observations),
            "critical_count": sum(1 for o in observations if o["severity"] == "critical"),
            "high_count": sum(1 for o in observations if o["severity"] == "high"),
            "by_issue_count": {k: len(v) for k, v in by_issue.items() if v},
            "by_category_count": {c["id"]: c["image_count"] for c in categories},
        },
        "categories": categories,
        "highlighted_cases": by_issue,
        "observations": observations,
        "missing_predictions": sorted(missing_predictions),
        "missing_groundtruth": sorted(missing_groundtruth),
    }


def build_category_groups(
    observations: List[Dict[str, Any]],
    clutter_gt_threshold: int,
) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for obs in observations:
        grouped.setdefault(obs["struggle_category"], []).append(obs)

    category_order = [
        "no_segmentation_output",
        "unevaluable_no_groundtruth",
        "cluttered_scene_low_iou",
        "cluttered_scene_missed_instances",
        "cluttered_scene_false_alarms",
        "total_mask_mismatch",
        "severe_under_detection",
        "missed_instances",
        "over_segmentation",
        "poor_mask_quality",
        "partial_miss_and_localization",
        "count_or_class_mismatch",
    ]

    result: List[Dict[str, Any]] = []
    for category_id in category_order:
        items = grouped.pop(category_id, [])
        if not items:
            continue
        ious = [float(o["metrics"].get("iou", 0)) for o in items]
        fns = [int(o["metrics"].get("false_negative", 0)) for o in items]
        fps = [int(o["metrics"].get("false_positive", 0)) for o in items]
        gts = [int(o.get("gt_count", 0)) for o in items]
        meta = STRUGGLE_CATEGORIES[category_id]
        result.append(
            {
                "id": category_id,
                "title": meta["title"],
                "summary": meta["summary"],
                "why_model_struggles": meta["why"],
                "image_count": len(items),
                "typical_signals": {
                    "avg_iou": round(sum(ious) / len(ious), 4),
                    "avg_false_negative": round(sum(fns) / len(fns), 2),
                    "avg_false_positive": round(sum(fps) / len(fps), 2),
                    "avg_gt_instances": round(sum(gts) / len(gts), 2),
                },
                "images": [o["image"] for o in items],
                "examples": [
                    {
                        "image": o["image"],
                        "iou": o["metrics"].get("iou"),
                        "false_negative": o["metrics"].get("false_negative"),
                        "false_positive": o["metrics"].get("false_positive"),
                        "gt_count": o.get("gt_count"),
                        "pred_count": o.get("pred_count"),
                        "reason": o.get("struggle_reason"),
                    }
                    for o in sorted(
                        items,
                        key=lambda row: (
                            -int(row["metrics"].get("false_negative", 0)),
                            float(row["metrics"].get("iou", 0)),
                        ),
                    )[:8]
                ],
            }
        )

    for category_id, items in grouped.items():
        if items and category_id in STRUGGLE_CATEGORIES:
            meta = STRUGGLE_CATEGORIES[category_id]
            result.append(
                {
                    "id": category_id,
                    "title": meta["title"],
                    "summary": meta["summary"],
                    "why_model_struggles": meta["why"],
                    "image_count": len(items),
                    "images": [o["image"] for o in items],
                    "examples": [],
                }
            )
    return result


def render_html_report(report: Dict[str, Any], evaluation_summary: Dict[str, Any]) -> str:
    issue_colors = {
        "no_segmentation": "#c0392b",
        "no_groundtruth": "#8e44ad",
        "low_iou": "#e67e22",
        "false_negative": "#d35400",
        "classification_error": "#2980b9",
        "false_positive": "#16a085",
    }
    severity_colors = {"critical": "#fdecea", "high": "#fff4e5"}

    rows_html = []
    for obs in report["observations"]:
        badges = "".join(
            f'<span class="badge" style="background:{issue_colors.get(i, "#555")}">{html.escape(i)}</span>'
            for i in obs["issues"]
        )
        m = obs["metrics"]
        img_cell = html.escape(obs["image"])
        if obs.get("segmented_image"):
            img_cell = (
                f'<div>{html.escape(obs["image"])}</div>'
                f'<img src="{html.escape(obs["segmented_image"])}" alt="segmented" width="160" />'
            )
        cat_title = html.escape(obs.get("struggle_category_title", ""))
        rows_html.append(
            f"""
            <tr class="{obs['severity']}" style="background:{severity_colors.get(obs['severity'], '#fff')}">
              <td>{img_cell}</td>
              <td>{cat_title}</td>
              <td><strong>{html.escape(obs['severity'])}</strong></td>
              <td>{badges}</td>
              <td>{m.get('iou', 0):.4f}</td>
              <td>{m.get('false_negative', 0)}</td>
              <td>{m.get('false_positive', 0)}</td>
              <td>{'yes' if m.get('classification_correct') else 'no'}</td>
              <td>{obs.get('gt_count', '—')}</td>
              <td>{obs.get('pred_count', '—')}</td>
              <td>{html.escape(obs.get('notes', ''))}</td>
            </tr>
            """
        )

    summary = report["summary"]
    eval_sum = evaluation_summary or {}
    issue_list = "".join(
        f"<li><strong>{html.escape(k)}</strong>: {v} image(s)</li>"
        for k, v in summary.get("by_issue_count", {}).items()
        if v > 0
    )

    category_sections = []
    for cat in report.get("categories", []):
        examples_html = "".join(
            f"<li><code>{html.escape(ex['image'])}</code> — IoU {ex.get('iou', 0):.2f}, "
            f"FN {ex.get('false_negative', 0)}, GT {ex.get('gt_count', 0)}: "
            f"{html.escape(ex.get('reason', ''))}</li>"
            for ex in cat.get("examples", [])[:5]
        )
        sig = cat.get("typical_signals", {})
        category_sections.append(
            f"""
            <section class="category-block" id="{html.escape(cat['id'])}">
              <h2>{html.escape(cat['title'])} <span class="count">({cat['image_count']} images)</span></h2>
              <p class="cat-summary">{html.escape(cat.get('summary', ''))}</p>
              <p class="cat-why"><strong>Why the model struggles:</strong> {html.escape(cat.get('why_model_struggles', ''))}</p>
              <p class="signals">Typical signals — avg IoU: {sig.get('avg_iou', 'n/a')}, "
              "avg FN: {sig.get('avg_false_negative', 'n/a')}, avg GT instances: {sig.get('avg_gt_instances', 'n/a')}</p>
              <details>
                <summary>Example images ({min(len(cat.get('examples', [])), 5)} shown)</summary>
                <ul>{examples_html or '<li>No examples</li>'}</ul>
              </details>
            </section>
            """
        )
    categories_html = "".join(category_sections)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Segmentation observation report — {html.escape(report.get('model', ''))}</title>
  <style>
    body {{ font-family: Segoe UI, Arial, sans-serif; margin: 24px; color: #222; }}
    h1 {{ margin-bottom: 0.2em; }}
    .meta {{ color: #555; margin-bottom: 24px; }}
    .cards {{ display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 24px; }}
    .card {{ background: #f4f6f8; border-radius: 8px; padding: 14px 18px; min-width: 140px; }}
    .card strong {{ display: block; font-size: 1.4em; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; vertical-align: top; text-align: left; }}
    th {{ background: #2c3e50; color: #fff; }}
    .badge {{ display: inline-block; color: #fff; padding: 2px 8px; border-radius: 4px; margin: 2px 4px 2px 0; font-size: 12px; }}
    .callout {{ border-left: 4px solid #c0392b; background: #fdf2f2; padding: 12px 16px; margin-bottom: 24px; }}
    .callout.critical {{ border-color: #c0392b; }}
    ul {{ margin: 8px 0 0 20px; }}
    .category-block {{ background: #fafbfc; border: 1px solid #e1e4e8; border-radius: 8px; padding: 16px 20px; margin-bottom: 20px; }}
    .category-block h2 {{ margin-top: 0; font-size: 1.15em; }}
    .category-block .count {{ color: #666; font-weight: normal; font-size: 0.9em; }}
    .cat-why {{ background: #fff; padding: 10px; border-radius: 6px; border-left: 3px solid #3498db; }}
    .signals {{ font-size: 13px; color: #555; }}
  </style>
</head>
<body>
  <h1>Segmentation struggle report</h1>
  <p class="meta">Model: <strong>{html.escape(report.get('model', ''))}</strong> ·
  Source: {html.escape(report.get('source_evaluation', ''))} ·
  Low IoU threshold: {report['thresholds']['low_iou']}</p>

  <div class="cards">
    <div class="card"><span>Struggling images</span><strong>{summary['struggling_images_count']}</strong></div>
    <div class="card"><span>Critical</span><strong>{summary['critical_count']}</strong></div>
    <div class="card"><span>High</span><strong>{summary['high_count']}</strong></div>
    <div class="card"><span>Eval avg IoU</span><strong>{eval_sum.get('average_iou', 0):.4f}</strong></div>
  </div>
  <p class="meta">Showing <strong>critical</strong> and <strong>high</strong> severity only; medium issues are excluded.</p>

  <section>
    <h2>Struggle categories</h2>
    <p>Images are grouped by the dominant failure mode. Cluttered scenes use ≥
    <strong>{report['thresholds'].get('cluttered_min_gt_instances', 8)}</strong> ground-truth instances (75th percentile of your dataset).</p>
    {categories_html if categories_html else '<p>No categories.</p>'}
  </section>

  <section>
    <h2>Issue tags (technical)</h2>
    <ul>{issue_list}</ul>
  </section>

  <section>
    <h2>All struggling images</h2>
    <table>
      <thead>
        <tr>
          <th>Image</th><th>Category</th><th>Severity</th><th>Issues</th><th>IoU</th><th>FN</th><th>FP</th>
          <th>Class OK</th><th>GT #</th><th>Pred #</th><th>Notes</th>
        </tr>
      </thead>
      <tbody>
        {''.join(rows_html) if rows_html else '<tr><td colspan="11">No struggling images found.</td></tr>'}
      </tbody>
    </table>
  </section>
</body>
</html>
"""


def save_observation_reports(
    report: Dict[str, Any],
    output_dir: Path,
    evaluation_summary: Optional[Dict[str, Any]] = None,
) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "observation_report.json"
    html_path = output_dir / "observation_report.html"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(render_html_report(report, evaluation_summary or {}))
    return json_path, html_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate struggle/observation report from evaluation.json")
    parser.add_argument("--evaluation", required=True, help="Path to evaluation.json")
    parser.add_argument("--output-dir", default=None, help="Output folder (defaults to evaluation parent)")
    parser.add_argument("--predicted-labels", default=None, help="Optional predicted labels folder")
    parser.add_argument("--groundtruth-labels", default=None, help="Optional ground-truth labels folder")
    parser.add_argument("--low-iou-threshold", type=float, default=0.5)
    args = parser.parse_args()

    eval_path = Path(args.evaluation).expanduser().resolve()
    output_dir = Path(args.output_dir or eval_path.parent).expanduser().resolve()
    with open(eval_path, "r", encoding="utf-8") as f:
        evaluation = json.load(f)

    predicted = Path(args.predicted_labels).resolve() if args.predicted_labels else output_dir / "labels"
    groundtruth = Path(args.groundtruth_labels).resolve() if args.groundtruth_labels else None

    report = build_observation_report(
        evaluation,
        predicted_labels_dir=predicted if predicted.exists() else None,
        groundtruth_dir=groundtruth,
        results_dir=output_dir,
        low_iou_threshold=args.low_iou_threshold,
    )
    json_path, html_path = save_observation_reports(report, output_dir, evaluation.get("summary"))
    print(f"Saved observation JSON: {json_path}")
    print(f"Saved observation HTML: {html_path}")
    print(f"Struggling images: {report['summary']['struggling_images_count']}")
    print(f"Critical: {report['summary']['critical_count']}")


if __name__ == "__main__":
    main()
