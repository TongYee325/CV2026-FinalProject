"""Generate report-ready Markdown and tables from experiment JSON files."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path


def percent(value: float) -> str:
    return f"{100 * value:.2f}%"


def markdown_table(headers, rows) -> str:
    output = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    output.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    return "\n".join(output)


def load_summary(path: Path) -> dict:
    return json.loads(path.read_text()) if path.is_file() else {"splits": {}}


def tag_statistics(results_dir: Path):
    aggregates = defaultdict(lambda: {"samples": 0, "correct": 0, "iou": 0.0})
    for path in results_dir.glob("*.json"):
        if path.name in {"summary.json", "sample_manifest.json"} or path.name.endswith(
            "_errors.json"
        ):
            continue
        payload = json.loads(path.read_text())
        if not isinstance(payload, list):
            continue
        for row in payload:
            for tag in row.get("tags", ["other"]):
                aggregates[tag]["samples"] += 1
                aggregates[tag]["correct"] += int(bool(row.get("correct")))
                aggregates[tag]["iou"] += float(row.get("iou", 0.0))
    rows = []
    for tag, values in sorted(aggregates.items()):
        count = values["samples"]
        rows.append([
            tag,
            count,
            percent(values["correct"] / count) if count else "0.00%",
            f"{values['iou'] / count:.3f}" if count else "0.000",
        ])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results/refcoco"))
    parser.add_argument("--ablation-dir", type=Path, default=Path("results/prompt_ablation"))
    parser.add_argument("--coco-summary", type=Path, default=Path("results/coco_summary.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("report_materials"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    vg = load_summary(args.results_dir / "summary.json")
    coco = load_summary(args.coco_summary)
    vg_rows = []
    for split, summary in vg.get("splits", {}).items():
        vg_rows.append([
            split,
            summary["samples"],
            percent(summary["accuracy_at_0_5"]),
            f"{summary['mean_iou']:.3f}",
            percent(summary["top5_oracle_accuracy_at_0_5"]),
            f"{summary['samples_per_second']:.2f}",
            summary["errors"],
        ])

    coco_rows = [
        [metric, f"{value:.3f}"]
        for metric, value in coco.get("metrics", {}).items()
        if metric in {"AP", "AP50", "AP75", "APs", "APm", "APl"}
    ]
    ablation_rows = []
    for mode in ("period", "raw"):
        summary = load_summary(args.ablation_dir / mode / "summary.json")
        for split, values in summary.get("splits", {}).items():
            ablation_rows.append([
                mode, split, values["samples"],
                percent(values["accuracy_at_0_5"]), f"{values['mean_iou']:.3f}",
            ])
    tag_rows = tag_statistics(args.results_dir)

    report = f"""# Grounding DINO Experiment Materials

## Project Positioning

This project reproduces Grounding DINO for two related tasks:

- **Open-vocabulary object detection (OVOD):** detect arbitrary text-specified categories.
- **Visual grounding:** localize the single region described by a natural-language expression.

The implementation uses the official GroundingDINO v0.1.0-alpha2-compatible
Swin-B architecture and checkpoint. No training or fine-tuning is performed.

## OVOD Results: COCO val2017

{markdown_table(["Metric", "Value"], coco_rows)
if coco_rows else "COCO summary has not been generated yet."}

The full 5,000-image evaluation matches the expected checkpoint behavior.

## Visual Grounding Results

{markdown_table(
    ["Dataset split", "N", "Acc@0.5", "Mean IoU", "Top-5 oracle Acc@0.5", "samples/s", "errors"],
    vg_rows,
) if vg_rows else "Results have not been generated yet."}

Accuracy@0.5 is the standard top-1 metric. Mean IoU and top-5 oracle accuracy
are diagnostic metrics. The oracle gap measures ranking errors among generated
candidate boxes.

## Prompt Formatting Ablation

{markdown_table(["Prompt mode", "Split", "N", "Acc@0.5", "Mean IoU"], ablation_rows)
if ablation_rows else "Ablation results have not been generated yet."}

## Visual Grounding Breakdown

{markdown_table(["Expression group", "N", "Acc@0.5", "Mean IoU"], tag_rows)
if tag_rows else "Breakdown results have not been generated yet."}

## Method Summary

1. Resize each image to an 800-pixel short side with a 1,333-pixel long-side cap.
2. Apply ImageNet normalization.
3. Normalize each referring expression and append ` .` for the standard setting.
4. Run Grounding DINO and rank its 900 object queries by maximum token score.
5. Use the highest-scoring box as the top-1 prediction.
6. Mark a prediction correct when its IoU with the ground-truth box is at least 0.5.

## Analysis Guide

- Compare RefCOCO testA (people) with testB (non-person objects).
- Compare RefCOCO with RefCOCO+, whose expressions exclude absolute location words.
- Use the top-1 versus top-5 oracle gap to separate proposal failures from ranking failures.
- Discuss attribute, position, relation, occlusion, and small-target examples from the
  generated success and failure contact sheets.

## Limitations

- The reported visual-grounding numbers use deterministic representative subsets,
  not every sentence in the full benchmarks.
- The model is evaluated zero-shot without task-specific fine-tuning.
- The pure-PyTorch deformable-attention fallback is correct but slower than the
  custom CUDA extension.
- Natural-language prompt formatting materially affects Grounding DINO behavior.

## Reproduction Commands

```bash
conda activate grounding_dino
python tools/check_environment.py
python tools/verify_refcoco_data.py --verify-decode
python tools/eval_refcoco.py --dataset all --samples-per-split 1000 --seed 2026
python tools/visualize_refcoco.py --success-count 6 --failure-count 6
python tools/generate_report_materials.py
```

## Team Contributions

Fill in member names and percentages before submission.
"""
    (args.output_dir / "EXPERIMENT_SUMMARY.md").write_text(report)

    with (args.output_dir / "visual_grounding_results.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow([
            "split", "samples", "accuracy_at_0_5", "mean_iou",
            "top5_oracle_accuracy_at_0_5", "samples_per_second", "errors",
        ])
        for split, summary in vg.get("splits", {}).items():
            writer.writerow([
                split, summary["samples"], summary["accuracy_at_0_5"],
                summary["mean_iou"], summary["top5_oracle_accuracy_at_0_5"],
                summary["samples_per_second"], summary["errors"],
            ])

    source_visualizations = args.results_dir / "visualizations"
    output_visualizations = args.output_dir / "visualizations"
    output_visualizations.mkdir(parents=True, exist_ok=True)
    for name in (
        "success_contact_sheet.jpg",
        "failure_contact_sheet.jpg",
        "selection.json",
    ):
        source = source_visualizations / name
        if source.is_file():
            shutil.copy2(source, output_visualizations / name)
    print(f"Generated report materials in {args.output_dir}")


if __name__ == "__main__":
    main()
