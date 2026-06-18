"""Evaluate Grounding DINO on reproducible RefCOCO-family sentence subsets."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, ".")

import torch  # noqa: E402
from PIL import Image, UnidentifiedImageError  # noqa: E402
from tqdm import tqdm  # noqa: E402

from grounding_dino_v2 import GroundingDINOV2  # noqa: E402
from tools.load_checkpoint_v2 import load_checkpoint_v2  # noqa: E402
from tools.refcoco_utils import (  # noqa: E402
    DATASET_CONFIGS,
    compute_iou,
    cxcywh_normalized_to_xyxy,
    deterministic_sample,
    expression_tags,
    flatten_sentences,
    load_dataset,
    normalize_expression,
    preprocess_image,
    resolve_image_path,
    xywh_to_xyxy,
)


def selected_datasets(value: str) -> List[str]:
    return list(DATASET_CONFIGS) if value == "all" else [value]


def build_manifest(refer_root: Path, datasets: List[str], samples_per_split: int,
                   seed: int, requested_splits: set[str] | None = None
                   ) -> Dict[str, List[dict]]:
    manifest = {}
    for dataset in datasets:
        refs, annotations = load_dataset(refer_root, dataset)
        split_by = DATASET_CONFIGS[dataset]["split_by"]
        for split in DATASET_CONFIGS[dataset]["splits"]:
            if requested_splits and split not in requested_splits:
                continue
            key = f"{dataset}_{split_by}_{split}"
            samples = flatten_sentences(refs, annotations, split)
            manifest[key] = deterministic_sample(samples, samples_per_split, seed, key)
    return manifest


def save_manifest(manifest: Dict[str, List[dict]], path: Path, seed: int,
                  samples_per_split: int) -> None:
    payload = {
        "seed": seed,
        "samples_per_split": samples_per_split,
        "splits": manifest,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


@torch.no_grad()
def evaluate_split(
    model: GroundingDINOV2,
    samples: List[dict],
    image_dir: Path,
    device: str,
    append_period: bool,
    top_k: int = 5,
) -> tuple[List[dict], dict]:
    model.eval()
    rows = []
    errors = []
    started = time.perf_counter()

    for sample in tqdm(samples, desc="Visual grounding"):
        image_path = resolve_image_path(image_dir, sample["image_id"])
        base_row = {
            **sample,
            "image_file": image_path.name,
            "prompt": normalize_expression(sample["expression"], append_period),
        }
        if sample["gt_box_xywh"] is None:
            errors.append({**base_row, "error": "annotation_missing"})
            rows.append({**base_row, "status": "annotation_missing", "correct": False})
            continue
        if not image_path.is_file():
            errors.append({**base_row, "error": "image_missing"})
            rows.append({**base_row, "status": "image_missing", "correct": False})
            continue

        try:
            image = Image.open(image_path).convert("RGB")
            image_width, image_height = image.size
            tensor = preprocess_image(image).unsqueeze(0).to(device)
        except (OSError, UnidentifiedImageError) as error:
            errors.append({**base_row, "error": f"image_error: {error}"})
            rows.append({**base_row, "status": "image_error", "correct": False})
            continue

        inference_start = time.perf_counter()
        output = model.predict(tensor, [base_row["prompt"]], confidence_threshold=0.0)[0]
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        inference_seconds = time.perf_counter() - inference_start

        boxes = output["boxes"]
        scores = output["scores"]
        gt_xyxy = xywh_to_xyxy(sample["gt_box_xywh"])
        if len(boxes) == 0:
            errors.append({**base_row, "error": "no_predictions"})
            rows.append({
                **base_row,
                "status": "no_predictions",
                "image_width": image_width,
                "image_height": image_height,
                "gt_box_xyxy": gt_xyxy,
                "pred_box_xyxy": None,
                "score": 0.0,
                "iou": 0.0,
                "top5_oracle_iou": 0.0,
                "correct": False,
                "top5_oracle_correct": False,
                "inference_seconds": inference_seconds,
                "tags": expression_tags(
                    sample["expression"], sample["gt_box_xywh"], image_width, image_height
                ),
            })
            continue

        order = torch.argsort(scores, descending=True)
        ranked_boxes = cxcywh_normalized_to_xyxy(
            boxes[order], image_width, image_height
        ).cpu().tolist()
        ranked_scores = scores[order].cpu().tolist()
        top_ious = [compute_iou(box, gt_xyxy) for box in ranked_boxes[:top_k]]
        top1_iou = top_ious[0]
        oracle_iou = max(top_ious)
        rows.append({
            **base_row,
            "status": "ok",
            "image_width": image_width,
            "image_height": image_height,
            "gt_box_xyxy": gt_xyxy,
            "pred_box_xyxy": ranked_boxes[0],
            "score": float(ranked_scores[0]),
            "iou": top1_iou,
            "top5_boxes_xyxy": ranked_boxes[:top_k],
            "top5_scores": [float(value) for value in ranked_scores[:top_k]],
            "top5_ious": top_ious,
            "top5_oracle_iou": oracle_iou,
            "correct": top1_iou >= 0.5,
            "top5_oracle_correct": oracle_iou >= 0.5,
            "inference_seconds": inference_seconds,
            "tags": expression_tags(
                sample["expression"], sample["gt_box_xywh"], image_width, image_height
            ),
        })

    elapsed = time.perf_counter() - started
    count = len(rows)
    summary = {
        "samples": count,
        "successful_inference": sum(row.get("status") == "ok" for row in rows),
        "errors": len(errors),
        "accuracy_at_0_5": sum(bool(row.get("correct")) for row in rows) / count if count else 0.0,
        "mean_iou": sum(float(row.get("iou", 0.0)) for row in rows) / count if count else 0.0,
        "top5_oracle_accuracy_at_0_5": (
            sum(bool(row.get("top5_oracle_correct")) for row in rows) / count if count else 0.0
        ),
        "elapsed_seconds": elapsed,
        "samples_per_second": count / elapsed if elapsed else 0.0,
    }
    return rows, {"summary": summary, "errors": errors}


def write_summary_csv(summaries: Dict[str, dict], output_path: Path) -> None:
    fields = [
        "split", "samples", "successful_inference", "errors", "accuracy_at_0_5",
        "mean_iou", "top5_oracle_accuracy_at_0_5", "elapsed_seconds",
        "samples_per_second",
    ]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for split, summary in summaries.items():
            writer.writerow({"split": split, **summary})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", default="pretrained_weights/groundingdino_swinb_cogcoor.pth"
    )
    parser.add_argument("--refer-root", type=Path, default=Path("data/refer"))
    parser.add_argument("--image-dir", type=Path, default=Path("data/coco/train2014"))
    parser.add_argument(
        "--dataset", default="all", choices=["all", "refcoco", "refcoco+", "refcocog"]
    )
    parser.add_argument("--samples-per-split", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--splits",
        help="Comma-separated split names, for example: val or val,testA,testB",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/refcoco"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--prompt-mode", choices=["period", "raw"], default="period")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets = selected_datasets(args.dataset)
    requested_splits = set(args.splits.split(",")) if args.splits else None
    manifest_path = args.manifest or args.output_dir / "sample_manifest.json"

    manifest_seed = args.seed
    manifest_samples_per_split = args.samples_per_split
    if args.manifest:
        manifest_payload = json.loads(args.manifest.read_text())
        manifest = manifest_payload["splits"]
        manifest_seed = int(manifest_payload.get("seed", args.seed))
        manifest_samples_per_split = int(
            manifest_payload.get("samples_per_split", args.samples_per_split)
        )
        allowed = set()
        for dataset in datasets:
            split_by = DATASET_CONFIGS[dataset]["split_by"]
            allowed.update(
                f"{dataset}_{split_by}_{split}"
                for split in DATASET_CONFIGS[dataset]["splits"]
                if not requested_splits or split in requested_splits
            )
        manifest = {key: value for key, value in manifest.items() if key in allowed}
    else:
        manifest = build_manifest(
            args.refer_root, datasets, args.samples_per_split, args.seed,
            requested_splits=requested_splits,
        )
        save_manifest(manifest, manifest_path, args.seed, args.samples_per_split)

    print(f"Using device: {device}")
    print(f"Evaluating {sum(map(len, manifest.values()))} sentences from {len(manifest)} splits")
    model = GroundingDINOV2().to(device)
    load_checkpoint_v2(model, args.checkpoint, verbose=True)
    model.eval()

    summaries = {}
    for split_key, samples in manifest.items():
        print(f"\n{split_key}: {len(samples)} samples")
        rows, metadata = evaluate_split(
            model, samples, args.image_dir, device,
            append_period=args.prompt_mode == "period",
        )
        split_path = args.output_dir / f"{split_key}.json"
        split_path.write_text(json.dumps(rows, indent=2))
        (args.output_dir / f"{split_key}_errors.json").write_text(
            json.dumps(metadata["errors"], indent=2)
        )
        summaries[split_key] = metadata["summary"]
        print(json.dumps(metadata["summary"], indent=2))

    summary_payload = {
        "checkpoint": args.checkpoint,
        "image_dir": str(args.image_dir),
        "seed": manifest_seed,
        "samples_per_split": manifest_samples_per_split,
        "prompt_mode": args.prompt_mode,
        "splits": summaries,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2))
    write_summary_csv(summaries, args.output_dir / "summary.csv")


if __name__ == "__main__":
    main()
