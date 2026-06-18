"""Validate COCO train2014 and RefCOCO-family annotation/image coverage."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, UnidentifiedImageError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.refcoco_utils import (  # noqa: E402
    DATASET_CONFIGS,
    load_dataset,
    resolve_image_path,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refer-root", type=Path, default=Path("data/refer"))
    parser.add_argument("--image-dir", type=Path, default=Path("data/coco/train2014"))
    parser.add_argument("--expected-images", type=int, default=82783)
    parser.add_argument("--verify-decode", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("results/data_validation.json"))
    args = parser.parse_args()

    images = sorted(args.image_dir.glob("*.jpg")) if args.image_dir.is_dir() else []
    referenced_ids = set()
    annotation_report = {}
    for dataset in DATASET_CONFIGS:
        refs, _ = load_dataset(args.refer_root, dataset)
        ids = {int(ref["image_id"]) for ref in refs}
        referenced_ids.update(ids)
        annotation_report[dataset] = {"refs": len(refs), "unique_images": len(ids)}

    missing = [
        str(resolve_image_path(args.image_dir, image_id))
        for image_id in sorted(referenced_ids)
        if not resolve_image_path(args.image_dir, image_id).is_file()
    ]
    decode_errors = []
    if args.verify_decode:
        for path in images:
            try:
                with Image.open(path) as image:
                    image.verify()
            except (OSError, UnidentifiedImageError) as error:
                decode_errors.append({"path": str(path), "error": str(error)})

    report = {
        "image_dir": str(args.image_dir),
        "expected_images": args.expected_images,
        "actual_images": len(images),
        "image_count_ok": len(images) == args.expected_images,
        "referenced_unique_images": len(referenced_ids),
        "missing_referenced_images": missing,
        "decode_errors": decode_errors,
        "annotations": annotation_report,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if not report["image_count_ok"] or missing or decode_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
