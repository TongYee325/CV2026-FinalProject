"""Evaluate a saved COCO prediction JSON and persist machine-readable metrics."""

import argparse
import json
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotations",
        default="data/coco/annotations/instances_val2017.json",
    )
    parser.add_argument("--predictions", default="tools/coco_results_full.json")
    parser.add_argument("--output", type=Path, default=Path("results/coco_summary.json"))
    args = parser.parse_args()

    coco_gt = COCO(args.annotations)
    coco_dt = coco_gt.loadRes(args.predictions)
    evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    names = [
        "AP", "AP50", "AP75", "APs", "APm", "APl",
        "AR1", "AR10", "AR100", "ARs", "ARm", "ARl",
    ]
    metrics = {name: float(value) for name, value in zip(names, evaluator.stats)}
    payload = {
        "annotations": args.annotations,
        "predictions": args.predictions,
        "prediction_count": len(coco_dt.dataset["annotations"]),
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
