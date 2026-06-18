"""Create deterministic, reviewable RefCOCO-family sample manifests."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.eval_refcoco import build_manifest, save_manifest, selected_datasets  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refer-root", type=Path, default=Path("data/refer"))
    parser.add_argument(
        "--dataset", default="all", choices=["all", "refcoco", "refcoco+", "refcocog"]
    )
    parser.add_argument("--splits")
    parser.add_argument("--samples-per-split", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    requested_splits = set(args.splits.split(",")) if args.splits else None
    manifest = build_manifest(
        args.refer_root,
        selected_datasets(args.dataset),
        args.samples_per_split,
        args.seed,
        requested_splits=requested_splits,
    )
    save_manifest(manifest, args.output, args.seed, args.samples_per_split)
    print(f"Saved {sum(map(len, manifest.values()))} samples to {args.output}")


if __name__ == "__main__":
    main()
