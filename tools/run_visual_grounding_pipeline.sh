#!/usr/bin/env bash
set -euo pipefail

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/tmp/hf_cache}"

python tools/check_environment.py
python tools/verify_refcoco_data.py
python tools/summarize_coco_results.py

python tools/eval_refcoco.py \
  --checkpoint pretrained_weights/groundingdino_swinb_cogcoor.pth \
  --refer-root data/refer \
  --image-dir data/coco/train2014 \
  --dataset all \
  --manifest manifests/refcoco_subset_seed2026.json \
  --output-dir results/refcoco

for mode in period raw; do
  python tools/eval_refcoco.py \
    --checkpoint pretrained_weights/groundingdino_swinb_cogcoor.pth \
    --refer-root data/refer \
    --image-dir data/coco/train2014 \
    --dataset refcoco \
    --splits val \
    --manifest manifests/refcoco_val_ablation_seed2026.json \
    --prompt-mode "${mode}" \
    --output-dir "results/prompt_ablation/${mode}"
done

python tools/visualize_refcoco.py \
  --results-dir results/refcoco \
  --image-dir data/coco/train2014 \
  --success-count 12 \
  --failure-count 12

python tools/generate_report_materials.py \
  --results-dir results/refcoco \
  --ablation-dir results/prompt_ablation \
  --output-dir report_materials
