# Grounding DINO Experiment Materials

## Project Positioning

This project reproduces Grounding DINO for two related tasks:

- **Open-vocabulary object detection (OVOD):** detect arbitrary text-specified categories.
- **Visual grounding:** localize the single region described by a natural-language expression.

The implementation uses the official GroundingDINO v0.1.0-alpha2-compatible
Swin-B architecture and checkpoint. No training or fine-tuning is performed.

## OVOD Results: COCO val2017

| Metric | Value |
| --- | --- |
| AP | 0.544 |
| AP50 | 0.714 |
| AP75 | 0.597 |
| APs | 0.375 |
| APm | 0.589 |
| APl | 0.696 |

The full 5,000-image evaluation matches the expected checkpoint behavior.

## Visual Grounding Results

| Dataset split | N | Acc@0.5 | Mean IoU | Top-5 oracle Acc@0.5 | samples/s | errors |
| --- | --- | --- | --- | --- | --- | --- |
| refcoco_unc_val | 1000 | 84.90% | 0.806 | 98.30% | 1.94 | 0 |
| refcoco_unc_testA | 1000 | 89.60% | 0.837 | 99.10% | 1.92 | 0 |
| refcoco_unc_testB | 1000 | 80.40% | 0.760 | 96.70% | 1.93 | 0 |
| refcoco+_unc_val | 1000 | 72.40% | 0.689 | 96.20% | 1.94 | 0 |
| refcoco+_unc_testA | 1000 | 80.60% | 0.762 | 97.90% | 1.95 | 0 |
| refcoco+_unc_testB | 1000 | 66.80% | 0.640 | 95.00% | 1.94 | 0 |
| refcocog_umd_val | 1000 | 79.00% | 0.744 | 97.00% | 1.91 | 0 |
| refcocog_umd_test | 1000 | 79.90% | 0.765 | 97.80% | 1.92 | 0 |

Accuracy@0.5 is the standard top-1 metric. Mean IoU and top-5 oracle accuracy
are diagnostic metrics. The oracle gap measures ranking errors among generated
candidate boxes.

## Prompt Formatting Ablation

| Prompt mode | Split | N | Acc@0.5 | Mean IoU |
| --- | --- | --- | --- | --- |
| period | refcoco_unc_val | 300 | 85.67% | 0.801 |
| raw | refcoco_unc_val | 300 | 34.00% | 0.353 |

## Visual Grounding Breakdown

| Expression group | N | Acc@0.5 | Mean IoU |
| --- | --- | --- | --- |
| attribute | 2965 | 81.69% | 0.769 |
| occlusion | 208 | 66.35% | 0.623 |
| other | 1827 | 72.14% | 0.690 |
| person | 2518 | 84.67% | 0.799 |
| position | 2165 | 84.34% | 0.797 |
| relation | 1955 | 72.48% | 0.692 |

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
python tools/visualize_refcoco.py --success-count 12 --failure-count 12
python tools/generate_report_materials.py
```

## Team Contributions

Fill in member names and percentages before submission.
