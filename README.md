# Grounding DINO - Course Project Framework

This is the basic framework for reproducing **Grounding DINO** for the CV 2026 Final Project (Topic 4: Open-Vocabulary Object Detection and Visual Grounding).

## Architecture Overview

Based on the paper (Fig. 3), the framework is organized into the following components:

```
Input Image ──> Image Backbone ──> Vanilla Image Features ──┐
                                                            ├──> Feature Enhancer ──> Enhanced Features
Input Text  ──> Text Backbone  ──> Vanilla Text Features  ──┘                              │
                                                                                          ↓
                                                                    Language-Guided Query Selection ──> Cross-Modality Queries
                                                                                                              │
                                                                                                              ↓
                                                                    Cross-Modality Decoder ──> Output Queries ──> Box + Class Predictions
```

## Project Structure

```
grounding_dino_project/
│
├── core/                                   ← 🔒 SHARED / READ-ONLY
│   ├── backbones/                          # Image (Swin) & Text (BERT) backbones
│   ├── neck/                               # Feature Enhancer (6 layers)
│   ├── query_selection/                    # Language-Guided Query Selection
│   ├── decoder/                            # Cross-Modality Decoder (6 layers)
│   └── utils/                              # Hungarian Matcher
│
├── shared_utils/                           # 🔒 SHARED utilities
│   ├── box_ops.py                          # Box conversions, GIoU
│   └── text_utils.py                       # Tokenization helpers, sub-sentence mask
│
├── configs/                                # 🔒 SHARED configs (extend, don't modify base)
│   ├── base_config.py
│   ├── ovod_config.py                      # ← Team 1 extends this
│   └── grounding_config.py               # ← Team 2 extends this
│
├── grounding_dino.py                       # 🔒 MAIN MODEL - DO NOT MODIFY without team meeting
│
├── ovod/                                   ← 🟢 TEAM 1 WORKSPACE (2 people)
│   ├── datasets/                           # COCO, LVIS, ODinW loaders & evaluators
│   ├── models/                             # OVOD model wrapper + post-processing
│   ├── losses/                             # OVOD losses (contrastive + bbox + giou)
│   └── train_eval/                         # Training & evaluation scripts
│
└── visual_grounding/                       ← 🟠 TEAM 2 WORKSPACE (2 people)
    ├── datasets/                           # RefCOCO/+/g, Flickr30K loaders
    ├── models/                             # REC model wrapper + post-processing
    ├── losses/                             # REC losses
    └── train_eval/                         # Training & evaluation scripts
```

---

## Team Division

### Team 1: Open-Vocabulary Object Detection (OVOD) — 2 people
**Responsibilities:**
- **Person A:** Datasets & Evaluation
  - Implement COCO dataset loader and evaluator (`ovod/datasets/coco_eval.py`)
  - Implement LVIS dataset loader and evaluator (`ovod/datasets/lvis_eval.py`)
  - Run zero-shot evaluation on COCO, LVIS
  - Compute AP, AP50, AP75 metrics

- **Person B:** Training & Text Prompting
  - Implement OVOD-specific text prompt formatting (`ovod/models/ovod_model.py`)
  - Implement sub-sentence level text representation (Sec 3.4)
  - Run OVOD training/fine-tuning loop (`ovod/train_eval/train_ovod.py`)
  - Handle category concatenation for text inputs

**Key Differences from Base Model:**
- Text input is a concatenation of category names separated by `.` (e.g., `"person . car . dog ."`)
- Uses **sub-sentence level** attention mask to block interactions between unrelated categories
- Evaluates on COCO, LVIS, ODinW benchmarks
- Post-processing: threshold-based filtering + mapping to category indices

### Team 2: Visual Grounding (Referring Expression Comprehension) — 2 people
**Responsibilities:**
- **Person C:** Datasets & Evaluation
  - Implement RefCOCO / RefCOCO+ / RefCOCOg dataset loaders (`visual_grounding/datasets/refcoco.py`)
  - Implement Flickr30K Entities loader (optional)
  - Compute REC accuracy (IoU > 0.5)
  - Run evaluation on all REC benchmarks

- **Person D:** Training & Inference
  - Implement REC-specific post-processing (`visual_grounding/models/grounding_model.py`)
  - Run REC training loop (`visual_grounding/train_eval/train_grounding.py`)
  - Handle sentence-level text inputs (single referring expression per sample)
  - Select single best box per sample

**Key Differences from OVOD:**
- Text input is a full **referring expression** (e.g., `"the red car on the left"`)
- Uses **sentence level** text representation (no sub-sentence mask needed)
- Evaluates on RefCOCO/+/g, Flickr30K Entities
- Post-processing: select the single query with the highest score

---

## Critical Rules (No Version Control!)

Since you are all editing on the same server **without git**, follow these rules to avoid conflicts:

1. **NEVER modify `core/` or `grounding_dino.py` without a team agreement.**
   - If you need to change the shared model, discuss it first.
   - One person makes the change while others are aware.

2. **Work ONLY in your assigned directory:**
   - Team 1 → `ovod/`
   - Team 2 → `visual_grounding/`

3. **Configs are semi-shared:**
   - You can modify `ovod_config.py` (Team 1) and `grounding_config.py` (Team 2) freely.
   - **Do not modify `base_config.py`** without notifying everyone.

4. **Coordinate on data downloads:**
   - Put all datasets in a shared `./data/` directory.
   - Don't duplicate large files.

5. **Checkpoint naming convention:**
   - Team 1: `ovod_outputs/ovod_epoch{N}.pth`
   - Team 2: `grounding_outputs/grounding_epoch{N}.pth`

---

## How to Use

### Team 1: Train OVOD
```bash
cd grounding_dino_project
python -m ovod.train_eval.train_ovod --data_root ./data --output_dir ./ovod_outputs
```

### Team 1: Evaluate OVOD
```bash
python -m ovod.train_eval.eval_ovod --checkpoint ./ovod_outputs/ovod_final.pth --data_root ./data
```

### Team 2: Train Visual Grounding
```bash
cd grounding_dino_project
python -m visual_grounding.train_eval.train_grounding --data_root ./data --output_dir ./grounding_outputs
```

### Team 2: Evaluate Visual Grounding
```bash
python -m visual_grounding.train_eval.eval_grounding --checkpoint ./grounding_outputs/grounding_final.pth --data_root ./data
```

---

## What is Already Implemented?

- ✅ Skeleton for all core architecture modules (backbones, feature enhancer, decoder, query selection)
- ✅ Main `GroundingDINO` model class
- ✅ Hungarian matcher skeleton
- ✅ Box operations (IoU, conversions)
- ✅ OVOD and Visual Grounding wrappers, losses, and train/eval scripts (stubs)

## What You Need to Implement?

### High Priority (Everyone)
- [ ] Replace backbone stubs with actual Swin Transformer + BERT (use `timm` + `transformers`)
- [ ] Implement deformable attention in image cross-attention modules
- [ ] Implement proper multi-scale feature handling with level embeddings
- [ ] Integrate actual tokenizer (HuggingFace BERT tokenizer)

### Team 1 Priority
- [ ] COCO dataset loading with pycocotools
- [ ] Sub-sentence level text attention mask
- [ ] Category name concatenation and prompt generation
- [ ] COCO/LVIS zero-shot evaluation pipeline

### Team 2 Priority
- [ ] RefCOCO dataset loading (`.pkl` or `.json` parsing)
- [ ] Sentence-level text representation
- [ ] Single-box selection post-processing for REC
- [ ] RefCOCO/+/g accuracy evaluation

---

## Paper Reference

Liu et al., "Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection", arXiv:2303.05499

Key sections to review:
- **Sec 3.1** — Feature Enhancer
- **Sec 3.2** — Language-Guided Query Selection
- **Sec 3.3** — Cross-Modality Decoder
- **Sec 3.4** — Sub-Sentence Level Text Feature
- **Sec 3.5** — Loss Functions
