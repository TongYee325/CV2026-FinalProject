# Grounding DINO — Course Project Reproduction

This project reproduces the inference and evaluation pipeline for **Grounding DINO**, an open-vocabulary object detector that locates arbitrary objects in images given text prompts.

> Liu et al., *"Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection"*, ECCV 2024.

**Goal:** Build a working inference pipeline using the authors' pretrained checkpoint. We do **not** train from scratch; the focus is on architecture alignment, checkpoint loading, and benchmark evaluation.

**Key result:** COCO val2017 zero-shot AP = **54.4%** with the official `groundingdino_swinb_cogcoor.pth` checkpoint.

---

## What This Project Does

Grounding DINO is a transformer-based detector. Given an image and a text prompt (class names or a descriptive phrase), it predicts a set of bounding boxes and their matching scores to the text.

The pipeline has three stages:

1. **Dual backbone:** Swin-B extracts multi-scale image features; BERT extracts text features.
2. **Cross-modal transformer:** An encoder fuses image and text features; a decoder refines object queries using both modalities.
3. **Output heads:** A shared MLP predicts boxes, and a parameter-free contrastive head scores each box against text tokens.

We evaluate the pretrained model on:

| Task | Dataset | Metric |
|------|---------|--------|
| Open-vocabulary object detection | COCO 2017 val | AP / AP50 / AP75 |
| Visual grounding (optional) | RefCOCO / RefCOCO+ / RefCOCOg | Accuracy@0.5 |

The project spec requires evaluation on **at least one** public dataset; COCO alone satisfies this requirement.

---

## Project Structure

```
CV2026-FinalProject/
├── core/
│   ├── backbones/                 # Image & text backbones + position encoding
│   │   ├── image_backbone.py      # Custom wrapper: Swin + sine position embeddings
│   │   ├── text_backbone.py       # Custom wrapper: HuggingFace BERT + tokenizer
│   │   ├── swin_transformer.py    # Copied from GroundingDINO-0.1.0-alpha2
│   │   ├── position_encoding.py   # Copied from GroundingDINO-0.1.0-alpha2
│   │   └── nested_tensor.py       # Local compatibility helper
│   └── official_compat/           # Official alpha2 transformer + attention
│       ├── transformer.py         # Encoder-decoder-fusion transformer
│       ├── ms_deform_attn.py      # Multi-scale deformable attention
│       ├── fuse_modules.py        # Bi-directional fusion blocks
│       ├── utils.py               # ContrastiveEmbed, MLP, inverse_sigmoid, ...
│       └── transformer_vanilla.py # Vanilla transformer helper
├── grounding_dino_v2.py           # Main model assembly (custom)
├── tools/
│   ├── load_checkpoint_v2.py      # Checkpoint loader with alpha2 key remapping
│   ├── eval_coco.py               # COCO val2017 evaluation
│   ├── eval_refcoco.py            # RefCOCO evaluation (optional)
│   ├── test_v2_real_image.py      # Single-image sanity check
│   ├── diagnose_coco_image.py     # Compare predictions with COCO GT
│   └── debug_model.py             # Diagnostic script for keys/features/logits
├── visualize.py                   # Draw predictions on images
├── demo_inference.py              # Forward-pass sanity check
├── environment.yml                # Conda environment
├── requirements.txt               # Pip dependencies
├── README.md                      # This file
└── ARCHITECTURE_AND_DEBUG_REPORT.md  # Detailed debugging notes
```

---

## 1. Environment Installation

### 1.1 Create Conda Environment

```bash
cd CV2026-FinalProject
conda env create -f environment.yml
conda activate grounding_dino
```

If your server has a different CUDA version, check with `nvidia-smi` and edit `pytorch-cuda=12.1` in `environment.yml` accordingly before creating.

### 1.2 Verify Installation

```bash
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

Expected output: PyTorch version + `CUDA: True`.

---

## 2. Download Pretrained Checkpoints

We use the official **GroundingDINO_SwinB** checkpoint released with `v0.1.0-alpha2` (~895 MB).

```bash
mkdir -p ./pretrained_weights

wget -P ./pretrained_weights \
  https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth
```

If the direct download is blocked, use the HuggingFace mirror:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

**Important:** This checkpoint corresponds to the older `v0.1.0-alpha2` revision, not the latest `main` branch. The latest `main` added a `bias` parameter to `ContrastiveEmbed` and changed other forward-path details; using it with this checkpoint collapses predictions. Our code is aligned with `alpha2`.

---

## 3. Download Datasets

### 3.1 COCO 2017 (required for OVOD evaluation)

```bash
mkdir -p ./data/coco && cd ./data/coco

# Images
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip && rm val2017.zip

# Annotations
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip && rm annotations_trainval2017.zip
```

Final structure:
```
data/coco/
├── val2017/
└── annotations/
    └── instances_val2017.json
```

### 3.2 RefCOCO / RefCOCO+ / RefCOCOg (optional visual grounding)

RefCOCO requires COCO train2014 images (~13 GB) in addition to the referring-expression annotations. This is **not required** for the course project, but can be added as supplementary visual-grounding results.

```bash
cd ./data
git clone https://github.com/lichengunc/refer.git
```

Follow `refer/README.md` to download the datasets and place them under:

```
data/refer/
├── refcoco/
├── refcoco+/
└── refcocog/
```

---

## 4. Architecture

### 4.1 Design Philosophy

Rather than rewriting the full model, we reuse as much of the official `GroundingDINO-0.1.0-alpha2` source as possible and only write the minimal "glue" needed to connect it to a standard PyTorch inference loop.

The main challenge is that the official repo is packaged as an installable module named `groundingdino` and relies on a custom CUDA extension. We:

1. Copy the relevant official modules into `core/official_compat/`.
2. Rewire their internal imports so they work without installing the `groundingdino` package.
3. Provide custom wrappers for the Swin image backbone and BERT text backbone.
4. Implement a checkpoint loader that maps official key names to our model key names.
5. Fall back to a pure-PyTorch implementation of multi-scale deformable attention when the custom CUDA extension is unavailable.

### 4.2 Components Copied Verbatim from GroundingDINO-0.1.0-alpha2

| Component | File | Source in official repo |
|-----------|------|------------------------|
| Transformer (encoder + decoder + fusion) | `core/official_compat/transformer.py` | `groundingdino/models/GroundingDINO/transformer.py` |
| Multi-scale deformable attention | `core/official_compat/ms_deform_attn.py` | `groundingdino/models/GroundingDINO/ms_deform_attn.py` |
| Fusion blocks (BiAttentionBlock) | `core/official_compat/fuse_modules.py` | `groundingdino/models/GroundingDINO/fuse_modules.py` |
| Utility layers (MLP, ContrastiveEmbed, inverse_sigmoid, etc.) | `core/official_compat/utils.py` | `groundingdino/models/GroundingDINO/utils.py` |
| Vanilla transformer helper | `core/official_compat/transformer_vanilla.py` | `groundingdino/models/GroundingDINO/transformer_vanilla.py` |
| Swin Transformer backbone | `core/backbones/swin_transformer.py` | `groundingdino/models/GroundingDINO/backbone/swin_transformer.py` |
| Sine position encoding | `core/backbones/position_encoding.py` | `groundingdino/models/GroundingDINO/backbone/position_encoding.py` |

We copied these **because they contain the exact forward logic trained into the checkpoint**. Rewriting them (even slightly) breaks weight compatibility, as we initially discovered when the latest `main` code produced AP ~0.

### 4.3 Custom Components We Implemented

| Component | File | Purpose |
|-----------|------|---------|
| Main model assembly | `grounding_dino_v2.py` | Builds backbone, BERT, input projections, transformer, and heads; implements `forward()` and `predict()`. |
| Image backbone wrapper | `core/backbones/image_backbone.py` | Wraps official Swin, computes `PositionEmbeddingSineHW` dynamically, returns feature maps + position embeddings. |
| Text backbone wrapper | `core/backbones/text_backbone.py` | Loads HuggingFace `BertModel`/`BertTokenizer` from cache; used by `grounding_dino_v2.py`. |
| Checkpoint loader V2 | `tools/load_checkpoint_v2.py` | Remaps `backbone.0.*` → `backbone.backbone.*`, skips position-index buffers and unused label encoder. |
| COCO evaluator | `tools/eval_coco.py` | Builds the COCO 80-class caption, runs inference, maps predicted token positions to COCO category IDs, and calls pycocotools. |
| RefCOCO evaluator | `tools/eval_refcoco.py` | Referring-expression accuracy evaluation. |
| Diagnostic scripts | `tools/debug_model.py`, `tools/test_v2_real_image.py`, `tools/diagnose_coco_image.py`, `tools/diagnose_predictions.py` | Sanity checks, GT comparison, and feature/logit inspection. |

### 4.4 Local Compatibility Modules

| File | Purpose |
|------|---------|
| `core/backbones/nested_tensor.py` | Minimal `NestedTensor` class required by the official backbone/position-encoding code. |

### 4.5 Why We Chose the alpha2 Revision

The released `groundingdino_swinb_cogcoor.pth` checkpoint matches the `v0.1.0-alpha2` tag. The latest `main` branch made the following incompatible changes:

| Aspect | Latest `main` | `v0.1.0-alpha2` (ours) |
|--------|---------------|------------------------|
| `ContrastiveEmbed` | Has a learned `bias` parameter | Parameter-free |
| BERT text encoding | Plain `attention_mask` | Uses `text_self_attention_masks` + `position_ids` when `sub_sentence_present=True` |
| Two-stage bbox head | Shared with decoder bbox head | Separate copy by default |
| Two-stage class head | Shared with decoder class head | Separate copy by default |

Using the `main` code with the alpha2 checkpoint gave ~0 AP; switching to alpha2 gave **54.4 AP**.

### 4.6 Checkpoint Key Mapping

| Official checkpoint key | Our model key | Handling |
|------------------------|---------------|----------|
| `backbone.0.*` | `backbone.backbone.*` | Remapped in loader |
| `backbone.1.*` | — | Skipped (parameter-free position encoding) |
| `bert.*` | `bert.*` | Direct match (except `position_ids` buffer) |
| `feat_map.*` | `feat_map.*` | Direct match |
| `input_proj.*` | `input_proj.*` | Direct match |
| `transformer.*` | `transformer.*` | Direct match |
| `bbox_embed.*` / `transformer.decoder.bbox_embed.*` | `bbox_embed.*` / `transformer.decoder.bbox_embed.*` | Direct match, shared parameters |
| `transformer.enc_out_bbox_embed.*` | `transformer.enc_out_bbox_embed.*` | Direct match (separate head) |
| `class_embed` / `transformer.decoder.class_embed` / `transformer.enc_out_class_embed` | Parameter-free `ContrastiveEmbed` | No parameters to load |
| `label_enc.*` | — | Skipped (not used in open-vocab inference) |

Loading the checkpoint reports:

```
Matched keys: 1106 / 1106
Missing keys: 0
Unused keys:  2   # bert.embeddings.position_ids, label_enc.weight
```

---

## 5. Multi-Scale Deformable Attention Fallback

The official repo compiles a custom CUDA kernel for deformable attention. We do **not** build it; instead we use the pure-PyTorch fallback inside `core/official_compat/ms_deform_attn.py`.

- The fallback still runs on GPU tensors (`value.is_cuda`).
- Speed on COCO val2017: ~2.7 images/sec on NVIDIA TITAN V.
- The warning `Failed to load custom C++ ops. Running on CPU mode Only!` is misleading: it means the C++ extension is absent, not that the model runs on CPU. CPU inference would be ~0.1 it/s.

To get full speed, you could build the official CUDA extension, but it is not required for correctness.

---

## 6. Quick Start: Run Inference

### 6.1 Sanity-Check Forward Pass

```bash
python tools/test_v2_real_image.py
```

This loads the checkpoint, runs one COCO image, and prints the top detected boxes.

### 6.2 Visualize Predictions on a Single Image

```bash
python visualize.py \
  --checkpoint ./pretrained_weights/groundingdino_swinb_cogcoor.pth \
  --image ./data/coco/val2017/000000000139.jpg \
  --text "person . car . dog . chair . tv ." \
  --output ./output_vis.jpg \
  --threshold 0.25
```

**Caption format:** separate phrases with ` . `, e.g. `"person . car . dog ."`. Because `sub_sentence_present=True` builds phrase masks from special tokens, single-word prompts without a separator may give no detections.

### 6.3 Compare Predictions with COCO Ground Truth

```bash
python tools/diagnose_coco_image.py --image-id 139 --threshold 0.25
```

This prints GT boxes and model predictions side-by-side for quick validation.

---

## 7. Evaluation

### 7.1 COCO val2017 Zero-Shot Object Detection

Run the full 5000-image evaluation:

```bash
python tools/eval_coco.py \
  --checkpoint ./pretrained_weights/groundingdino_swinb_cogcoor.pth \
  --coco-dir ./data/coco \
  --ann-file ./data/coco/annotations/instances_val2017.json \
  --image-dir ./data/coco/val2017 \
  --threshold 0.05 \
  --output ./tools/coco_results.json
```

For a quick 100-image sanity check, add `--max-images 100`.

### 7.2 RefCOCO Visual Grounding (Optional)

Requires COCO train2014 images and RefCOCO annotations.

```bash
python tools/eval_refcoco.py \
  --checkpoint ./pretrained_weights/groundingdino_swinb_cogcoor.pth \
  --refer-root ./data/refer \
  --dataset all
```

---

## 8. Results

### COCO val2017 Zero-Shot (GroundingDINO Swin-B)

| Metric | Value |
|--------|-------|
| **AP** | **0.544** |
| AP50 | 0.714 |
| AP75 | 0.597 |
| APs | 0.375 |
| APm | 0.589 |
| APl | 0.697 |
| AR@1 | 0.397 |
| AR@10 | 0.666 |
| AR@100 | 0.724 |

Full pycocotools output:

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.544
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.714
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.597
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.375
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.589
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.697
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.397
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.666
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.724
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.558
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.765
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.873
```

These numbers are in the expected range for the `groundingdino_swinb_cogcoor.pth` zero-shot checkpoint.

### RefCOCO

Not evaluated; COCO alone satisfies the project requirement of evaluating on at least one public dataset.

---

## 9. Notes

- **Single-word prompts:** Because the model uses `sub_sentence_present=True`, captions should contain separators (e.g., `"person ."` instead of `"person"`). The COCO evaluation caption naturally satisfies this.
- **Checkpoint compatibility:** Only `groundingdino_swinb_cogcoor.pth` (alpha2) is guaranteed to work. Newer checkpoints from the `main` branch are incompatible.
- **CUDA extension:** The model works without compiling the official CUDA ops. Building them would improve throughput but is not required.

---

## 10. Troubleshooting

### `ModuleNotFoundError: No module named 'timm'` or `'transformers'`

```bash
conda activate grounding_dino
pip install timm transformers
```

### BERT tokenizer not found (server has no internet)

Manually download these files from `https://huggingface.co/bert-base-uncased/tree/main`:
- `config.json`
- `vocab.txt`
- `tokenizer_config.json`
- `tokenizer.json`

Place them in:
```
~/.cache/huggingface/hub/models--bert-base-uncased/snapshots/main/
```

### CUDA out of memory

- Reduce image size in the eval script (default already uses 800px shorter side / 1333px max).
- Ensure batch size is 1.
- Use `CUDA_VISIBLE_DEVICES=N` to select a GPU with free memory.

### Very low AP (~0)

Common causes:
1. Using a checkpoint from the latest `main` branch instead of `v0.1.0-alpha2`.
2. Wrong COCO category ID mapping (fixed in `tools/eval_coco.py`).
3. Model running on CPU because CUDA is unavailable (check `torch.cuda.is_available()`).

---

## 11. Report Checklist

- [ ] Architecture diagram showing copied vs. custom components.
- [ ] Quantitative results table (COCO AP / AP50 / AP75).
- [ ] Qualitative visualizations (`visualize.py` outputs).
- [ ] Discussion of the alpha2 vs. `main` mismatch and why it mattered.
- [ ] Discussion of failure cases.
- [ ] Contribution section per team member.
