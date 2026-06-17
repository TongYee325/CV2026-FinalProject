"""Debug script to identify why predictions are garbage."""
import sys
sys.path.insert(0, ".")

import torch
from grounding_dino_v2 import GroundingDINOV2, generate_masks_with_special_tokens_and_transfer_map
from tools.load_checkpoint_v2 import load_checkpoint_v2
from PIL import Image
from torchvision import transforms

device = "cuda" if torch.cuda.is_available() else "cpu"

model = GroundingDINOV2().to(device)
load_checkpoint_v2(model, "pretrained_weights/groundingdino_swinb_cogcoor.pth", verbose=False)
model.eval()

# ------------------------------------------------------------------
# Key alignment check
# ------------------------------------------------------------------
ckpt = torch.load("pretrained_weights/groundingdino_swinb_cogcoor.pth", map_location="cpu", weights_only=False)
state = ckpt["model"] if "model" in ckpt else ckpt

model_keys = set(model.state_dict().keys())
ckpt_keys = set(state.keys())
missing = sorted(model_keys - ckpt_keys)
unused = sorted(ckpt_keys - model_keys)

print("=== Key alignment ===")
print(f"  Model keys: {len(model_keys)}")
print(f"  Checkpoint keys: {len(ckpt_keys)}")
print(f"  Missing model keys ({len(missing)}):")
for k in missing:
    print(f"    {k}")
print(f"  Unused checkpoint keys ({len(unused)}):")
for k in unused:
    print(f"    {k}")

# Load one image
image = Image.open("data/coco/val2017/000000000139.jpg").convert("RGB")
orig_w, orig_h = image.size

# Resize + normalize (official)
scale = 800 / min(orig_w, orig_h)
if max(orig_w, orig_h) * scale > 1333:
    scale = 1333 / max(orig_w, orig_h)
new_w, new_h = int(orig_w * scale), int(orig_h * scale)
image = image.resize((new_w, new_h), Image.Resampling.BILINEAR)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
img_tensor = transform(image).unsqueeze(0).to(device)

caption = "person . bicycle . car . motorcycle . airplane . bus . train . truck . boat"

with torch.no_grad():
    # Tokenize
    enc = model.tokenizer(caption, padding="longest", return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    # 1. Check backbone features
    features, pos_embeds = model.backbone(img_tensor)
    print("=== Backbone ===")
    for i, f in enumerate(features):
        print(f"  features[{i}]: shape={f.shape}, mean={f.mean():.4f}, std={f.std():.4f}, min={f.min():.4f}, max={f.max():.4f}")

    # 2. Check text features (mirror model.forward text encoding)
    tokenized = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": torch.zeros_like(input_ids),
    }
    text_self_attention_masks, position_ids, _ = generate_masks_with_special_tokens_and_transfer_map(
        tokenized, model.specical_tokens, model.tokenizer
    )
    if model.sub_sentence_present:
        tokenized_for_encoder = {k: v for k, v in tokenized.items() if k != "attention_mask"}
        tokenized_for_encoder["attention_mask"] = text_self_attention_masks
        tokenized_for_encoder["position_ids"] = position_ids
    else:
        tokenized_for_encoder = tokenized
    text_outputs = model.bert(**tokenized_for_encoder)
    encoded_text = model.feat_map(text_outputs.last_hidden_state)
    print("\n=== Text ===")
    print(f"  encoded_text: shape={encoded_text.shape}, mean={encoded_text.mean():.4f}, std={encoded_text.std():.4f}")
    print(f"  last_hidden_state: mean={text_outputs.last_hidden_state.mean():.4f}, std={text_outputs.last_hidden_state.std():.4f}")
    print(f"  feat_map weight: mean={model.feat_map.weight.mean():.4f}, std={model.feat_map.weight.std():.4f}")
    print(f"  feat_map bias: mean={model.feat_map.bias.mean():.4f}, std={model.feat_map.bias.std():.4f}")
    print(f"  checkpoint feat_map weight: mean={state['feat_map.weight'].mean():.4f}, std={state['feat_map.weight'].std():.4f}")
    print(f"  checkpoint feat_map bias: mean={state['feat_map.bias'].mean():.4f}, std={state['feat_map.bias'].std():.4f}")

    # 3. Run full forward
    outputs = model.forward(img_tensor, input_ids, attention_mask)
    logits = outputs["pred_logits"][0]   # [900, num_tokens]
    boxes = outputs["pred_boxes"][0]     # [900, 4]

    print("\n=== Outputs ===")
    finite_mask = torch.isfinite(logits)
    finite_logits = logits[finite_mask]
    print(f"  logits: shape={logits.shape}")
    print(f"  finite logits: {finite_mask.sum().item()} / {logits.numel()}")
    if finite_logits.numel() > 0:
        print(f"  finite logits mean={finite_logits.mean():.4f}, std={finite_logits.std():.4f}, min={finite_logits.min():.4f}, max={finite_logits.max():.4f}")
    else:
        print("  all logits are -inf")
    scores = logits.sigmoid()
    print(f"  scores: mean={scores.mean():.4f}, top10={scores.max(-1)[0].topk(10)[0].tolist()}")
    print(f"  boxes: mean={boxes.mean():.4f}, std={boxes.std():.4f}")

    # Inspect per-token text features
    print("\n=== Text per-token ===")
    for t in range(encoded_text.shape[1]):
        tok = encoded_text[0, t]
        print(f"  token {t:2d}: std={tok.std():.4f}, max={tok.max():.4f}, min={tok.min():.4f}")

    # 4. Check class_embed / ContrastiveEmbed
    print("\n=== class_embed ===")
    for i, ce in enumerate(model.class_embed):
        params = list(ce.parameters())
        if params:
            p = params[0]
            print(f"  class_embed[{i}] param: shape={p.shape}, mean={p.mean():.6f}, std={p.std():.6f}, min={p.min():.6f}, max={p.max():.6f}")
        else:
            print(f"  class_embed[{i}]: parameter-free")

    # 5. Inspect checkpoint keys
    print("\n=== Checkpoint text-related keys ===")
    for k in sorted(state.keys()):
        if any(x in k for x in ["feat_map", "bert", "class_embed"]):
            t = state[k]
            if t.dtype.is_floating_point:
                print(f"  {k}: shape={t.shape}, mean={t.mean():.6f}, std={t.std():.6f}")
            else:
                print(f"  {k}: shape={t.shape}, dtype={t.dtype}")

    # 6. Check bbox_embed structure
    bbox_keys = [k for k in state.keys() if k.startswith("bbox_embed")]
    print(f"\n=== Checkpoint bbox_embed keys ({len(bbox_keys)}) ===")
    for k in bbox_keys[:6]:
        print(f"  {k}: shape={state[k].shape}")
