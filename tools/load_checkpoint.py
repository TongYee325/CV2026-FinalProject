"""
Checkpoint Loading Utility
==========================
Maps official Grounding DINO pretrained weights to our model architecture.

Usage:
    from tools.load_checkpoint import load_grounding_dino_checkpoint
    model = GroundingDINO(...)
    load_grounding_dino_checkpoint(model, "./pretrained_weights/groundingdino_swint_ogc.pth")
"""

import torch
import re
from typing import Dict, List, Tuple


def inspect_checkpoint_keys(checkpoint_path: str):
    """
    Print all keys in the checkpoint to understand its structure.
    Run this first before writing the mapper.
    """
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    if "model" in state_dict:
        state_dict = state_dict["model"]

    keys = sorted(state_dict.keys())
    print(f"Total keys in checkpoint: {len(keys)}")
    print("\nFirst 30 keys:")
    for k in keys[:30]:
        print(f"  {k}: {tuple(state_dict[k].shape)}")
    print("\n...")
    print("\nLast 10 keys:")
    for k in keys[-10:]:
        print(f"  {k}: {tuple(state_dict[k].shape)}")

    # Group by prefix
    prefixes = {}
    for k in keys:
        prefix = k.split('.')[0]
        prefixes[prefix] = prefixes.get(prefix, 0) + 1
    print("\nKey prefixes:")
    for p, c in sorted(prefixes.items(), key=lambda x: -x[1]):
        print(f"  {p}: {c} keys")

    return state_dict


def build_key_mapping() -> Dict[str, str]:
    """
    Build a mapping from official checkpoint keys to our model keys.

    Official GroundingDINO naming (approximate, based on DETR-like conventions):
        backbone.0.body.*          -> image_backbone.backbone.*
        bert.*                     -> text_backbone.text_encoder.*
        transformer.encoder.layers.*   -> feature_enhancer.layers.*
        transformer.decoder.layers.*   -> decoder.layers.*
        input_proj.*               -> img_input_proj.*
        bbox_embed.*               -> bbox_head.*
        level_embed                -> level_embed
        query_selection.*          -> query_selection.*
        token_proj / text_proj     -> txt_input_proj

    Returns:
        Dict[official_key_pattern, our_key_pattern] using regex substitution
    """
    # List of (official_regex, our_replacement) tuples
    # Order matters: more specific patterns first
    mappings = [
        # Backbones
        (r"^backbone\.0\.body\.(.*)$", r"image_backbone.backbone.\1"),
        (r"^bert\.(.*)$", r"text_backbone.text_encoder.\1"),

        # Feature enhancer (encoder)
        (r"^transformer\.encoder\.layers\.(\d+)\.self_attn\.(.+)$",
         r"feature_enhancer.layers.\1.img_self_attn.\2"),
        (r"^transformer\.encoder\.layers\.(\d+)\.norm1\.(.+)$",
         r"feature_enhancer.layers.\1.img_norm1.\2"),
        (r"^transformer\.encoder\.layers\.(\d+)\.norm2\.(.+)$",
         r"feature_enhancer.layers.\1.img_norm2.\2"),
        (r"^transformer\.encoder\.layers\.(\d+)\.norm3\.(.+)$",
         r"feature_enhancer.layers.\1.img_norm3.\2"),
        (r"^transformer\.encoder\.layers\.(\d+)\.linear1\.(.+)$",
         r"feature_enhancer.layers.\1.img_ffn.0.\2"),
        (r"^transformer\.encoder\.layers\.(\d+)\.linear2\.(.+)$",
         r"feature_enhancer.layers.\1.img_ffn.3.\2"),

        # NOTE: Official repo has separate text-side encoder layers.
        # Our implementation uses a single shared layer.
        # These weights may not map perfectly — we handle via strict=False.

        # Decoder
        (r"^transformer\.decoder\.layers\.(\d+)\.self_attn\.(.+)$",
         r"decoder.layers.\1.self_attn.\2"),
        (r"^transformer\.decoder\.layers\.(\d+)\.cross_attn_image\.(.+)$",
         r"decoder.layers.\1.cross_attn_image.\2"),
        (r"^transformer\.decoder\.layers\.(\d+)\.cross_attn_text\.(.+)$",
         r"decoder.layers.\1.cross_attn_text.\2"),
        (r"^transformer\.decoder\.layers\.(\d+)\.norm1\.(.+)$",
         r"decoder.layers.\1.norm1.\2"),
        (r"^transformer\.decoder\.layers\.(\d+)\.norm2\.(.+)$",
         r"decoder.layers.\1.norm2.\2"),
        (r"^transformer\.decoder\.layers\.(\d+)\.norm3\.(.+)$",
         r"decoder.layers.\1.norm3.\2"),
        (r"^transformer\.decoder\.layers\.(\d+)\.norm4\.(.+)$",
         r"decoder.layers.\1.norm4.\2"),
        (r"^transformer\.decoder\.layers\.(\d+)\.linear1\.(.+)$",
         r"decoder.layers.\1.ffn.0.\2"),
        (r"^transformer\.decoder\.layers\.(\d+)\.linear2\.(.+)$",
         r"decoder.layers.\1.ffn.3.\2"),

        # Input projections
        (r"^input_proj\.(\d+)\.(.*)$", r"img_input_proj.\1.\2"),

        # Box head
        (r"^bbox_embed\.layers\.0\.(.*)$", r"bbox_head.0.\1"),
        (r"^bbox_embed\.layers\.1\.(.*)$", r"bbox_head.2.\1"),
        (r"^bbox_embed\.layers\.2\.(.*)$", r"bbox_head.4.\1"),

        # Level embed
        (r"^level_embed$", r"level_embed"),

        # Text projection
        (r"^text_proj\.(.*)$", r"txt_input_proj.\1"),
    ]
    return mappings


def remap_checkpoint_keys(
    checkpoint_state_dict: Dict[str, torch.Tensor],
    model_state_dict: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], List[str], List[str]]:
    """
    Remap checkpoint keys to match our model keys.

    Returns:
        new_state_dict: Remapped state dict
        missing_keys: Keys in our model but not in checkpoint
        unexpected_keys: Keys in checkpoint but not mapped to our model
    """
    mappings = build_key_mapping()

    new_state_dict = {}
    used_checkpoint_keys = set()
    our_keys = set(model_state_dict.keys())

    for ckpt_key, tensor in checkpoint_state_dict.items():
        mapped_key = None
        for pattern, replacement in mappings:
            if re.match(pattern, ckpt_key):
                mapped_key = re.sub(pattern, replacement, ckpt_key)
                break

        if mapped_key is not None and mapped_key in our_keys:
            # Verify shape match
            expected_shape = model_state_dict[mapped_key].shape
            if tensor.shape == expected_shape:
                new_state_dict[mapped_key] = tensor
                used_checkpoint_keys.add(ckpt_key)
            else:
                print(f"  Shape mismatch for {ckpt_key} -> {mapped_key}: "
                      f"checkpoint {tuple(tensor.shape)} vs model {tuple(expected_shape)}")

    missing_keys = list(our_keys - set(new_state_dict.keys()))
    unexpected_keys = list(set(checkpoint_state_dict.keys()) - used_checkpoint_keys)

    return new_state_dict, missing_keys, unexpected_keys


def load_grounding_dino_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    strict: bool = False,
    verbose: bool = True,
):
    """
    Load official Grounding DINO checkpoint into our model.

    Args:
        model: Your GroundingDINO instance
        checkpoint_path: Path to .pth file
        strict: If False, ignores missing/unexpected keys (recommended for partial loading)
        verbose: Print loading statistics
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Official checkpoints sometimes wrap the state dict under "model" key
    if "model" in checkpoint:
        checkpoint_state_dict = checkpoint["model"]
    else:
        checkpoint_state_dict = checkpoint

    model_state_dict = model.state_dict()

    # Try direct load first (maybe keys already match)
    try:
        model.load_state_dict(checkpoint_state_dict, strict=True)
        if verbose:
            print(f"Loaded checkpoint directly (exact key match): {checkpoint_path}")
        return
    except RuntimeError:
        pass  # Keys don't match, need remapping

    # Remap keys
    new_state_dict, missing, unexpected = remap_checkpoint_keys(
        checkpoint_state_dict, model_state_dict
    )

    # Load remapped weights
    model.load_state_dict(new_state_dict, strict=False)

    if verbose:
        print(f"Loaded checkpoint (with remapping): {checkpoint_path}")
        print(f"  Matched keys: {len(new_state_dict)} / {len(model_state_dict)}")
        print(f"  Missing keys: {len(missing)}")
        if missing and len(missing) <= 20:
            for k in missing:
                print(f"    - {k}")
        elif missing:
            print(f"    (first 20): {missing[:20]}")
        print(f"  Unused checkpoint keys: {len(unexpected)}")
        if unexpected and len(unexpected) <= 20:
            for k in unexpected:
                print(f"    - {k}")
        elif unexpected:
            print(f"    (first 20): {unexpected[:20]}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to official .pth checkpoint")
    args = parser.parse_args()

    print("Inspecting checkpoint keys...\n")
    inspect_checkpoint_keys(args.checkpoint)
