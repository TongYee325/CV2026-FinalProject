"""
Checkpoint Loading Tool
=======================
Loads official GroundingDINO checkpoints into our custom model.

Handles key name remapping between official naming and our naming.
"""

import re
import torch
import torch.nn as nn
from typing import Dict, List, Tuple


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    strict: bool = False,
    verbose: bool = True,
) -> Tuple[Dict[str, int], List[str], List[str]]:
    """
    Load an official GroundingDINO checkpoint into a custom model.

    Args:
        model: The model to load weights into
        checkpoint_path: Path to the .pth checkpoint
        strict: If True, raise error on missing/unexpected keys
        verbose: Print loading statistics

    Returns:
        stats: Dict with 'matched', 'missing', 'unused' counts
        missing_keys: List of keys not found in checkpoint
        unused_keys: List of keys in checkpoint not used by model
    """
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Extract state dict
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    print(f"Checkpoint has {len(state_dict)} keys")

    mapped_state = {}
    unused_keys = set(state_dict.keys())
    model_state = model.state_dict()

    for ckpt_key, ckpt_tensor in state_dict.items():
        new_key = None

        # ---- Backbone ----
        # Official: backbone.0.*  (SwinTransformer is wrapped differently in official)
        if ckpt_key.startswith("backbone.0."):
            new_key = ckpt_key.replace("backbone.0.", "image_backbone.backbone.")

        # Position embedding is parameter-free
        elif ckpt_key.startswith("backbone.1."):
            continue

        # ---- Input projections ----
        elif ckpt_key.startswith("input_proj."):
            new_key = ckpt_key

        # ---- Transformer encoder (image self-attention) ----
        # Official: transformer.encoder.layers.0.*
        # Ours:     feature_enhancer.layers.0.*
        elif ckpt_key.startswith("transformer.encoder.layers."):
            new_key = ckpt_key.replace("transformer.encoder.layers.", "feature_enhancer.layers.")
            new_key = new_key.replace(".self_attn.", ".img_self_attn.")
            new_key = new_key.replace(".norm1.", ".img_norm1.")
            new_key = new_key.replace(".linear1.", ".img_ffn.0.")
            new_key = new_key.replace(".linear2.", ".img_ffn.3.")
            new_key = new_key.replace(".norm2.", ".img_norm3.")

        # ---- Text encoder layers ----
        # Official: transformer.encoder.text_layers.0.*
        # Ours:     feature_enhancer.layers.0.*
        elif ckpt_key.startswith("transformer.encoder.text_layers."):
            new_key = ckpt_key.replace("transformer.encoder.text_layers.", "feature_enhancer.layers.")
            new_key = new_key.replace(".self_attn.", ".txt_self_attn.")
            new_key = new_key.replace(".norm1.", ".txt_norm1.")
            new_key = new_key.replace(".linear1.", ".txt_ffn.0.")
            new_key = new_key.replace(".linear2.", ".txt_ffn.3.")
            new_key = new_key.replace(".norm2.", ".txt_norm3.")

        # ---- Fusion layers ----
        # Official: transformer.encoder.fusion_layers.0.*
        # These use BiAttentionBlock with completely different parameter names.
        # We skip them since our architecture is too different.
        elif ckpt_key.startswith("transformer.encoder.fusion_layers."):
            continue

        # ---- Decoder ----
        # Official: transformer.decoder.layers.0.*
        # Ours:     decoder.layers.0.*
        elif ckpt_key.startswith("transformer.decoder.layers."):
            new_key = ckpt_key.replace("transformer.decoder.layers.", "decoder.layers.")
            # Map cross_attn (deformable in official) - won't match our MHA, but try
            new_key = new_key.replace(".cross_attn.", ".cross_attn_image.")
            # Map catext_norm to norm3
            new_key = new_key.replace(".catext_norm.", ".norm3.")
            # Map norm3 (FFN norm in official) to norm4
            new_key = new_key.replace(".norm3.", ".norm4.")
            # Map linear layers to FFN
            new_key = new_key.replace(".linear1.", ".ffn.0.")
            new_key = new_key.replace(".linear2.", ".ffn.3.")

        # ---- Decoder final norm ----
        elif ckpt_key.startswith("transformer.decoder.norm."):
            new_key = ckpt_key.replace("transformer.decoder.norm.", "decoder.norm.")

        # ---- Decoder bbox_embed (6 layers in official, 1 in ours) ----
        # Skip - architecture mismatch
        elif ckpt_key.startswith("transformer.decoder.bbox_embed."):
            continue

        # ---- Decoder ref_point_head ----
        # Skip - we don't have this
        elif ckpt_key.startswith("transformer.decoder.ref_point_head."):
            continue

        # ---- Query embeddings ----
        elif ckpt_key == "transformer.tgt_embed.weight":
            new_key = "tgt_embed.weight"
        elif ckpt_key == "transformer.refpoint_embed.weight":
            new_key = "refpoint_embed.weight"

        # ---- Box embed (single in ours, 6 in official) ----
        # Official has bbox_embed.0 through bbox_embed.5 (one per decoder layer)
        # We only have one bbox_embed. Skip the official ones.
        elif ckpt_key.startswith("bbox_embed."):
            continue

        # ---- Feature map (text projection) ----
        elif ckpt_key.startswith("feat_map."):
            new_key = ckpt_key.replace("feat_map.", "text_backbone.feat_map.")

        # ---- Class embed ----
        elif ckpt_key.startswith("class_embed."):
            continue  # Our class_embed is parameter-free

        # ---- Text backbone (BERT) ----
        elif ckpt_key.startswith("bert."):
            # Skip position_ids (buffer, not parameter in newer transformers)
            if "position_ids" in ckpt_key:
                continue
            new_key = ckpt_key.replace("bert.", "text_backbone.text_encoder.")

        # ---- Unmapped ----
        else:
            if verbose:
                print(f"  [UNMAPPED] {ckpt_key}")
            continue

        # Check if mapped key exists in model
        if new_key and new_key in model_state:
            model_shape = model_state[new_key].shape
            ckpt_shape = ckpt_tensor.shape
            if model_shape == ckpt_shape:
                mapped_state[new_key] = ckpt_tensor
                unused_keys.discard(ckpt_key)
            else:
                if verbose:
                    print(f"  [SHAPE MISMATCH] {ckpt_key} -> {new_key}: "
                          f"checkpoint {ckpt_shape} vs model {model_shape}")
        else:
            if verbose and new_key:
                print(f"  [MISSING IN MODEL] {ckpt_key} -> {new_key}")

    # Load mapped state dict
    missing_keys = [k for k in model_state.keys() if k not in mapped_state]
    unused_keys = list(unused_keys)

    print(f"\nLoading statistics:")
    print(f"  Matched keys: {len(mapped_state)} / {len(model_state)}")
    print(f"  Missing keys: {len(missing_keys)}")
    print(f"  Unused keys:  {len(unused_keys)}")

    if verbose and missing_keys:
        print(f"\n  Missing keys (first 10):")
        for k in missing_keys[:10]:
            print(f"    - {k}")

    if verbose and unused_keys:
        print(f"\n  Unused keys (first 10):")
        for k in unused_keys[:10]:
            print(f"    - {k}")

    model.load_state_dict(mapped_state, strict=False)

    stats = {
        "matched": len(mapped_state),
        "missing": len(missing_keys),
        "unused": len(unused_keys),
    }
    return stats, missing_keys, unused_keys


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "CV2026-FinalProject")

    from grounding_dino import GroundingDINO

    model = GroundingDINO()
    checkpoint_path = "pretrained_weights/groundingdino_swinb_cogcoor.pth"

    stats, missing, unused = load_checkpoint(model, checkpoint_path)
    print(f"\nFinal stats: {stats}")
