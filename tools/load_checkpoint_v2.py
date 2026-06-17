"""
Checkpoint Loading for GroundingDINO V2
========================================
Simplified loader for v2 model that uses official architecture.
Most keys map directly; only backbone needs remapping.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple


def load_checkpoint_v2(
    model: nn.Module,
    checkpoint_path: str,
    verbose: bool = True,
) -> Tuple[Dict[str, int], List[str], List[str]]:
    """Load official checkpoint into v2 model."""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    print(f"Checkpoint has {len(state_dict)} keys")

    mapped_state = {}
    unused_keys = set(state_dict.keys())
    model_state = model.state_dict()

    for ckpt_key, ckpt_tensor in state_dict.items():
        new_key = None

        # Backbone: backbone.0.* → backbone.backbone.*
        if ckpt_key.startswith("backbone.0."):
            new_key = ckpt_key.replace("backbone.0.", "backbone.backbone.")

        # Position embedding (parameter-free, skip)
        elif ckpt_key.startswith("backbone.1."):
            continue

        # BERT: bert.* → bert.* (direct match)
        elif ckpt_key.startswith("bert."):
            # Skip position_ids buffer
            if "position_ids" in ckpt_key:
                continue
            new_key = ckpt_key

        # Transformer, input_proj, feat_map, bbox_embed: direct match
        elif (
            ckpt_key.startswith("transformer.")
            or ckpt_key.startswith("input_proj.")
            or ckpt_key.startswith("feat_map.")
            or ckpt_key.startswith("bbox_embed.")
        ):
            new_key = ckpt_key

        # Class embed (ContrastiveEmbed bias)
        elif ckpt_key.startswith("class_embed."):
            new_key = ckpt_key

        # Label encoder (not in v2, skip)
        elif ckpt_key.startswith("label_enc."):
            continue

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
                    print(f"  [SHAPE MISMATCH] {ckpt_key} → {new_key}: "
                          f"{ckpt_shape} vs {model_shape}")
        else:
            if verbose and new_key:
                print(f"  [MISSING IN MODEL] {ckpt_key} → {new_key}")

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
    sys.path.insert(0, ".")
    from grounding_dino_v2 import GroundingDINOV2

    model = GroundingDINOV2()
    stats, missing, unused = load_checkpoint_v2(
        model, "pretrained_weights/groundingdino_swinb_cogcoor.pth"
    )
    print(f"\nFinal: {stats}")
