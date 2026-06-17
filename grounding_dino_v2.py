"""
GroundingDINO V2 — Official Architecture
=========================================
This model uses the official GroundingDINO transformer architecture
copied into our codebase for maximum checkpoint compatibility.

Compared to our custom model (grounding_dino.py), this uses:
  - Official MSDeformAttn (deformable attention)
  - Official BiAttentionBlock (fusion layers)
  - Official DeformableTransformerEncoder/Decoder
  - Language-guided query selection
  - Per-layer bbox/class embeddings
  - Level embeddings

Usage:
    from grounding_dino_v2 import GroundingDINOV2
    model = GroundingDINOV2()
    load_checkpoint(model, "groundingdino_swinb_cogcoor.pth")
"""

import copy
import math
import sys
import os
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def inverse_sigmoid(x, eps=1e-3):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)

# ------------------------------------------------------------------
# Use local copies of official modules (pure PyTorch fallback)
# ------------------------------------------------------------------
from core.official_compat.ms_deform_attn import (
    MultiScaleDeformableAttention as MSDeformAttn,
)
from core.official_compat.fuse_modules import BiAttentionBlock
from core.official_compat.transformer import Transformer
from core.official_compat.utils import MLP, ContrastiveEmbed


def generate_masks_with_special_tokens_and_transfer_map(
    tokenized, special_tokens_list, tokenizer
):
    """Generate attention mask between each pair of special tokens."""
    input_ids = tokenized["input_ids"]
    bs, num_token = input_ids.shape
    special_tokens_mask = torch.zeros((bs, num_token), device=input_ids.device).bool()
    for special_token in special_tokens_list:
        special_tokens_mask |= input_ids == special_token
    idxs = torch.nonzero(special_tokens_mask)
    attention_mask = (
        torch.eye(num_token, device=input_ids.device).bool().unsqueeze(0).repeat(bs, 1, 1)
    )
    position_ids = torch.zeros((bs, num_token), device=input_ids.device)
    cate_to_token_mask_list = [[] for _ in range(bs)]
    previous_col = 0
    for i in range(idxs.shape[0]):
        row, col = idxs[i]
        if (col == 0) or (col == num_token - 1):
            attention_mask[row, col, col] = True
            position_ids[row, col] = 0
        else:
            attention_mask[row, previous_col + 1 : col + 1, previous_col + 1 : col + 1] = True
            position_ids[row, previous_col + 1 : col + 1] = torch.arange(
                0, col - previous_col, device=input_ids.device
            )
            c2t_maski = torch.zeros((num_token), device=input_ids.device).bool()
            c2t_maski[previous_col + 1 : col] = True
            cate_to_token_mask_list[row].append(c2t_maski)
        previous_col = col
    cate_to_token_mask_list_out = []
    for cate_to_token_mask_listi in cate_to_token_mask_list:
        if len(cate_to_token_mask_listi) > 0:
            cate_to_token_mask_list_out.append(torch.stack(cate_to_token_mask_listi, dim=0))
        else:
            cate_to_token_mask_list_out.append(
                torch.zeros((0, num_token), device=input_ids.device, dtype=torch.bool)
            )
    return attention_mask, position_ids.to(torch.long), cate_to_token_mask_list_out

from core.backbones.image_backbone import ImageBackbone
from core.backbones.text_backbone import TextBackbone


class GroundingDINOV2(nn.Module):
    """
    GroundingDINO V2 using official transformer architecture.

    Args:
        image_backbone: Name of Swin backbone (default "swin_B_384_22k")
        text_backbone: Name of BERT model (default "bert-base-uncased")
        d_model: Hidden dimension (256)
        num_queries: Number of queries (900)
        num_feature_levels: Number of feature levels (4)
        num_encoder_layers: Number of encoder layers (6)
        num_decoder_layers: Number of decoder layers (6)
        dropout: Dropout rate
    """

    def __init__(
        self,
        image_backbone: str = "swin_B_384_22k",
        text_backbone: str = "bert-base-uncased",
        d_model: int = 256,
        num_queries: int = 900,
        num_feature_levels: int = 4,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dropout: float = 0.0,
        sub_sentence_present: bool = True,
        dec_pred_bbox_embed_share: bool = True,
        two_stage_bbox_embed_share: bool = False,
        two_stage_class_embed_share: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_queries = num_queries
        self.num_feature_levels = num_feature_levels
        self.max_text_len = 256
        self.sub_sentence_present = sub_sentence_present
        self.dec_pred_bbox_embed_share = dec_pred_bbox_embed_share
        self.two_stage_bbox_embed_share = two_stage_bbox_embed_share
        self.two_stage_class_embed_share = two_stage_class_embed_share

        # ---- Backbones ----
        self.backbone = ImageBackbone(
            model_name=image_backbone,
            pretrained=False,
            out_indices=[1, 2, 3],
        )
        self.tokenizer = TextBackbone(model_name=text_backbone, max_tokens=256).tokenizer
        self.bert = TextBackbone(model_name=text_backbone, max_tokens=256).text_encoder

        # Text projection
        self.feat_map = nn.Linear(self.bert.config.hidden_size, d_model, bias=True)
        nn.init.constant_(self.feat_map.bias.data, 0)
        nn.init.xavier_uniform_(self.feat_map.weight.data)

        # Special tokens for mask generation
        self.specical_tokens = self.tokenizer.convert_tokens_to_ids(
            ["[CLS]", "[SEP]", ".", "?"]
        )

        # ---- Input projections ----
        in_channels = self.backbone.hidden_dims  # [256, 512, 1024]
        self.input_proj = nn.ModuleList()
        for ch in in_channels:
            self.input_proj.append(
                nn.Sequential(
                    nn.Conv2d(ch, d_model, kernel_size=1),
                    nn.GroupNorm(32, d_model),
                )
            )
        # Extra scale (downsampled)
        self.input_proj.append(
            nn.Sequential(
                nn.Conv2d(in_channels[-1], d_model, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(32, d_model),
            )
        )

        # ---- Transformer ----
        self.transformer = Transformer(
            d_model=d_model,
            nhead=8,
            num_queries=num_queries,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=2048,
            dropout=dropout,
            activation="relu",
            normalize_before=False,
            return_intermediate_dec=True,
            query_dim=4,
            num_patterns=0,
            num_feature_levels=num_feature_levels,
            enc_n_points=4,
            dec_n_points=4,
            learnable_tgt_init=True,
            two_stage_type="standard",
            embed_init_tgt=True,
            use_text_enhancer=True,
            use_fusion_layer=True,
            use_text_cross_attention=True,
            text_dropout=dropout,
            fusion_dropout=dropout,
            fusion_droppath=0.0,
        )

        # ---- Output heads ----
        # Class embed: parameter-free contrastive head
        _class_embed = ContrastiveEmbed()

        # Box embed: MLP per decoder layer
        _bbox_embed = MLP(d_model, d_model, 4, 3)
        nn.init.constant_(_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_bbox_embed.layers[-1].bias.data, 0)

        # Share bbox embed across layers (official default)
        if dec_pred_bbox_embed_share:
            box_embed_layerlist = [_bbox_embed for _ in range(num_decoder_layers)]
        else:
            box_embed_layerlist = [copy.deepcopy(_bbox_embed) for _ in range(num_decoder_layers)]
        class_embed_layerlist = [_class_embed for _ in range(num_decoder_layers)]

        self.bbox_embed = nn.ModuleList(box_embed_layerlist)
        self.class_embed = nn.ModuleList(class_embed_layerlist)
        self.transformer.decoder.bbox_embed = self.bbox_embed
        self.transformer.decoder.class_embed = self.class_embed

        # For two-stage query selection: encoder output bbox + class heads
        if two_stage_bbox_embed_share:
            assert dec_pred_bbox_embed_share
            self.transformer.enc_out_bbox_embed = _bbox_embed
        else:
            self.transformer.enc_out_bbox_embed = copy.deepcopy(_bbox_embed)

        if two_stage_class_embed_share:
            assert dec_pred_bbox_embed_share
            self.transformer.enc_out_class_embed = _class_embed
        else:
            self.transformer.enc_out_class_embed = copy.deepcopy(_class_embed)

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            images: [B, 3, H, W]
            input_ids: [B, L]
            attention_mask: [B, L]

        Returns:
            Dict with 'pred_logits' and 'pred_boxes'
        """
        B = images.shape[0]

        # ---- Extract image features ----
        features, pos_embeds = self.backbone(images)

        # Input projections
        srcs = []
        masks = []
        for i, (feat, pos) in enumerate(zip(features, pos_embeds)):
            srcs.append(self.input_proj[i](feat))
            masks.append(
                torch.zeros(feat.shape[0], feat.shape[2], feat.shape[3], device=feat.device, dtype=torch.bool)
            )
        # Extra scale
        extra_src = self.input_proj[-1](features[-1])
        extra_mask = torch.zeros(
            extra_src.shape[0], extra_src.shape[2], extra_src.shape[3], device=extra_src.device, dtype=torch.bool
        )
        extra_pos = F.interpolate(
            pos_embeds[-1], size=extra_src.shape[-2:], mode="bilinear", align_corners=False
        )
        srcs.append(extra_src)
        masks.append(extra_mask)
        pos_embeds.append(extra_pos)

        # ---- Extract text features ----
        # Build tokenized dict with token_type_ids (needed by BERT)
        tokenized = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": torch.zeros_like(input_ids),
        }

        # Generate text self-attention masks and position ids
        text_self_attention_masks, position_ids, _ = (
            generate_masks_with_special_tokens_and_transfer_map(
                tokenized, self.specical_tokens, self.tokenizer
            )
        )

        # BERT forward: use sub-sentence attention mask when enabled
        if self.sub_sentence_present:
            tokenized_for_encoder = {
                k: v for k, v in tokenized.items() if k != "attention_mask"
            }
            tokenized_for_encoder["attention_mask"] = text_self_attention_masks
            tokenized_for_encoder["position_ids"] = position_ids
        else:
            tokenized_for_encoder = tokenized

        text_outputs = self.bert(**tokenized_for_encoder)
        encoded_text = self.feat_map(text_outputs.last_hidden_state)
        text_token_mask = attention_mask.bool()

        # Truncate to max length
        if encoded_text.shape[1] > self.max_text_len:
            encoded_text = encoded_text[:, : self.max_text_len, :]
            text_token_mask = text_token_mask[:, : self.max_text_len]
            position_ids = position_ids[:, : self.max_text_len]
            text_self_attention_masks = text_self_attention_masks[:, : self.max_text_len, : self.max_text_len]
            tokenized["token_type_ids"] = tokenized["token_type_ids"][:, : self.max_text_len]

        text_dict = {
            "encoded_text": encoded_text,
            "text_token_mask": text_token_mask,
            "position_ids": position_ids,
            "text_self_attention_masks": text_self_attention_masks,
        }

        # ---- Run transformer ----
        hs, references, hs_enc, ref_enc, init_box_proposal = self.transformer(
            srcs=srcs,
            masks=masks,
            refpoint_embed=None,
            pos_embeds=pos_embeds,
            tgt=None,
            text_dict=text_dict,
        )

        # hs: list of [B, N_q, D] tensors, length = n_dec
        # references: list of [B, N_q, 4] tensors, length = n_dec+1

        # ---- Apply output heads per layer ----
        outputs_classes = []
        outputs_coords = []

        for layer_cls_embed, layer_bbox_embed, layer_hs, layer_ref_sig in zip(
            self.class_embed, self.bbox_embed, hs, references[:-1]
        ):
            # Class logits
            outputs_class = layer_cls_embed(layer_hs, text_dict)
            outputs_classes.append(outputs_class)

            # Boxes: delta from reference points
            tmp = layer_bbox_embed(layer_hs)
            if layer_ref_sig.shape[-1] == 4:
                tmp += inverse_sigmoid(layer_ref_sig)
            else:
                tmp[..., :2] += inverse_sigmoid(layer_ref_sig)
            outputs_coord = tmp.sigmoid()
            outputs_coords.append(outputs_coord)

        outputs_classes = torch.stack(outputs_classes)  # [n_dec, B, N_q]
        outputs_coords = torch.stack(outputs_coords)    # [n_dec, B, N_q, 4]

        return {
            "pred_logits": outputs_classes[-1],    # [B, N_q]
            "pred_boxes": outputs_coords[-1],      # [B, N_q, 4]
            "aux_outputs": [
                {"pred_logits": outputs_classes[i], "pred_boxes": outputs_coords[i]}
                for i in range(outputs_classes.shape[0] - 1)
            ],
        }

    @torch.no_grad()
    def predict(
        self,
        images: torch.Tensor,
        captions: List[str],
        confidence_threshold: float = 0.3,
    ) -> List[Dict[str, torch.Tensor]]:
        """Convenience method for inference with text captions."""
        self.eval()
        enc = self.tokenizer(
            captions,
            padding="longest",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(images.device)
        attention_mask = enc["attention_mask"].to(images.device)

        outputs = self.forward(images, input_ids, attention_mask)

        results = []
        for i in range(images.shape[0]):
            logits = outputs["pred_logits"][i]          # [nq, num_text_tokens]
            boxes = outputs["pred_boxes"][i]            # [nq, 4]

            scores, labels = logits.sigmoid().max(-1)   # [nq], [nq]
            keep = scores > confidence_threshold

            results.append({
                "boxes": boxes[keep],
                "scores": scores[keep],
                "labels": labels[keep],
            })
        return results
