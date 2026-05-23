"""
Text Backbone Module
====================
Real BERT via HuggingFace transformers.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional


class TextBackbone(nn.Module):
    """
    Text backbone wrapper using HuggingFace transformers.

    Grounding DINO uses BERT-base by default, with max token length 256.
    Text is tokenized with a BPE scheme.

    Args:
        model_name: HuggingFace model name, e.g. 'bert-base-uncased'
        max_tokens: Maximum number of text tokens (default 256)
        freeze: Whether to freeze text encoder weights
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        max_tokens: int = 256,
        freeze: bool = False,
    ):
        super().__init__()
        self.model_name = model_name
        self.max_tokens = max_tokens

        try:
            from transformers import AutoModel, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.text_encoder = AutoModel.from_pretrained(model_name)
        except ImportError:
            raise ImportError(
                "transformers is required for the text backbone. "
                "Install it: pip install transformers"
            )

        self.hidden_dim = self.text_encoder.config.hidden_size  # 768 for bert-base

        if freeze:
            self._freeze()

    def _freeze(self):
        """Freeze text encoder parameters."""
        for param in self.text_encoder.parameters():
            param.requires_grad = False

    def tokenize(self, texts: list, device: str = "cpu") -> Dict[str, torch.Tensor]:
        """
        Tokenize input texts.

        Args:
            texts: List of text strings
            device: Target device for tensors

        Returns:
            Dictionary with 'input_ids', 'attention_mask', etc.
        """
        encoding = self.tokenizer(
            texts,
            padding="max_length",
            max_length=self.max_tokens,
            truncation=True,
            return_tensors="pt",
        )
        return {k: v.to(device) for k, v in encoding.items()}

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: [B, max_tokens]
            attention_mask: [B, max_tokens]

        Returns:
            text_features: [B, max_tokens, hidden_dim]
        """
        outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return outputs.last_hidden_state  # [B, N_t, D]
