"""
Text Backbone Module
====================
Wraps a text encoder (e.g. BERT-base) to extract text features.
Grounding DINO uses BERT-base by default, with max token length 256.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional


class TextBackbone(nn.Module):
    """
    Text backbone wrapper.
    
    In the paper, authors use BERT-base from HuggingFace.
    Text is tokenized with BPE scheme.
    
    Args:
        model_name: HuggingFace model name (e.g. 'bert-base-uncased')
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
        
        # TODO: Load actual BERT model from transformers
        # from transformers import AutoModel, AutoTokenizer
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # self.text_encoder = AutoModel.from_pretrained(model_name)
        self.text_encoder = None
        self.tokenizer = None
        
        self.hidden_dim = 768  # BERT-base hidden size
        
        if freeze:
            self._freeze()
    
    def _freeze(self):
        """Freeze text encoder parameters."""
        if self.text_encoder is not None:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
    
    def tokenize(self, texts: list) -> Dict[str, torch.Tensor]:
        """
        Tokenize input texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            Dictionary with 'input_ids', 'attention_mask', etc.
        """
        # TODO: Implement with actual tokenizer
        # return self.tokenizer(
        #     texts,
        #     padding="max_length",
        #     max_length=self.max_tokens,
        #     truncation=True,
        #     return_tensors="pt",
        # )
        B = len(texts)
        return {
            "input_ids": torch.zeros(B, self.max_tokens, dtype=torch.long),
            "attention_mask": torch.ones(B, self.max_tokens, dtype=torch.long),
        }
    
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
        # TODO: Implement actual forward pass
        # outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        # return outputs.last_hidden_state  # [B, N_t, D]
        
        B, N = input_ids.shape
        return torch.zeros(B, N, self.hidden_dim, device=input_ids.device)
