"""
Text Utility Functions
======================
Helpers for text tokenization and prompt formatting.
"""

import torch
from typing import List, Dict


def prepare_text_inputs(
    texts: List[str],
    tokenizer,
    max_tokens: int = 256,
) -> Dict[str, torch.Tensor]:
    """
    Tokenize a list of text strings.
    
    Args:
        texts: List of text strings
        tokenizer: HuggingFace tokenizer instance
        max_tokens: Maximum sequence length
        
    Returns:
        Dict with input_ids and attention_mask tensors
    """
    return tokenizer(
        texts,
        padding="max_length",
        max_length=max_tokens,
        truncation=True,
        return_tensors="pt",
    )


def create_subsentence_mask(
    input_ids: torch.Tensor,
    category_separators: List[int],
    pad_token_id: int = 0,
) -> torch.Tensor:
    """
    Create sub-sentence level attention mask to block interactions between
    unrelated category names.
    
    As described in Sec. 3.4 and Fig. 4 of the paper.
    
    Args:
        input_ids: [B, N_t] token ids
        category_separators: List of token ids that separate categories (e.g. [SEP], [DOT])
        pad_token_id: Token id for padding
        
    Returns:
        attention_mask: [B, N_t, N_t] where False means allow attention
    """
    B, N = input_ids.shape
    mask = torch.zeros(B, N, N, dtype=torch.bool)
    
    for b in range(B):
        # Find boundaries between categories
        sep_positions = [0]
        for i in range(N):
            if input_ids[b, i].item() in category_separators or input_ids[b, i].item() == pad_token_id:
                sep_positions.append(i)
        sep_positions.append(N)
        
        # Allow attention only within each sub-sentence
        for i in range(len(sep_positions) - 1):
            start = sep_positions[i]
            end = sep_positions[i + 1]
            mask[b, start:end, start:end] = True
    
    return mask
