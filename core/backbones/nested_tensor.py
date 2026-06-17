"""
Minimal NestedTensor implementation for compatibility with official GroundingDINO code.
"""

import torch


class NestedTensor(object):
    def __init__(self, tensors, mask):
        self.tensors = tensors
        self.mask = mask

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)

    @property
    def shape(self):
        return self.tensors.shape

    def size(self, *args, **kwargs):
        return self.tensors.size(*args, **kwargs)

    def to(self, *args, **kwargs):
        cast_tensor = self.tensors.to(*args, **kwargs)
        cast_mask = self.mask.to(*args, **kwargs) if self.mask is not None else None
        return type(self)(cast_tensor, cast_mask)

    def flatten(self, *args, **kwargs):
        return self.tensors.flatten(*args, **kwargs)

    def unsqueeze(self, *args, **kwargs):
        return type(self)(self.tensors.unsqueeze(*args, **kwargs), self.mask)
