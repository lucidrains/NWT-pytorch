import torch
from torch import nn, einsum
from einops import rearrange

class Memcodes(nn.Module):
    def __init__(
        self,
        *,
        dim
    ):
        super().__init__()

    def forward(self, x):
        return x