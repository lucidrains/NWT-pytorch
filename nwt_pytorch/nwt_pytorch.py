import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, repeat
from einops.layers.torch import EinMix as Mix

# helper functions

def batched_index_select(values, indices, dim = 1):
    value_dims = values.shape[(dim + 1):]
    values_shape, indices_shape = map(lambda t: list(t.shape), (values, indices))
    indices = indices[(..., *((None,) * len(value_dims)))]
    indices = indices.expand(*((-1,) * len(indices_shape)), *value_dims)
    value_expand_len = len(indices_shape) - (dim + 1)
    values = values[(*((slice(None),) * dim), *((None,) * value_expand_len), ...)]

    value_expand_shape = [-1] * len(values.shape)
    expand_slice = slice(dim, (dim + value_expand_len))
    value_expand_shape[expand_slice] = indices.shape[expand_slice]
    values = values.expand(*value_expand_shape)

    dim += value_expand_len
    return values.gather(dim, indices)

# main classes

class Memcodes(nn.Module):
    def __init__(
        self,
        *,
        dim,
        num_codes,
        heads = 8,
        temperature = 1.,
    ):
        super().__init__()
        assert (dim % heads) == 0, 'dimension must be divisible by number of heads'
        self.heads = heads
        self.dim = dim
        self.scale = (dim // heads) ** -0.5
        self.temperature = temperature

        num_codebooks = heads
        codebook_dim = dim // heads

        self.codes = nn.Parameter(torch.randn(num_codebooks, num_codes, codebook_dim))
        self.to_k = Mix('h n d -> h n c', weight_shape = 'h d c', h = heads, d = codebook_dim, c = codebook_dim)
        self.to_v = Mix('h n d -> h n c', weight_shape = 'h d c', h = heads, d = codebook_dim, c = codebook_dim)

    def get_codes_from_indices(self, codebook_indices, *, merge_output_heads = True):
        batch = codebook_indices.shape[0]

        values = self.to_v(self.codes)
        values = repeat(values, 'h n d -> b h n d', b = batch)

        out = batched_index_select(values, codebook_indices, dim = 2)

        if not merge_output_heads:
            return out

        return rearrange(out, 'b h n d -> b n (h d)')

    def forward(self, x, *, merge_output_heads = True):
        assert x.shape[-1] == self.dim

        # split out heads

        q = rearrange(x, 'b n (h d) -> b h n d', h = self.heads)

        q = q * self.scale

        # get key / values of codes

        k, v = self.to_k(self.codes), self.to_v(self.codes)

        # straight through gumbel softmax

        logits = einsum('b h i d, h j d -> b h i j', q, k)

        if self.training:
            attn = F.gumbel_softmax(logits, tau = self.temperature, dim = -1, hard = True)
            codebook_indices = attn.argmax(dim = -1)
        else:
            codebook_indices = logits.argmax(dim = -1)
            attn = F.one_hot().float()

        out = einsum('b h i j, h j d -> b h i d', attn, v)

        if not merge_output_heads:
            return out, codebook_indices

        # merge heads if specified

        out = rearrange(out, 'b h n d -> b n (h d)')
        return out, codebook_indices
