## NWT - Pytorch (wip)

Implementation of <a href="https://arxiv.org/abs/2106.04283">NWT</a>, audio-to-video generation, in Pytorch.

<a href="https://next-week-tonight.github.io/NWT/">Generated samples</a>

## Install

```bash
$ pip install nwt-pytorch
```

## Usage

The paper proposes a new discrete latent representation named `Memcodes`, which can be succinctly described as a type of multi-head hard-attention to learned memory (codebook) key / values. They claim the need for less codes and smaller codebook dimension in order to achieve better reconstructions.

```python
import torch
from nwt_pytorch import Memcodes

codebook = Memcodes(
    dim = 512,            # dimension of incoming features (codebook dimension will be dim / heads)
    heads = 8,            # head dimension, which is equivalent ot number of codebooks
    num_codes = 1024,     # number of codes per codebook
    temperature = 1.      # gumbel softmax temperature
)

x = torch.randn(1, 1024, 512)
out, codebook_indices = codebook(x) # (1, 1024, 512), (1, 1024, 8)
# (batch, seq, dimension), (batch, seq, heads)

# reconstruct output from codebook indices (codebook indices are autoregressed out from an attention net in paper)

assert torch.allclose(codebook.get_codes_from_indices(codebook_indices), out)
```

## Citations

```bibtex
@misc{mama2021nwt,
    title   = {NWT: Towards natural audio-to-video generation with representation learning}, 
    author  = {Rayhane Mama and Marc S. Tyndel and Hashiam Kadhim and Cole Clifford and Ragavan Thurairatnam},
    year    = {2021},
    eprint  = {2106.04283},
    archivePrefix = {arXiv},
    primaryClass = {cs.SD}
}
```
