import math
import torch
from torch import nn
import torch.nn.functional as F

from torch.cuda.amp import autocast

from einops import einsum, rearrange


from xformers.components.attention.feature_maps import (
    SMHyperbolic,
    SMOrf,
    SMReg,
)
from xformers.components.attention.favor import FavorAttention


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        *******************************************************************************************************
        Borrowed from llama original code: https://github.com/meta-llama/llama/blob/main/llama/model.py    ****
        *******************************************************************************************************

        Initialize the RMSNorm normalization layer.
        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.
        Args:
            x (torch.Tensor): The input tensor.
        Returns:
            torch.Tensor: The normalized tensor.
        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.
        Args:
            x (torch.Tensor): The input tensor.
        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.
        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class SwiGLUFeedForward(nn.Module):
    def __init__(self, d_hidden):
        super().__init__()
        d_inter = 4 * d_hidden
        self.w1 = nn.Linear(d_hidden, d_inter, bias=False)
        self.w2 = nn.Linear(d_inter, d_hidden, bias=False)
        self.w3 = nn.Linear(d_hidden, d_inter, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
    

class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.max_seq_len_cached = 0
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        if self.max_seq_len_cached < seq_len:
            self.max_seq_len_cached = seq_len
            t = torch.arange(
                self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
            )

            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1)
            self.register_buffer(
                "cos_cached", emb.cos().to(dtype), persistent=False
            )
            self.register_buffer(
                "sin_cached", emb.sin().to(dtype), persistent=False
            ) # seq_len, dim

    def rotate_half(self, x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
        
    
    def forward(self, x):
        bs, l, nh, dh = x.shape
        self._set_cos_sin_cache(seq_len=l, device=x.device, dtype=x.dtype)
        cos = self.cos_cached[:l].unsqueeze(0).unsqueeze(2)
        sin = self.sin_cached[:l].unsqueeze(0).unsqueeze(2)
        return (x * cos) + (self.rotate_half(x) * sin)


class LowRankHighOrderSelfAttention(nn.Module):
    def __init__(
        self, 
        d_model,
        d_head, 
        n_head,
        dropout=0.,
        use_linear_att=True,
        feature_map='SMReg',
        rotary_emb_list=None,
        ignore_list=None
    ):
        super().__init__()
        self.d_head = d_head
        self.n_head = n_head
        self.rotary_emb_list = [] if rotary_emb_list is None else rotary_emb_list
        self.ignore_list = [] if ignore_list is None else ignore_list
        self.rotary_emb = RotaryEmbedding(d_head, max_position_embeddings=1024)
        self.query_proj = nn.Linear(d_model, d_head * n_head)
        self.key_proj = nn.Linear(d_model, d_head * n_head)
        self.value_proj = nn.Linear(d_model, d_head * n_head)
        self.out_proj = nn.Linear(d_head * n_head, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.feature_map = {
            'SMReg' : SMReg,
            'SMHyperbolic' : SMHyperbolic,
            'SMOrf' : SMOrf,
        }[feature_map](
            dim_features=math.ceil(d_head * (1 + math.log2(d_head))),
            iter_before_redraw=None,
            normalize_inputs=False
        )
        self.use_linear_att = use_linear_att

    def _softmax_kernel(self, x, normalize=True):
        x_ = x.float()
        if normalize:
            x_ = (x_ - x_.min()) / (x_.max() - x_.min())
        return self.feature_map(x_)

    def _standard_attention(self, q, k, v):
        att = einsum(q, k, 'bs l1 nh d, bs l2 nh d -> bs l1 l2 nh')
        att = F.softmax(att / math.sqrt(q.shape[3]), dim=2)
        return einsum(att, v, 'bs l1 l2 nh, bs ... l2 nh d -> bs ... l1 nh d')
        
    def _linear_attention(self, q, k, v):
        q_prime = self._softmax_kernel(q) 
        k_prime = self._softmax_kernel(k)
        
        att_raw = einsum(
            q_prime,
            einsum(
                k_prime, 
                v.float(), 
                'bs l nh d1, bs ... l nh d2 -> bs ... nh d1 d2'
            ),
            'bs l nh d1, bs ... nh d1 d2 -> bs ... l nh d2'
        )
        att_norm = einsum(
            q_prime,
            einsum(
                k_prime, 
                torch.ones_like(v.float()).to(v.device), 
                'bs l nh d1, bs ... l nh d2 -> bs ... nh d1 d2'
            ),
            'bs l nh d1, bs ... nh d1 d2 -> bs ... l nh d2'
        )
        return self.dropout(att_raw / att_norm)
        

    def compute_attention(self, query, key, value, dim, use_rotary_emb=False):
        def pool(x):
            return einsum(x, 'bs ... l nh dh -> bs l nh dh')        

        l = query.shape[dim]
        q = query.transpose(dim, -2)   #(bs ... l d)  
        k = key.transpose(dim, -2)     #(bs ... l d)
        v = value.transpose(dim, -3)   #(bs ... l nh dh)
        orig_dims = v.shape[:-3]

        q = q.unflatten(dim=-1, sizes=(self.n_head, self.d_head))   # (bs ... l nh dh)
        k = k.unflatten(dim=-1, sizes=(self.n_head, self.d_head))   # (bs ... l nh dh)

        q = pool(q)
        k = pool(k)
    
        if use_rotary_emb:
            q = self.rotary_emb(q)
            k = self.rotary_emb(k)

        if self.use_linear_att:
            h = self._linear_attention(q, k, v)
        else:
            h = self._standard_attention(q, k, v)
        return h.transpose(dim, -3)
        


    def forward(self, X, Z=None):
        query = self.query_proj(X)
        key = self.key_proj(X if Z is None else Z)
        value = self.value_proj(X if Z is None else Z).unflatten(dim=-1, sizes=(self.n_head, self.d_head))
        
        for dim in range(1, len(X.shape) - 1):
            if dim in self.ignore_list:
                continue
            value = self.compute_attention(
                query=query, 
                key=key,
                value=value,
                dim=dim,
                use_rotary_emb=dim in self.rotary_emb_list
            )

        return self.out_proj(value.flatten(start_dim=-2))