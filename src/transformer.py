import torch
from torch import nn

from .modules import RMSNorm, LowRankHighOrderSelfAttention, SwiGLUFeedForward

from torch import nn

from .modules import *


class HighOrderTransformerBlock(nn.Module):
    def __init__(
        self, 
        d_hidden, 
        d_head, 
        n_head, 
        dropout=0., 
        use_linear_att=True,
        feature_map='SMReg',
        rotary_emb_list=None,
        ignore_list=None
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.norm1 = RMSNorm(d_hidden)
        self.norm2 = RMSNorm(d_hidden)
        self.norm3 = RMSNorm(d_hidden)
        self.normz = RMSNorm(d_hidden)
        self.attention = LowRankHighOrderSelfAttention(
            d_hidden, 
            d_head, 
            n_head, 
            dropout, 
            use_linear_att, 
            feature_map, 
            rotary_emb_list, 
            ignore_list
        )
        self.cross_attention = LowRankHighOrderSelfAttention(
            d_hidden, 
            d_head, 
            n_head, 
            dropout, 
            use_linear_att, 
            feature_map, 
            rotary_emb_list, 
            ignore_list
        )
        self.feedforward = SwiGLUFeedForward(d_hidden)

    def forward(self, X, Z=None):
        h = X + self.attention(self.norm1(X))
        if Z is not None:
            h = h + self.cross_attention(self.norm3(h), self.normz(Z))
        return self.dropout(h + self.feedforward(self.norm2(h)))
    

class HighOrderTransformer(nn.Module):
    def __init__(
            self, 
            d_hidden,
            n_blocks = 2, 
            d_head = 16, 
            n_head = 4, 
            dropout=0., 
            use_linear_att=True,
            feature_map='SMReg',
            rotary_emb_list=None,
            ignore_list=None
        ):
        super().__init__()
        blocks = []
        for _ in range(n_blocks):
            blocks += [
                HighOrderTransformerBlock(
                    d_hidden,
                    d_head, 
                    n_head, 
                    dropout, 
                    use_linear_att, 
                    feature_map, 
                    rotary_emb_list, 
                    ignore_list
                )
            ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, X, Z=None):
        hiddens = []
        h = X
        for i, block in enumerate(self.blocks):
            h = block(h, None if Z is None else Z[i])
            hiddens += [h]
        return h, torch.stack(hiddens)
