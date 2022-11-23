import math

import torch
import torch.nn as nn

from torch.nn.init import xavier_uniform_

import comp_utils


# this constant is useful because it has the property that it is a float x,
# s.t. torch.exp(torch.tensor([x])) = 0.0, so it can be used in softmax
MINUS_INFINITY = -1e6


class Attention(nn.Module):
    r"""The original attention layer, which computes a single 
    (possibly masked) self- or cross-attention head

    I explicitly assume 3D batch-first inputs, i.e.
    X \in \mathbb{R}^{b \times l_x \times d_x}
    Z \in \mathbb{R}^{b \times l_z \times d_z}

    Moreover, none of d_attn, d_out, d_x, d_z are assumed to be 
    equal by design. So basically this cimputes the attention
    from here https://arxiv.org/pdf/2207.09238.pdf rather than 
    the OG https://arxiv.org/pdf/1706.03762.pdf 
    """

    def __init__(self, d_attn, d_out, d_x, d_z=None,
                 q_bias=True, k_bias=True, v_bias=True):
        super(Attention, self).__init__()
        d_z = d_x if d_z is None else d_z

        self.d_attn = d_attn

        self.W_q = nn.Linear(d_x, d_attn, bias=q_bias)
        self.W_k = nn.Linear(d_z, d_attn, bias=k_bias)
        self.W_v = nn.Linear(d_z, d_out, bias=v_bias)

    def forward(self, primary, context, padding_mask=None, attention_mask=None):
        queries = self.W_q(primary)  # b x l_x x d_attn
        keys = self.W_k(context)  #  b x l_z x d_attn
        values = self.W_v(context)  # b x l_z x d_out

        attention = torch.bmm(queries, keys.transpose(2, 1))  # b x l_x x l_z

        if padding_mask is not None:
            mask = padding_mask.unsqueeze(dim=1)  # b x 1 x l_z
            if attention_mask is not None:
                mask = attention_mask.logical_and(mask)  # b x l_x x l_z
        elif attention_mask is not None:
            mask = attention_mask
        else:
            mask = torch.ones_like(attention, dtype=int)

        attention = attention / math.sqrt(self.d_attn)
        attention = attention.masked_fill(mask == 0, MINUS_INFINITY)
        attention = attention.softmax(dim=-1)  # b x l_x x l_z

        return torch.bmm(attention, values)  # b x l_x x d_out


class MHAttention_SimulMulImpl(nn.Module):
    r"""Multihead Attention Layer that is implemented with
    matmul operations, so essentially computes all the attention 
    heads at the same time and then takes care to reshape the
    output
    """

    def __init__(self, h, d_mid, d_attn, d_out, d_x, d_z=None, o_bias=True):
        super(MHAttention_SimulMulImpl, self).__init__()

        self.num_heads = h
        self.d_attn = d_attn
        self.d_mid = d_mid
        self.d_out = d_out
        d_z = d_x if d_z is None else d_z

        self.W_qs = nn.Parameter(xavier_uniform_(torch.zeros(d_x, h*d_attn)))
        self.W_ks = nn.Parameter(xavier_uniform_(torch.zeros(d_z, h*d_attn)))
        self.W_vs = nn.Parameter(xavier_uniform_(torch.zeros(d_z, h*d_mid)))
        self.W_o = nn.Linear(h*d_mid, d_out, bias=o_bias)

    def forward(self, primary, context, padding_mask=None, attention_mask=None):
        b, l_x, l_z = primary.shape[0], primary.shape[1], context.shape[1]
        h, d_attn, d_mid = self.num_heads, self.d_attn, self.d_mid

        queries = torch.matmul(primary, self.W_qs)  # b x l_x x h*d_attn
        keys = torch.matmul(context, self.W_ks)     # b x l_z x h*d_attn
        values = torch.matmul(context, self.W_vs)   # b x l_z x h*d_mid

        queries = queries.transpose(1, 2).reshape(
            b, h, d_attn, l_x).transpose(2, 3)
        keys = keys.transpose(1, 2).reshape(b, h, d_attn, l_z).transpose(2, 3)
        values = values.transpose(1, 2).reshape(
            b, h, d_mid, l_z).transpose(2, 3)

        attention = torch.matmul(
            queries, keys.transpose(2, 3))  # b x h x l_x x l_z

        if padding_mask is not None:
            mask = padding_mask.unsqueeze(dim=1)  # b x 1 x l_z
            if attention_mask is not None:
                mask = attention_mask.logical_and(
                    mask).unsqueeze(dim=1)  # b x 1 x l_x x l_z
            else:
                mask = mask.unsqueeze(dim=1)  # b x 1 x 1 x l_z
        elif attention_mask is not None:
            mask = attention_mask.unsqueze(dim=1)  # b x 1 x l_x x l_z
        else:
            mask = torch.ones_like(attention, dtype=int)

        attention = attention / math.sqrt(d_attn)
        attention = attention.masked_fill(mask == 0, MINUS_INFINITY)
        attention = attention.softmax(dim=-1)  # b x h x l_x x l_z

        preproj = torch.matmul(attention, values)  # b x h x l_x x d_out
        preproj = preproj.transpose(2, 3).reshape(
            b, h*d_mid, l_x).transpose(1, 2)  # b x l_x x h*d_mid
        return self.W_o(preproj)


class MHAttention_ListImpl(nn.Module):
    r"""Multihead Attention Layer that is implemented by
    concatenating the outputs of several attention heads
    """

    def __init__(self, h, d_mid, d_attn, d_out, d_x, d_z=None,
                 q_bias=True, k_bias=True, v_bias=True, o_bias=True):
        super(MHAttention_ListImpl, self).__init__()

        attention_layer = Attention(d_attn=d_attn, d_out=d_mid,
                                    d_x=d_x, d_z=d_z,
                                    q_bias=q_bias, k_bias=k_bias, v_bias=v_bias)
        self.heads = comp_utils.clone(attention_layer, times=h)
        self.W_o = nn.Linear(h*d_mid, d_out, bias=o_bias)

    def forward(self, primary, context, padding_mask=None, attention_mask=None):
        attn_concat = torch.cat([
            head(primary, context, padding_mask, attention_mask)
            for head in self.heads], dim=-1)  # b x l_x x h*d_mid
        return self.W_o(attn_concat)  #  b x l_x x d_out


class LayerNorm(nn.Module):
    r"""Layer normalisation
    """

    def __init__(self, d_input, epsilon=1e-5):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_input))
        self.beta = nn.Parameter(torch.zeros(d_input))
        self.eps = epsilon

    def forward(self, x):
        m = x.mean(dim=-1, keepdim=True)
        v = x.std(dim=-1, unbiased=False, keepdim=True)
        return ((x-m)/(v+self.eps))*self.gamma + self.beta


class PWFeedForward(nn.Module):
    r"""Position-wise feed forward nn
    """

    def __init__(self, d_e, d_mlp, activation):
        super(PWFeedForward, self).__init__()

        if activation.lower() == "relu":
            self.activation = nn.ReLU()
        else:
            self.activation = nn.GELU()

        self.W_1 = nn.Linear(d_e, d_mlp)
        self.W_2 = nn.Linear(d_mlp, d_e)

    def forward(self, x):
        return self.W_2(self.activation(self.W_1(x)))


class PositionalEncoding(nn.Module):
    r"""The sinusoidal positional encodings
    """

    def __init__(self, d_model, max_len=10000):
        super(PositionalEncoding, self).__init__()

        TEN_THOUSAND = 100000

        positionals = torch.arange(start=0, end=max_len).unsqueeze(dim=1)

        dimensionals = torch.arange(start=0, end=d_model)
        dimensionals = 2 * torch.div(dimensionals,
                                     2, rounding_mode="trunc").unsqueeze(dim=0)
        dimensionals = math.log(TEN_THOUSAND) * \
            (torch.div(dimensionals, d_model))
        dimensionals = torch.exp(dimensionals)

        pe = positionals / dimensionals
        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])
        pe = pe.unsqueeze(dim=0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.shape[1], :].requires_grad_(False)

        return x
