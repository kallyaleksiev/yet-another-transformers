import math

import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_

import comp_utils
from components import MHAttention_ListImpl, LayerNorm, PWFeedForward, PositionalEncoding


class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, d_ff=2048, dropout=0.1, activation="ReLU"):
        super(Transformer, self).__init__()

        self.d_model = d_model
        self.embed_matrix = nn.Parameter(
            xavier_uniform_(torch.zeros(vocab_size, d_model)))
        self.pos_encoding = PositionalEncoding(d_model=d_model)

        self.enc_dropout = nn.Dropout(p=dropout)
        encoder_layer = EncoderLayer(
            d_model=d_model, d_ff=d_ff, dropout=dropout, activation=activation)
        self.encoder_stack = comp_utils.clone(encoder_layer, times=6)

        self.dec_dropout = nn.Dropout(p=dropout)
        decoder_layer = DecoderLayer(
            d_model=d_model, d_ff=d_ff, dropout=dropout, activation=activation)
        self.decoder_stack = comp_utils.clone(decoder_layer, times=6)

        self.linear = nn.Linear(d_model, d_model)

    def forward(self, primary, context, attention_mask, padding_mask):
        z = self._embedding(context)
        z = self.enc_dropout(z + self.pos_encoding(z))
        z = self._encode(z, attention_mask, padding_mask)

        x = self._embedding(primary)
        x = self.dec_dropout(x + self.pos_encoding(x))
        x = self._decode(x, z, attention_mask, padding_mask)

        x = self.linear(x)

        return x

    def _encode(self, x, attention_mask, padding_mask):
        for enc_layer in self.encoder_stack:
            x = enc_layer(x, attention_mask, padding_mask)

        return x

    def _decode(self, x, enc_output, attention_mask, padding_mask):
        for dec_layer in self.decoder_stack:
            x = dec_layer(x, enc_output, attention_mask, padding_mask)

        return x

    def _embedding(self, tokens_batch):
        return self.embed_matrix[tokens_batch] / math.sqrt(self.d_model)

    def _unembedding(self, x):
        return torch.matmul(x, self.embed_matrix.T)


class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, d_ff=2048, dropout=0.1, activation="ReLU"):
        super(EncoderLayer, self).__init__()

        assert d_model % 8 == 0, "Dimension of the model must be divisible by 8 (the default number of heads)"
        d_mid = d_model // 8

        self.mha = MHAttention_ListImpl(h=8, d_mid=d_mid, d_attn=d_mid,
                                        d_out=d_model, d_x=d_model, d_z=d_model,
                                        q_bias=False, k_bias=False, v_bias=False,
                                        o_bias=True)
        self.dropout1 = nn.Dropout(p=dropout)
        self.layernorm1 = LayerNorm(d_input=d_model)

        self.pwff = PWFeedForward(
            d_e=d_model, d_mlp=d_ff, activation=activation)
        self.dropout2 = nn.Dropout(p=dropout)
        self.layernorm2 = LayerNorm(d_input=d_model)

    def forward(self, x, attention_mask, padding_mask):
        x = self.layernorm1(x + self._attention_block(x,
                            attention_mask, padding_mask))
        x = self.layernorm2(x + self._pwff_block(x))
        return x

    def _attention_block(self, x, attention_mask, padding_mask):
        x = self.mha(x, x, padding_mask=padding_mask,
                     attention_mask=attention_mask)
        return self.dropout1(x)

    def _pwff_block(self, x):
        x = self.pwff(x)
        return self.dropout2(x)


class DecoderLayer(nn.Module):
    def __init__(self, d_model=512, d_ff=512, dropout=0.1, activation="ReLU"):
        super(DecoderLayer, self).__init__()

        assert d_model % 8 == 0, "Dimension of the model must be divisible by 8 (the default number of heads)"
        d_mid = d_model // 8

        self.mha1 = MHAttention_ListImpl(h=8, d_mid=d_mid, d_attn=d_mid,
                                         d_out=d_model, d_x=d_model, d_z=d_model,
                                         q_bias=False, k_bias=False, v_bias=False,
                                         o_bias=True)
        self.dropout1 = nn.Dropout(p=dropout)
        self.layernorm1 = LayerNorm(d_input=d_model)

        self.mha2 = MHAttention_ListImpl(h=8, d_mid=d_mid, d_attn=d_mid,
                                         d_out=d_model, d_x=d_model, d_z=d_model,
                                         q_bias=False, k_bias=False, v_bias=False,
                                         o_bias=True)
        self.dropout2 = nn.Dropout(p=dropout)
        self.layernorm2 = LayerNorm(d_input=d_model)

        self.pwff = PWFeedForward(
            d_e=d_model, d_mlp=d_ff, activation=activation)
        self.dropout3 = nn.Dropout(p=dropout)
        self.layernorm3 = LayerNorm(d_input=d_model)

    def forward(self, x, encoder_output, attention_mask, padding_mask):
        x = self.layernorm1(x + self._self_attention_block(x, padding_mask))
        x = self.layernorm2(x + self._enc_dec_attention_block(x,
                            encoder_output, attention_mask, padding_mask))
        x = x + self._pwff_block(x)
        return x

    def _self_attention_block(self, x, padding_mask):
        x = self.mha1(x, x, padding_mask=padding_mask)
        return self.dropout1(x)

    def _enc_dec_attention_block(self, x, encoder_output, attention_mask, padding_mask):
        x = self.mha2(primary=x, context=encoder_output,
                      padding_mask=padding_mask, attention_mask=attention_mask)
        return self.dropout2(x)

    def _pwff_block(self, x):
        x = self.pwff(x)
        return self.dropout3(x)


""" Testing models which we use to run experiments
"""


class Test_RottenTomatoes_Classifier(nn.Module):
    def __init__(self, vocab_size):
        super(Test_RottenTomatoes_Classifier, self).__init__()

        self.small_transformer = Transformer(vocab_size=vocab_size,
                                             d_model=16,
                                             d_ff=64,
                                             dropout=0.1,)
        self.classifier = nn.Linear(16, 2)

    def forward(self, input_ids, padding_mask):
        trans_out = self.small_transformer(
            primary=input_ids, context=input_ids, attention_mask=None, padding_mask=padding_mask)

        cls_out = trans_out[:, 0, :].squeeze(dim=1)
        return self.classifier(cls_out)
