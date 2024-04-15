import Attention
import copy
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import Embedding_Encoder
import Positional_Encoding
import Multihead_Attention
import PositionwiseFeedForward
import LayerNorm
import SublayerConnection
import EncoderLayer
import Encoder

class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        '''
        :param size: embedding dimension
        :param self_attn: masked multihead attention
        :param src_attn: multihead attention
        :param feed_forward: pointwise feed forward
        :param dropout: avoid overfitting
        '''
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        # 3 sublayer for a decoder layer
        self.sublayer = Multihead_Attention.clones(SublayerConnection.SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, source_mask, target_mask):
        '''
        :param x: output of previous layer
        :param memory: result of encoder
        :param source_mask: delete unnecessary info to improve model performance
        :param target_mask: hide info to prevent data leakage
        :return: output tensor
        '''
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, target_mask))
        # for second layer, Q = x, K = V = m
        x = self.sublayer[1](x, lambda x: self.self_attn(x, m, m, source_mask))
        return self.sublayer[2](x, self.feed_forward)

# get pe_result
d_model = 512
vocab = 1000
dropout = 0.1
max_len = 60
original_text = Variable(torch.LongTensor([[132, 8, 521, 308], [491, 398, 999, 223]]))
emb = Positional_Encoding.Embeddings(d_model, vocab)
embr = emb(original_text)
x = embr
pe = Positional_Encoding.PositionalEncoding(d_model, dropout, max_len)
pe_result = pe(x)

# exemplified parameters
size = 512
head = 8
d_model = 512
d_ff = 64
dropout = 0.2
self_attn = src_attn = Multihead_Attention.MultiHeadedAttention(head, d_model, dropout)
ff = PositionwiseFeedForward.PositionwiseFeedForward(d_model, d_ff, dropout)

# input parameters
x = pe_result
memory = Encoder.en_result
# In practice source_mask and target_mask are not the same
mask = Variable(torch.zeros(2, 4, 4))
source_mask = target_mask = mask

# application
dl = DecoderLayer(size, self_attn, src_attn, ff, dropout)
dl_result = dl(x, memory, source_mask, target_mask)
print(dl_result)
print(dl_result.shape)