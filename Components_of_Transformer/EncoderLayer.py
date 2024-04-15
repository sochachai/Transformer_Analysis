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

class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        '''
        :param size: embedding dimension
        :param self_attn: attention
        :param feed_forward: positionwise feed forward
        :param dropout: avoid overfitting
        '''
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = Multihead_Attention.clones(SublayerConnection.SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        '''
        :param x: output of previous layer
        :param mask: tensor mask to prevent data leakage
        '''
        # input matrix followed by operating function, returns an output matrix
        # sublayer(x, function) will return function(x)
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask)) # the forward function of multihead attention
        return self.sublayer[1](x, self.feed_forward) # the forward function of pointwise feedforward
'''
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

# get parameters
size = 512
head = 2
d_model = 512
d_ff = 64
x = pe_result
dropout = 0.2
self_attn = Multihead_Attention.MultiHeadedAttention(head, d_model)
ff = PositionwiseFeedForward.PositionwiseFeedForward(d_model, d_ff, dropout)
mask = Variable(torch.zeros(2, 4, 4))

# application
el = EncoderLayer(size, self_attn, ff, dropout)
el_result = el(x, mask)
print(el_result)
print(el_result.shape)
'''