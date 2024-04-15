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

class Encoder(nn.Module):
    def __init__(self, layer, N):
        '''
        :param layer: one encoder layer
        :param N: the number of encoder layers
        '''
        super(Encoder, self).__init__()
        self.layers = Multihead_Attention.clones(layer, N)
        self.norm = LayerNorm.LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


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
head = 8
d_model = 512
d_ff = 64
x = pe_result
c = copy.deepcopy
attn = Multihead_Attention.MultiHeadedAttention(head, d_model)
ff = PositionwiseFeedForward.PositionwiseFeedForward(d_model, d_ff, dropout)
dropout = 0.2
layer = EncoderLayer.EncoderLayer(size, c(attn), c(ff), dropout)
N = 8
mask = Variable(torch.zeros(2, 4, 4))

# application
en = Encoder(layer, N)
en_result = en(x, mask)
print(en_result)
print(en_result.shape)
