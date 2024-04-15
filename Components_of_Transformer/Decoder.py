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
import DecoderLayer

class Decoder(nn.Module):
    def __init__(self, layer, N):
        '''
        :param layer: decoder layer
        :param N: the number of decoder layers
        '''
        super(Decoder, self).__init__()
        self.layers = Multihead_Attention.clones(layer, N)
        self.norm = LayerNorm.LayerNorm(layer.size)

    def forward(self, x, memory, source_mask, target_mask):
        for layer in self.layers:
            x = layer(x, memory, source_mask, target_mask)
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

# exemplified parameters
size = 512
head = 8
d_model = 512
d_ff = 64
dropout = 0.2
c = copy.deepcopy
attn = Multihead_Attention.MultiHeadedAttention(head, d_model)
ff = PositionwiseFeedForward.PositionwiseFeedForward(d_model, d_ff, dropout)
layer = DecoderLayer.DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout)
N = 8

# input parameters
x = pe_result
memory = Encoder.en_result # result from Encoder.py
# In practice source_mask and target_mask are not the same
mask = Variable(torch.zeros(2, 4, 4))
source_mask = target_mask = mask

# application
de = Decoder(layer, N)
de_result = de(x, memory, source_mask, target_mask)
print(de_result)
print(de_result.shape)
