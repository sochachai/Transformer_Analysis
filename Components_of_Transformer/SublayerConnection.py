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

class SublayerConnection(nn.Module):
    def __init__(self, size, dropout = 0.1):
        '''
        :param size: embedding size
        :param dropout: deactivation of neurons to avoid overfitting
        '''
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm.LayerNorm(size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, sublayer):
        '''
        :param x: the output of previous layer
        :param sublayer: a function, e.g. Multihead_Attention, PositionwiseFeedForward etc.
        :return: x plus sublayer functioning on normed x with dropout
        '''
        return x + self.dropout(sublayer(self.norm(x)))

'''
# get pe_result
d_model = 512
vocab = 1000
dropout = 0.1
max_len = 60

original_text = Variable(torch.LongTensor([[132,8,521,308],[491,398,999,223]]))
emb = Embedding_Encoder.Embeddings(d_model, vocab)
embr = emb(original_text)

x= embr
pe = Positional_Encoding.PositionalEncoding(d_model, dropout, max_len)
pe_result = pe(x)

# Substantiate variables
size = 512
dropout = 0.2
head = 2
d_model = 512

# Input
x = pe_result
mask = Variable(torch.zeros(2,4,4))
self_attn = Multihead_Attention.MultiHeadedAttention(head, d_model)
sublayer = lambda x: self_attn(x, x, x, mask)

# Application
sc = SublayerConnection(size, dropout)
sc_result = sc(x, sublayer)
print(sc_result)
'''