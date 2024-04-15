import Attention
import copy
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import Positional_Encoding
import Multihead_Attention
import PositionwiseFeedForward

class LayerNorm(nn.Module):
    def __init__(self, features, eps = 1e-6):
        '''
        :param features: embedding dimension
        :param eps: avoid zero denominator
        '''
        super(LayerNorm, self).__init__()
        # nn.Parameter formats the variables to intrinsic parameters
        # they will be updated along with the model in contrast with buffer
        self.a2 = nn.Parameter(torch.ones(features))
        self.b2 = nn.Parameter(torch.zeros(features))
        # initialization of eps
        self.eps = eps

    def forward(self, x):
        '''
        :param x: the output of previous layer
        :return: numerical standardized x
        '''
        mean = x.mean(-1, keepdim = True)
        std = x.std(-1, keepdim = True)
        '''
        print(f"x_shape is {x.shape}")
        print(f"mean_shape is {mean.shape}")
        print(f"std_shape is {std.shape}")
        print(f"a2_shape is {self.a2.shape}")
        print(f"a2(x-mean)_shape is {(self.a2*(x-mean)).shape}")
        print(f"(std+eps)_shape is {(std+self.eps).shape}")
        print(f"(self.a2 * (x - mean) / (std + self.eps))_shape is\
                {(self.a2 * (x - mean) / (std + self.eps)).shape}")
        print(self.a2 * (x - mean) / (std + self.eps))
        '''
        return self.a2 * (x - mean) / (std + self.eps) + self.b2

'''
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



head = 8
embedding_dim = 512
dropout = 0.2
query = key = value = pe_result
#print(pe_result)
mask = Variable(torch.zeros(2, 4, 4))
mha = Multihead_Attention.MultiHeadedAttention(head, embedding_dim, dropout)
mha_result = mha(query, key, value, mask)
print(mha_result)
print(mha_result.shape)

d_model = 512
d_ff = 64
dropout = 0.2
x = mha_result
ff = PositionwiseFeedForward.PositionwiseFeedForward(d_model, d_ff, dropout)
ff_result = ff(x)

features = d_model = 512
eps = 1e-6
x = ff_result
print(x)

layer_norm = LayerNorm(features, eps)
layer_norm_result = layer_norm(x)
print(layer_norm_result)
'''