import Attention
import copy
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import Positional_Encoding
import Multihead_Attention

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        '''
        :param d_model: embedding dimension
        :param d_ff: transitional dimension
        :param dropout: default dropout set to 0.1
        '''
        super(PositionwiseFeedForward, self).__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w2(self.dropout(F.relu(self.w1(x))))

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
ff = PositionwiseFeedForward(d_model, d_ff, dropout)
ff_result = ff(x)
print(ff_result)
print(ff_result.shape)
'''