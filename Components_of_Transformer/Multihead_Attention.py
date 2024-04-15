import Attention
import copy
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import Positional_Encoding

def clones(module, N):
    '''
    :param module: one attention layer
    :param N: the number of module
    '''
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)]) # deepcopy uses a different memory

class MultiHeadedAttention(nn.Module):
    def __init__(self, head, embedding_dim, dropout = 0.1):
        '''
        :param head: the number of heads
        :param embedding_dim: the embedding dimension
        :param dropout: default dropout rate set to 0.1
        '''
        # Inherit the initialization
        super(MultiHeadedAttention, self).__init__()
        # head must be an integral factor of embedding_dim
        assert embedding_dim % head == 0
        # each head is assigned with the following dimension
        self.d_k = embedding_dim // head # division in the integral domain Z
        # substantiate head
        self.head = head
        # create linear layers, we need 4 of them for Q, K, V and the final connection
        self.linears = clones(nn.Linear(embedding_dim, embedding_dim), 4)
        # create the attention and dropout rate
        self.attn = None
        self.dropout = nn.Dropout(p = dropout)

    def forward(self, query, key, value, mask = None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        batch_size = query.size(0)

        # We have 4 linears and only zipping the first 3 with Q, K, V, the last linear is not used.
        # view(batch_size, -1, self.head, self.d_k).transpose(1,2)
        # is not the same with view(batch_size, self.head, -1, self.d_k)
        # self.head and self.d_k should be neighboring to get embedding_dim in order for the tensor
        # to interpret the relationship between the meaning of words of their positions in a sentence

        query, key, value = \
            [model(x).view(batch_size, -1, self.head, self.d_k).transpose(1,2)
             for model, x in zip(self.linears, (query, key, value))]

        x, self.attn = Attention.attention(query, key, value, mask = mask, dropout = self.dropout)

        # Reshape x
        # must use contiguous method after the transpose before the view method
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.head * self.d_k)

        # Pass x to the 4th linear layer
        return self.linears[-1](x)


# Example
input = Variable(torch.rand(5,5))
print(input)

mask = Variable(torch.zeros(5,5))
#mask = Variable(torch.tensor([[1,2,3,4,5],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]))
print(mask)

print(input.masked_fill(mask == 0, -1e9))

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
mha = MultiHeadedAttention(head, embedding_dim, dropout)
mha_result = mha(query, key, value, mask)
print(mha_result)
print(mha_result.shape)