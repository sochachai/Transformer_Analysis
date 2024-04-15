import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import Positional_Encoding



def attention(query, key, value, mask = None, dropout = None):
    '''
    :param query: vectorized original text
    :param key: key words of text
    :param value: the original value of key, summarization of query
    :param mask: hide words to avoid data leakage
    :param dropout: dropout rate of neural network
    :return:
    '''
    d_k = query.size(-1)
    print(f"query shape is {query.shape}")
    print(f"key transpose shape is {key.transpose(-2, -1).shape}")
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    print('scores.shape is ', scores.shape)
    #print('scores is ', scores)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9) # compare each position with 0

    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn



input = Variable(torch.rand(5,5))
print(f"The input is {input}.")

mask = Variable(torch.zeros(5,5))
#mask = Variable(torch.tensor([[1,2,3,4,5],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]))
print(f"The mask is {mask}")

print(f"The masked input is {input.masked_fill(mask == 0, -1e9)}")

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
print(f"pe_result shape is {pe_result.shape}")

query = key = value = pe_result
attn, p_attn = attention(query, key, value)
print("attn:", attn)
print(f"attn shape is {attn.shape}")
print("p_attn:", p_attn)

mask = Variable(torch.zeros(2, 4, 4))
print(f"mask is {mask}")
attn, p_attn = attention(query, key, value, mask = mask)
print("attn:", attn)
print("p_attn:", p_attn)


