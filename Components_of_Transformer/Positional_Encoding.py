import torch
import torch.nn as nn
import math
from torch.autograd import Variable

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        '''
        :param d_model: dimension of embeddings
         :param vocab: size of vocabulary
        '''
        # Inherit the initialization of nn.Module
        super(Embeddings, self).__init__()
        # Define a word embedding object
        self.lut = nn.Embedding(vocab, d_model)
        # Get the value of d_model
        self.d_model = d_model

    def forward(self, x):
        '''
        :param x: Tensorization of original text
        '''
        return self.lut(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len = 5000):
        '''
        :param d_model: dimension of the encoding
        :param dropout: dropout rate from 0 to 1
        :param max_len: the maximum length of a sentence
        '''
        # Inherit the initialization of nn.Module
        super(PositionalEncoding, self).__init__()

        # Objectify dropout
        self.dropout = nn.Dropout(p=dropout)

        # Inherit a positional encoder matrix, max_len * d_model
        pe = torch.zeros(max_len, d_model)

        # Inherit an absolute position matrix, max_len * 1
        position = torch.arange(0, max_len).unsqueeze(1)

        # Define the conversion matrix, initialization with gap = 2
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0)/d_model))

        # Copy the absolute position matrix to the positional encoder matrix
        # by sine and cosine wave according to the parity of column indices
        pe[:, 0::2] = torch.sin(position * div_term) # even indiced columns are imputed by sine
        pe[:, 1::2] = torch.cos(position * div_term) # odd indiced columns are imputed by cosine

        # Extend pe to 3-dimensional tensor
        pe = pe.unsqueeze(0)

        # Register pe to a buffer, the buffer is not a parameter of the class
        # the buffer will not be updated along with the model update
        # but it can be loaded along with the model
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        :param x: Tensor of text
        :return: x + the positional encoding
        '''
        # Shrink the size of pe to save storage
        # by converting the second dimension, i.e. the dimension of max_len
        # to the size of the sentence len of x, i.e. the second dimension of x
        x = x + Variable(self.pe[:,:x.size(1)], requires_grad = False) # False: pe will not be updated
        return self.dropout(x)


d_model = 512
vocab = 1000
dropout = 0.1
max_len = 60

original_text = Variable(torch.LongTensor([[132,8,521,308],[491,398,999,223]]))
emb = Embeddings(d_model, vocab)
embr = emb(original_text)

x= embr
pe = PositionalEncoding(d_model, dropout, max_len)
pe_result = pe(x)
print(pe_result)
print(pe_result.shape)



