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

'''
# Demonstrative Example of Embeddings
embedding = nn.Embedding(9, 3) # 3 is the d_model
input = torch.LongTensor([[3,2,1,5],[6,3,8,5]])
print(embedding(input))

embedding = nn.Embedding(9, 3, padding_idx=0)
input = torch.LongTensor([[0,6,0,3]])
print(embedding(input))

# Practical Example of Embeddings
d_model = 512
vocab = 1000
x = Variable(torch.LongTensor([[132,8,521,308],[491,398,999,223]]))
emb = Embeddings(d_model, vocab)

#The positions of d_model and vocab are switched in comparison with nn.Embedding
#Each number is mapped to a vector of dimension d_model
#The upper bound of the numbers cannot exceed (vocab - 1)

embr = emb(x)
print("embr:", embr)
print(embr.shape)
'''