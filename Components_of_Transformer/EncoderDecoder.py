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
import Decoder
import Generator

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, source_embed, target_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = source_embed
        self.tgt_embed = target_embed
        self.generator = generator

    def forward(self, source, target, source_mask, target_mask):
        # encoded source used as memory in decode function
        return self.decode(self.encode(source, source_mask), source_mask,
                           target, target_mask)

    def encode(self, source, source_mask):
        return self.encoder(self.src_embed(source), source_mask)

    def decode(self, memory, source_mask, target, target_mask):
        # embedded target as x in the decoder function
        return self.decoder(self.tgt_embed(target), memory, source_mask, target_mask)

# exemplified parameters
vocab_size = 1000
d_model= 512
encoder = Encoder.en
decoder = Decoder.de
source_embed = nn.Embedding(vocab_size, d_model)
target_embed = nn.Embedding(vocab_size, d_model)
generator = Generator.gen

# input parameters (assuming source and target are the same)
source = target = Variable(torch.LongTensor([[100, 2, 421, 500], [491, 998, 1, 221]]))
source_mask = target_mask = Variable(torch.zeros(2, 4, 4))

# application
ed = EncoderDecoder(encoder, decoder, source_embed, target_embed, generator)
ed_result = ed(source, target, source_mask, target_mask)
print(ed_result)
print(ed_result.shape)