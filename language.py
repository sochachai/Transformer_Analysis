import copy
import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# import pickle # we read the torch model using the load_state_dict method. pickle is not required.
# Issue caused by torch version described below:
# the original package should be pyitcast.transformer_utils
# the problem is that pyitcast.transformer_utils relies on an old version of torch (1.3.1 should work)
# and causes the error "IndexError: invalid index of a 0-dim tensor. Use `tensor.item()` in Python..."
# but the installation of an old version of torch that match pyitcast.transformer is not trivial
# versions of 1.11.0 or higher is not compatible with pyitcast.transformer and installation of them will not
# solve the issue
# To solve this issue:
# download the pyitcast.transformer_utils (open by clicking the error message) as a py file
# modify the last line of the SimpleLossCompute class from "return loss.data[0] * norm"
# to "return loss.data * norm"

from my_transformer_utils import Batch
from my_transformer_utils import run_epoch
from my_transformer_utils import greedy_decode
#from my_transformer_utils import get_std_opt # get_std_opt is based on Adam optimizer
#from my_transformer_utils import LabelSmoothing # offset human label errors to prevent overfitting
#from my_transformer_utils import SimpleLossCompute # calculate loss after smoothing, use cross_entropy_loss

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        '''
        :param d_model: embedding dimension
        :param vocab: size of vocabulary
        '''
        # Initialization
        super(Embeddings, self).__init__()
        # Defrine a word embedding object
        self.lut = nn.Embedding(vocab, d_model)
        # Instantiate d_model
        self.d_model = d_model

    def forward(self, x):
        '''
        :param x: tensor representing the original text
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
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9) # compare each position with 0

    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn

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

        x, self.attn = attention(query, key, value, mask = mask, dropout = self.dropout)

        # Reshape x
        # must use contiguous method after the transpose before the view method
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.head * self.d_k)

        # Pass x to the 4th linear layer
        return self.linears[-1](x)

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
        return self.a2 * (x - mean) / (std + self.eps) + self.b2

class SublayerConnection(nn.Module):
    def __init__(self, size, dropout = 0.1):
        '''
        :param size: embedding size
        :param dropout: deactivation of neurons to avoid overfitting
        '''
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, sublayer):
        '''
        :param x: the output of previous layer
        :param sublayer: a function, e.g. Multihead_Attention, PositionwiseFeedForward etc.
        :return: x plus sublayer functioning on normed x with dropout
        '''
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        '''
        :param size: embedding dimension
        :param self_attn: attention
        :param feed_forward: positionwise feed forward
        :param dropout: avoid overfitting
        '''
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        '''
        :param x: output of previous layer
        :param mask: tensor mask to prevent data leakage
        '''
        # input matrix followed by operating function, returns an output matrix
        # sublayer(x, function) will return function(x)
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask)) # the forward function of multihead attention
        return self.sublayer[1](x, self.feed_forward) # the forward function of pointwise feedforward

class Encoder(nn.Module):
    # Encoder is a collection of Encoder Layers
    def __init__(self, layer, N):
        '''
        :param layer: one encoder layer
        :param N: the number of encoder layers
        '''
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers: x = layer(x, mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        '''
        :param size: embedding dimension
        :param self_attn: masked multihead attention
        :param src_attn: multihead attention
        :param feed_forward: pointwise feed forward
        :param dropout: avoid overfitting
        '''
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        # 3 sublayer for a decoder layer
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, source_mask, target_mask):
        '''
        :param x: output of previous layer
        :param memory: result of encoder
        :param source_mask: delete unnecessary info to improve model performance
        :param target_mask: hide info to prevent data leakage
        :return: output tensor
        '''
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, target_mask))
        # for second layer, Q = x, K = V = m
        x = self.sublayer[1](x, lambda x: self.self_attn(x, m, m, source_mask))
        return self.sublayer[2](x, self.feed_forward)

class Decoder(nn.Module):
    # Decoder is a collection of Decoder Layers
    def __init__(self, layer, N):
        '''
        :param layer: decoder layer
        :param N: the number of decoder layers
        '''
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, source_mask, target_mask):
        for layer in self.layers:
            x = layer(x, memory, source_mask, target_mask)
        return self.norm(x)

class Generator(nn.Module):
    def __init__(self, d_model, vocab_size):
        '''
        :param d_model: embedding dimension
        :param vocab_size: the size of the vocabulary
        '''
        super(Generator, self).__init__()
        self.project = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return F.log_softmax(self.project(x), dim = -1)

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

def make_model(source_vocab, target_vocab, N=6,\
               d_model=512, d_ff=2048, head=8, dropout=0.1):
    c = copy.deepcopy
    attn = MultiHeadedAttention(head, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn),c(ff), dropout), N),
        # note the order of vocab_size and d_model;
        # nn.Embedding is not the same with Embeddings;
        # check Embedding_Encoder.py for more;
        nn.Sequential(Embeddings(d_model, source_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, target_vocab), c(position)),
        Generator(d_model, target_vocab)
    )
    for p in model.parameters():
        if p.dim() > 1: nn.init.xavier_uniform_(p) # initialization: make p uniformly sampled; check xavier_uniform for details
    return model

# use a function to generate data
def data_generator(V, batch_size, num_batch):
    '''
    :param V: the maximal data value + 1
    :param batch_size: sample data size of one round of training after which model parameters are updated
    :param num_batch: number of rounds of training
    '''
    for i in range(num_batch):
        # data value from 1 to V-1, with data matrix shape = batch_size times 10
        data = torch.from_numpy(np.random.randint(1, V, size = (batch_size, 40)))

        # set starting position label
        data[:, 0] = 1

        # for a copy task source data and target data should be the same
        # no gradient calculation involved
        source = Variable(data, requires_grad = True)
        target = Variable(data, requires_grad = True)
        yield Batch(source, target)

def data_generator_letter(batch_size, num_batch, initial_document_index):
    '''
    :param batch_size: sample data size of one round of training after which model parameters are updated
                       how many sentences
    :param num_batch: number of rounds of training
                      how many text documents
    '''
    fix_letter_indicator_value = 0
    max_sentence_len = 40
    for i in range(num_batch):
        original_file = 'drive/MyDrive/original_document' + '/original_document' + '_'\
                          + str(i + initial_document_index) + '.txt'
        with open(original_file, 'r') as file:
            original_sentences = file.readlines()
        encrypted_file = 'drive/MyDrive/encrypted_document' + '/encrypted_document' + '_'\
                          + str(i + initial_document_index) + '.txt'
        with open(encrypted_file, 'r') as file:
            encrypted_sentences = file.readlines()

        # Limit Number of sentences
        original_sentences = original_sentences[0:batch_size]
        #original_sentences = original_sentences[batch_size * 1 : batch_size * (1+1)]
        original_sentences = [sentence.rstrip('\n').lower() for sentence in original_sentences]

        # Limit Number of sentences
        encrypted_sentences = encrypted_sentences[0:batch_size]
        encrypted_sentences = [sentence.rstrip('\n').lower() for sentence in encrypted_sentences]

        def get_numeric(symbol):
            if symbol == 'a': return 1
            elif symbol == 'b': return 2
            elif symbol == 'c': return 3
            elif symbol == 'd': return 4
            elif symbol == 'e': return 5
            elif symbol == 'f': return 6
            elif symbol == 'g': return 7
            elif symbol == 'h': return 8
            elif symbol == 'i': return 9
            elif symbol == 'j': return 10
            elif symbol == 'k': return 11
            elif symbol == 'l': return 12
            elif symbol == 'm': return 13
            elif symbol == 'n': return 14
            elif symbol == 'o': return 15
            elif symbol == 'p': return 16
            elif symbol == 'q': return 17
            elif symbol == 'r': return 18
            elif symbol == 's': return 19
            elif symbol == 't': return 20
            elif symbol == 'u': return 21
            elif symbol == 'v': return 22
            elif symbol == 'w': return 23
            elif symbol == 'x': return 24
            elif symbol == 'y': return 25
            elif symbol == 'z': return 26
            else: return fix_letter_indicator_value

        numeric_original_text = np.empty([1,max_sentence_len])
        for sentence_index, one_sentence in enumerate(original_sentences):
            numeric_sentence = np.pad([get_numeric(item) for index, item in enumerate(one_sentence[:max_sentence_len])],\
                                      (0, max_sentence_len - len(one_sentence[:max_sentence_len])), 'constant',\
                                      constant_values=(fix_letter_indicator_value, fix_letter_indicator_value))
            numeric_original_text = np.vstack((numeric_original_text, numeric_sentence))
        # the first row of numeric_original_text is redundant
        # attach a column of zeros in front of numeric_encrypted_text as indicators of start of sentences
        row_num = numeric_original_text[1:,].shape[0]
        numeric_original_text = np.hstack((np.zeros((row_num,1)),numeric_original_text[1:,]))
        numeric_original_text = torch.from_numpy(numeric_original_text)

        numeric_encrypted_text = np.empty([1,max_sentence_len])
        for sentence_index, one_sentence in enumerate(encrypted_sentences):
            numeric_sentence = np.pad([get_numeric(item) for index, item in enumerate(one_sentence[:max_sentence_len])],\
                                      (0, max_sentence_len - len(one_sentence[:max_sentence_len])), 'constant',\
                                      constant_values=(fix_letter_indicator_value, fix_letter_indicator_value))
            numeric_encrypted_text = np.vstack((numeric_encrypted_text, numeric_sentence))
        # the first row of numeric_encrypted_text is redundant
        # attach a column of zeros in front of numeric_encrypted_text as indicators of start of sentences
        row_num = numeric_encrypted_text[1:,].shape[0]
        numeric_encrypted_text = np.hstack((np.zeros((row_num,1)),numeric_encrypted_text[1:,]))
        numeric_encrypted_text = torch.from_numpy(numeric_encrypted_text)
        # for a copy task source data and target data should be the same
        # no gradient calculation involved
        source = Variable(numeric_encrypted_text, requires_grad = False)
        target = Variable(numeric_original_text, requires_grad = False)
        source = source.type(torch.int64)
        target = target.type(torch.int64)

        #print(f"source is {source}")
        #return Batch(source, target)
        yield Batch(source, target)

def run(model, loss, epochs=10):
    '''
    :param model: model
    :param loss: loss function
    :param epochs: number of rounds of training
    '''
    for epoch in range(epochs):
        # train model, update model parameters
        model.train()
        run_epoch(data_generator_letter(10, 10, 0), model, loss)
        #evaluate model, no parameters update
        model.eval()
        run_epoch(data_generator_letter(10, 2, 10), model, loss)


'''
Load the pre-trained model(an EncoderDecoder object)
'''
# model = pickle.load(open('models/letter_decryption_core_v0.pkl','rb'))
# We cannot use the above since the model is not saved in pure dictionary format
# When loaded there exists unpredictable dependency issues

# Use the model pickled in dictionary format instead as follows
# Instantiate a model skeleton for loading
V = 27 # maximal digit + 1
the_model = make_model(V, V, N = 6)

#import pickle

#file = open('models/model_2.pkl','rb')    # Full path to the Pickle file, or simply the file name
                                    # if the file is in your current working directory
#pickle_file = pickle.load(file)     # Load Pickle file

# Load model
#the_model.load_state_dict(torch.load(pickle_file))
the_model.load_state_dict(torch.load('models/letter_decryption_core_v1.pkl'))
#the_model.load_state_dict(torch.load('models/model_2.pkl'))
#with gzip.open('models/model_2.pkl', 'rb') as ifp:
#    print(pickle.load(ifp))
'''
The rest builds the sentence translator/decoder based on the pre-trained model loaded above.
'''
# Predefined maximal sentence length
max_sentence_len = 40
def en_to_num(symbol):
    if symbol == 'a': return 1
    elif symbol == 'b': return 2
    elif symbol == 'c': return 3
    elif symbol == 'd': return 4
    elif symbol == 'e': return 5
    elif symbol == 'f': return 6
    elif symbol == 'g': return 7
    elif symbol == 'h': return 8
    elif symbol == 'i': return 9
    elif symbol == 'j': return 10
    elif symbol == 'k': return 11
    elif symbol == 'l': return 12
    elif symbol == 'm': return 13
    elif symbol == 'n': return 14
    elif symbol == 'o': return 15
    elif symbol == 'p': return 16
    elif symbol == 'q': return 17
    elif symbol == 'r': return 18
    elif symbol == 's': return 19
    elif symbol == 't': return 20
    elif symbol == 'u': return 21
    elif symbol == 'v': return 22
    elif symbol == 'w': return 23
    elif symbol == 'x': return 24
    elif symbol == 'y': return 25
    elif symbol == 'z': return 26
    else: return 0

def num_to_en(number):
    if number == 1: return 'a'
    elif number == 2: return 'b'
    elif number == 3: return 'c'
    elif number == 4: return 'd'
    elif number == 5: return 'e'
    elif number == 6: return 'f'
    elif number == 7: return 'g'
    elif number == 8: return 'h'
    elif number == 9: return 'i'
    elif number == 10: return 'j'
    elif number == 11: return 'k'
    elif number == 12: return 'l'
    elif number == 13: return 'm'
    elif number == 14: return 'n'
    elif number == 15: return 'o'
    elif number == 16: return 'p'
    elif number == 17: return 'q'
    elif number == 18: return 'r'
    elif number == 19: return 's'
    elif number == 20: return 't'
    elif number == 21: return 'u'
    elif number == 22: return 'v'
    elif number == 23: return 'w'
    elif number == 24: return 'x'
    elif number == 25: return 'y'
    elif number == 26: return 'z'
    else: return ' '

def number_to_text(numbers, input_text):
    # number_text is numeric representation of the original text without number rotation(i.e. encryption)
    number_text = np.pad([en_to_num(item) for index, item in enumerate(input_text[:max_sentence_len])],\
                                      (0, max_sentence_len - len(input_text[:max_sentence_len])), 'constant',\
                                      constant_values=(0, 0))
    # numbers is the predicted numeric representation
    text = ' '
    for index, item in enumerate(zip(numbers, number_text)):
        if item[0] == 0:
            text = text + str(num_to_en(item[1]))
        else:
            text = text + str(num_to_en(item[0]))
    return text

def text_decryption_v0(input_text):
    model.eval()
    # get source input
    source = Variable(torch.LongTensor([[en_to_num(letter) for index, letter in enumerate(input_text)]]))
    # get source mask
    # 1 for no masking
    source_mask = Variable(torch.ones(1, 1, len(input_text)))

    # get result
    result = greedy_decode(model, source, source_mask, max_len=41, start_symbol=0)
    output_text = number_to_text(result[0].tolist()[1:], input_text)
    return output_text
def text_decryption_v0(input_text):
    output_text = ''
    for index, item in enumerate(input_text):
        if item == 'a': decrypted_letter = 'v'
        elif item == 'b': decrypted_letter = 'w'
        elif item == 'c': decrypted_letter = 'x'
        elif item == 'd': decrypted_letter = 'w'
        elif item == 'e': decrypted_letter = 'z'
        elif item == 'f': decrypted_letter = 'a'
        elif item == 'g': decrypted_letter = 'b'
        elif item == 'h': decrypted_letter = 'c'
        elif item == 'i': decrypted_letter = 'd'
        elif item == 'j': decrypted_letter = 'e'
        elif item == 'k': decrypted_letter = 'f'
        elif item == 'l': decrypted_letter = 'g'
        elif item == 'm': decrypted_letter = 'h'
        elif item == 'n': decrypted_letter = 'i'
        elif item == 'o': decrypted_letter = 'j'
        elif item == 'p': decrypted_letter = 'k'
        elif item == 'q': decrypted_letter = 'l'
        elif item == 'r': decrypted_letter = 'm'
        elif item == 's': decrypted_letter = 'n'
        elif item == 't': decrypted_letter = 'o'
        elif item == 'u': decrypted_letter = 'p'
        elif item == 'v': decrypted_letter = 'q'
        elif item == 'w': decrypted_letter = 'r'
        elif item == 'x': decrypted_letter = 's'
        elif item == 'y': decrypted_letter = 't'
        elif item == 'z': decrypted_letter = 'u'
        else: decrypted_letter = item
        output_text += decrypted_letter
    return output_text

def get_letter_sequence(primary_letter, repeated_times):
    letter_sequence = ''
    for i in range(repeated_times):
        letter_sequence += primary_letter
    return letter_sequence

def get_one_letter_from_sequence(letter_sequence):
    # get the set of unique letters in the sequence
    unique_letters = set(list(letter_sequence))
    # set up a dictionary to record the repeating times of letters
    unique_letters_count = {}
    for index, item in enumerate(unique_letters):
        unique_letters_count[item] = 0
    # get the repeating times of letters
    for index, item in enumerate(letter_sequence):
        unique_letters_count[item] += 1
    # get the most frequent letter
    most_frequent_letter_index = np.argmax([unique_letters_count[item] for index, item in enumerate(unique_letters_count)])
    most_frequent_letter = list(unique_letters_count)[most_frequent_letter_index]
    return most_frequent_letter

decrypted_alphabet = []
for index, item in enumerate(['a','b','c','d','e','f','g','h',\
                              'i','j','k','l','m','n','o','p',\
                              'q','r','s','t','u','v','w','x','y','z']):
    letter_sequence = get_letter_sequence(item,10)
    decrypted_letter_sequence = text_decryption_v0(letter_sequence)
    decrypted_letter = get_one_letter_from_sequence(decrypted_letter_sequence)
    decrypted_alphabet.append(decrypted_letter)
'''
print(decrypted_alphabet) # serves as unit test
# Result: ['v', 'w', 'x', 'w', 'z', 'a', 'b', 'c', 'd', 'e',\
#'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u']
# Slightly different with the Google Colab result in local machine.
# The only missing letter in the decrypted alphabet is 'y'
'''
encrypted_alphabet = ['a','b','c','d','e','f','g','h',\
                    'i','j','k','l','m','n','o','p',\
                    'q','r','s','t','u','v','w','x','y','z']
transformer_dictionary = {item[0]: item[1] for index, item in enumerate(zip(encrypted_alphabet,\
                                                                             decrypted_alphabet))}
transformer_dictionary['d'] = 'y' # manual correction of one letter
def sentence_translator(encrypted_text):
    # convert the encrypted text to lower case
    lower_case_encrypted_text = ''
    for index, item in enumerate(encrypted_text):
        try:lower_case_encrypted_text += item.lower()
        except:lower_case_encrypted_text += item
    # text decryption using the transformer_dictionary, which is derived from the pre-trained model
    decrypted_text = ''
    for index, item in enumerate(lower_case_encrypted_text):
        # make the first letter in capital
        try:decrypted_text += transformer_dictionary[item].upper() if index == 0 else transformer_dictionary[item]
        except:decrypted_text += item
    return decrypted_text

print(sentence_translator('YmNx nx fs fuuqj.'))