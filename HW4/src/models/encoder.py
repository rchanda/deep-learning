import constants as C
import data.utils as U

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, key_size, value_size, num_layers, bidirectional, p_layers):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.key_size = key_size
        self.value_size = value_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.p_layers = p_layers

        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=True)

        for p in range(self.p_layers):
            self.add_module("pblstm"+str(p), 
                nn.LSTM(input_size=4*hidden_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=True))
        #pBSLTM

        if self.bidirectional == True:
            self.linear_keys = nn.Linear(2*hidden_size, key_size) 
            self.linear_values = nn.Linear(2*hidden_size, key_size)
        else:
            self.linear_keys = nn.Linear(hidden_size, key_size) 
            self.linear_values = nn.Linear(hidden_size, key_size)

    def _pool_packed(self, packed_output):
        #UNPACK
        output, lengths = pad_packed_sequence(packed_output)
        # output = (L, B, 2H)

        #POOL
        output = output.transpose(1,0) 
        # output = (B, L, 2H)
        
        if output.size(1)%2 != 0:
            output = output[:,:-1,:]
        output = output.contiguous().view(output.size(0), output.size(1)/2, output.size(2)*2)
        #output = (B, L/2, 2*2H)
        output = output.transpose(0,1)
        #output = (L/2, B, 2*2H)

        lengths = lengths / 2
        
        #PACK
        output_packed = pack_padded_sequence(output, lengths)
        return output_packed


    def forward(self, input_variable, input_lengths):
        # input_variable (L, B, 40)
        packed_input = pack_padded_sequence(input_variable, input_lengths)
        packed_output, hidden = self.rnn(packed_input)
        # hidden = (num_layers*2, B, H)

        for p in range(self.p_layers):
            layer_fn = getattr(self, "pblstm"+str(p))
            packed_output = self._pool_packed(packed_output)
            packed_output, hidden = layer_fn(packed_output, hidden)

        output, lengths = pad_packed_sequence(packed_output)

        keys = self.linear_keys(output)
        # keys (L, B, K)
        values = self.linear_values(output)
        # values (L, B, V)

        assert(input_variable.size(2) == self.input_size)

        if self.bidirectional == True:
            assert(output.size(2) == 2*self.hidden_size)
        else:
            assert(output.size(2) == self.hidden_size)

        return lengths, keys, values


def _test():
    input_len = 16
    batch_size = 5
    input_size = 10
    input_variable = U.var(torch.randn(input_len, batch_size, input_size).float())
    input_lengths = list(range(9, 8+batch_size+1))

    hidden_size = 5
    key_size = 10
    value_size = 10
    num_layers = 3
    bidirectional = True
    p = 1

    encoder = EncoderRNN(input_size, hidden_size, key_size, value_size, num_layers, bidirectional, p)
    if U.is_cuda():
        encoder = encoder.cuda()
    lengths, keys, values = encoder(input_variable, input_lengths[::-1])


if __name__ == "__main__":
    _test()

