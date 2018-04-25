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

        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.p_layers = p_layers

        self.lstm = nn.LSTM(
            input_size=self.input_size, hidden_size=self.hidden_size, num_layers=num_layers, bidirectional=bidirectional)

        self.plstms = nn.ModuleList(
                [nn.LSTM(input_size=4*self.hidden_size, hidden_size=self.hidden_size, num_layers=num_layers, bidirectional=bidirectional) 
                            for i in range(self.p_layers)])
        #pBSLTM

        if bidirectional == True:
            self.linear_keys = nn.Linear(2*self.hidden_size, key_size) 
            self.linear_values = nn.Linear(2*self.hidden_size, key_size)
        else:
            self.linear_keys = nn.Linear(self.hidden_size, key_size) 
            self.linear_values = nn.Linear(self.hidden_size, key_size)


    def _pool_packed(self, packed_output):
        #UNPACK
        output, lengths = pad_packed_sequence(packed_output)
        # output = (L, B, 2H)

        #POOL
        output = output.transpose(1,0) 
        # output = (B, L, 2H)
        
        if output.size(1)%2 != 0:
            output = output[:,:-1,:]
        output = output.contiguous().view(output.size(0), output.size(1)//2, output.size(2)*2)
        #output = (B, L/2, 2*2H)
        output = output.transpose(0,1)
        #output = (L/2, B, 2*2H)

        lengths = lengths.data.numpy() // 2

        #PACK
        output_packed = pack_padded_sequence(output, lengths)
        return output_packed


    def forward(self, input_variable, input_lengths):
        # input_variable (L, B, 40)
        packed_input = pack_padded_sequence(input_variable, input_lengths)
        packed_output, hidden = self.lstm(packed_input)
        # hidden = (num_layers*2, B, H)

        for p in range(self.p_layers):
            packed_output = self._pool_packed(packed_output)
            packed_output, hidden = self.plstms[p](packed_output, hidden)

        output, lengths = pad_packed_sequence(packed_output)
        lengths = lengths.data.numpy()
        
        keys = self.linear_keys(output)
        # keys (L, B, K)
        values = self.linear_values(output)
        # values (L, B, V)

        assert(input_variable.size(2) == self.input_size)

        if self.bidirectional == True:
            assert(output.size(2) == 2*self.hidden_size)
        else:
            assert(output.size(2) == self.hidden_size)

        assert(output.size(0) == input_variable.size(0) // (2**self.p_layers))

        return keys, values, lengths


def _test():
    max_input_len = 40
    batch_size = 5
    input_size = 40
    input_variable = U.var(torch.randn(max_input_len, batch_size, input_size).float())
    input_lengths = [8, 16, 24, 32, 40]

    hidden_size = 100
    key_size = 10
    value_size = 10
    num_layers = 3
    bidirectional = True
    p_layers = 3

    encoder = EncoderRNN(input_size, hidden_size, key_size, value_size, num_layers, bidirectional, p_layers)
    if U.is_cuda():
        encoder = encoder.cuda()
    keys, values, lengths = encoder(input_variable, input_lengths[::-1])


if __name__ == "__main__":
    _test()

