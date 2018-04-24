import constants as C
import data.utils as U

import numpy as np
import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class EncoderRNN(nn.Module):
	def __init__(self, input_size, hidden_size, key_size, value_size, num_layers, bidirectional, p):
		super(EncoderRNN, self).__init__(input_size, hidden_size, key_size, value_size, num_layers, bidirectional, p)
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.key_size = key_size
		self.value_size = value_size
		self.num_layers = num_layers
		self.bidirectional = bidirectional

		self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=True)
		#pBSLTM

		if self.bidirectional == True:
			self.linear_keys = nn.Linear(2*hidden_size, key_size) 
			self.linear_values = nn.Linear(2*hidden_size, key_size)
		else:
			self.linear_keys = nn.Linear(hidden_size, key_size) 
			self.linear_values = nn.Linear(hidden_size, key_size)


	def forward(self, input_variable, input_lengths):
		print(input_variable.size(), len(input_lengths))

		# input_variable (L, B, 40)
		inputs_packed = pack_padded_sequence(input_variable, input_lengths)
		output, hidden = self.rnn(inputs_packed)
		output, _ = pad_packed_sequence(output)
		# output = (L, B, 2H)
		# hidden = (num_layers*2, B, H)
		
		# output.transpose(1,0)
		# output = (B, L, 2H)
		#if output.size(1)%2 != 0:
		#	output = output[:,:-1,:]
		
		#output.view(output.size(0), output.size(1)/2, output.size(2)*2)
		# output = (B, L/2, 2*2H)

		keys = self.linear_keys(output)
		#keys (L, B, K)
		values = self.linear_values(output)
		#values (L, B, V)
		
		assert(input_variable.size(2) == self.input_size)

		if self.bidirectional == True:
			assert(output.size(2) == 2*self.hidden_size)
		else:
			assert(output.size(2) == self.hidden_size)

		return keys, values


def _test():
	input_len = 5
	batch_size = 4
	input_size = 10
	input_variable = U.var(torch.randn(input_len, batch_size, input_size).contiguous())
	input_lengths = list(range(1, batch_size+1))

	hidden_size = 5
	key_size = 10
	value_size = 10
	num_layers = 3
	bidirectional = True
	p = 1

	encoder = EncoderRNN(input_size, hidden_size, key_size, value_size, num_layers, bidirectional, p)

	keys, values = encoder(input_variable, input_lengths[::-1])


if __name__ == "__main__":
	_test()

