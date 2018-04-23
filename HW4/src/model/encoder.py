from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn as nn


class EncoderRNN(nn.Module):
	def __init__(self, input_size=40, hidden_size=256, key_size=128, value_size=128, num_layers=3, p=1):
		self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=True)
		#pBSLTM

		self.linear_keys = nn.Linear(2*hidden_size, key_size) 
		self.linear_values = nn.Linear(2*hidden_size, key_size)

	def forward(self, inputs, inputs_lengths):
		# inputs (L, B, 40)
		inputs_packed = pack_padded_sequence(inputs, inputs_lengths)
		output, hidden = self.rnn(inputs_packed)
		output, _ = pad_packed_sequence(output)
		# output = (L, B, 2*H)
		# hidden = (num_layers*2, B, H)

		keys = self.linear_keys(output)
		#keys (L, B, K)
		values = self.linear_values(output)
		#values (L, B, V)

		return output, hidden, keys, values