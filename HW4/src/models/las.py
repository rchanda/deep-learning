import numpy as np
import torch
import torch.nn as nn

import data.utils as U
from models.decoder import DecoderRNN
from models.encoder import EncoderRNN


class LAS(nn.Module):
	def __init__(self, encoder, decoder, teacher_forcing_ratio):
		super(LAS, self).__init__()
		self.encoder = encoder
		self.decoder = decoder
		self.teacher_forcing_ratio = teacher_forcing_ratio


	def forward(self, input_variable, input_lengths, target_variable):
		encoder_keys, encoder_values, input_lengths = self.encoder(input_variable, input_lengths)
		decoder_outputs = self.decoder(target_variable, encoder_keys, encoder_values, input_lengths, self.teacher_forcing_ratio)

		return decoder_outputs


def _test():
	max_input_len = 4
	max_target_len = 10

	batch_size = 4
	hidden_size = 10
	key_size = 10
	value_size = 10
	embedding_size = 4

	input_size = 40
	output_size = 33

	teacher_forcing_ratio = 1.0
	num_layers = 3
	bidirectional = True
	p = 1

	input_variable = U.var(torch.randn(max_input_len, batch_size, input_size))
	input_lengths = sorted(np.random.randint(low=1, high=max_input_len, size=batch_size), reverse=True)
	
	target_variable = U.var(torch.ones(batch_size, max_target_len).long())

	encoder = EncoderRNN(input_size, hidden_size, key_size, value_size, num_layers, bidirectional, p)
	decoder = DecoderRNN(output_size, embedding_size, hidden_size, key_size, value_size, num_layers)

	las = LAS(encoder, decoder, teacher_forcing_ratio)
	decoder_outputs = las(input_variable, input_lengths, target_variable)

    
if __name__ == "__main__":
	_test()


