import numpy as np
import torch.nn as nn

from attention import Attention


class DecoderRNN(nn.Module):
	def __init__(self, output_size, embedding_size, hidden_size, key_size, value_size, num_layers=2):
		self.embedding = nn.Embedding(output_size, embedding_size)
		self.rnn = nn.LSTMCell(input_size=embedding_size+value_size, hidden_size=hidden_size, num_layers=num_layers)
		self.attention = Attention(hidden_size, key_size, value_size, output_size)


	def forward_step(self, decoder_input, context, encoder_keys, encoder_values):
		# input_var = (batch_size, 1)
		# context = (batch_size, value_size)

		embedding = self.embedding(input_var)
		# embedding = (batch_size, embedding_size)
		decoder_input = torch.cat((embeding, context), dim=1)
		# inputs = (batch_size, embedding_size+value_size)

		outputs, hidden = self.rnn(decoder_input, hidden)
		outputs, context = self.attention(outputs, encoder_keys, encoder_values)

		return outputs, hidden, context


	def forward(self, teacher_forcing_ratio, decoder_targets, decoder_lens, 
						encoder_hidden, encoder_keys, encoder_values, encoder_lens):
		# decoder_inputs = (batch_size, max_len)
		batch_size = decoder_targets.size(0)
		max_len = decoder_targets.size(1)

		decoder_input = #SOS
		decoder_hidden = encoder_hidden
		decoder_outputs = []
		context = 0

		use_teacher_forcing = True if np.random.random() < teacher_forcing_ratio else False

		for timestamp in range(max_len): 
			decoder_output, decoder_hidden, context = self.forward_step(decoder_input, decoder_hidden, 
															context, encoder_keys, encoder_values)
			step_output = decoder_output.squeeze(1)
			decoder_outputs.append(step_output)

			if use_teacher_forcing:
				decoder_input = decoder_targets[:, timestamp].unsqueeze(1)
			else:
				decoder_input = symbols

		return decoder_outputs