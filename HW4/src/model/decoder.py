import torch.nn as nn

from attention import Attention


class DecoderRNN(nn.Module):
	def __init__(self, output_size, embedding_size, hidden_size, key_size, value_size, num_layers=2):
		self.embedding = nn.Embedding(output_size, embedding_size)
		self.rnn = nn.LSTMCell(input_size=embedding_size+value_size, hidden_size=hidden_size, num_layers=num_layers)
		self.attention = Attention(mask, hidden_size, key_size, value_size, output_size)


	def forward_step(self, inputs, context, encoder_keys, encoder_values):
		# input_var = (batch_size, 1)
		# context = (batch_size, value_size)

		embedding = self.embedding(input_var)
		# embedding = (batch_size, embedding_size)

		inputs = torch.cat((embeding, context), dim=1)
		# inputs = (batch_size, embedding_size+value_size)

		outputs, hidden = self.rnn(inputs, hidden)

		self.attention(outputs, encoder_keys, encoder_values)

	def forward(self, decoder_inputs, inputs_lens, encoder_hidden, encoder_keys, encoder_values):
		# decoder_inputs = (batch_size, max_len)
		batch_size = inputs.size(0)
		max_len = inputs.size(1) - 1

		inputs = inputs[:, 0].unsqueeze(1)
		# inputs = (batch_size, 1)

		for i in range(max_len):
			self.forward_step(inputs, context, encoder_keys, encoder_values)