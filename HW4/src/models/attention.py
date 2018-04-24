import torch
import torch.nn as nn
import torch.nn.functional as F

import data.utils as U

class Attention(nn.Module):
	def __init__(self, hidden_size, key_size, value_size, output_size):
		super(Attention, self).__init__()
		self.output_size = output_size

		self.mask = None

		self.linear_query = nn.Linear(hidden_size, key_size)

		combined_size = hidden_size+value_size
		self.linear_out = nn.Linear(combined_size, self.output_size)


	def set_mask(self, mask):
		self.mask = mask


	def forward(self, outputs, encoder_keys, encoder_values):
		# outputs = (batch_size, 1, hidden_size)
		# query = (batch_size, 1, key_size)
		# encoder_keys = (input_len, batch_size, key_size)
		# encoder_values = (input_len, batch_size, value_size)

		input_len = encoder_keys.size(0)
		batch_size = encoder_keys.size(1)

		query = self.linear_query(outputs)
		assert(query.size(0) == encoder_keys.size(1))
		assert(query.size(2) == encoder_keys.size(2))

		encoder_keys = encoder_keys.permute(1, 2, 0)
		encoder_values = encoder_values.permute(1, 0, 2)
		attention = torch.bmm(query, encoder_keys)
		# attention = (batch_size, 1, input_len)

		if self.mask is not None:
			attention.data.masked_fill_(self.mask, -float('inf'))

		attention_weights = F.softmax(attention.view(-1, input_len), dim=1).view(batch_size, -1, input_len)
		context = torch.bmm(attention_weights, encoder_values)
		# context = (batch_size, 1, value_size)

		combined = torch.cat((context, outputs), dim=2)
		# combined = (batch_size, 1, hidden_size+value_size)

		outputs = F.tanh(self.linear_out(combined).view(batch_size, -1, self.output_size))
		# outputs = (batch_size, 1, output_size)

		return context, outputs


if __name__ == "__main__":
	input_len = 4
	batch_size = 4
	hidden_size = 12
	key_size = 10
	value_size = 10
	output_size = 33

	inputs_lens = list(range(1, batch_size+1))
	mask = U.create_mask(inputs_lens).unsqueeze(1)

	attention = Attention(hidden_size, key_size, value_size, output_size)
	attention.set_mask(mask)


	outputs = U.var(torch.randn(batch_size, 1, hidden_size))
	encoder_keys = U.var(torch.randn(input_len, batch_size, key_size))
	encoder_values = U.var(torch.randn(input_len, batch_size, value_size))
	context, outputs = attention.forward(outputs, encoder_keys, encoder_values)




