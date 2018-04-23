import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
	def __init__(self, hidden_size, key_size, value_size, output_size):
		super(Attention, self).__init__()
		self.mask = None

		self.linear_query = nn.Linear(hidden_size, key_size)
		self.linear_out = nn.Linear(hidden_size+value_size, output_size)


	def set_mask(mask):
		self.mask = mask


	def forward(self, outputs, encoder_keys, encoder_values):
		# outputs = (batch_size, 1, hidden_size)
		# query = (batch_size, 1, key_size)
		# keys = (input_len, batch_size, key_size)
		# values = (input_len, batch_size, value_size)

		query = self.linear_query(outputs)
		assert(query.size(0) == keys.size(1))
		assert(query.size(2) == keys.size(2))

		input_len = keys.size(0)
		batch_size = keys.size(1)

		keys = keys.permute(1, 2, 0)
		values = values.permute(1, 0, 2)
		attention = torch.bmm(query, keys)
		# attention = (batch_size, 1, input_len)

		if self.mask is not None:
			attention.data.masked_fill_(self.mask, -float('inf'))

		attention_weights = F.softmax(attention.view(-1, input_len)).view(batch_size, -1, input_len)
		context = torch.bmm(attention_weights, values)
		#context = (batch_size, 1, value_size)

		combined = torch.cat((context, outputs), dim=2)
		# combined = (batch_size, 1, hidden_size+value_size)

		outputs = F.tanh(self.linear_out(combined).view(batch_size, -1, output_size))
		
		return outputs, context

