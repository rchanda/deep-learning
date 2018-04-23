

import torch
from torch.autograd import Variable

USE_CUDA = torch.cuda.is_available()

class Trainer:
	def __init__(self, dataloader):
		self.dataloader = dataloader

	def train(self):
		for (batch_idx, data) in enumerate(self.dataloader.dataloader):
			feats_batch_padded, feats_batch_lens, trans_x_batch_padded, trans_y_batch_padded, trans_batch_lens = data
			print(batch_idx, feats_batch_padded.shape)



def _var(array):
	dtype = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
	return Variable(torch.LongTensor(array).type(dtype))