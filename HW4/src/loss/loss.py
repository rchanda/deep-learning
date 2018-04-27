import torch.nn as nn


class CrossEntropyLoss3D(nn.CrossEntropyLoss):
	def __init__(self, reduce, ignore_index):
		super(CrossEntropyLoss3D, self).__init__(reduce=reduce, ignore_index=ignore_index)
	def forward(self, input, target):
		return super(CrossEntropyLoss3D, self).forward(input.view(-1, input.size(2)), target.view(-1))