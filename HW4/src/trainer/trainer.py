
import data.utils as U

import torch
from torch.autograd import Variable
from loss.loss import NLLLoss


class Trainer:
	def __init__(self, loss=NLLLoss):
		super(Trainer, self).__init__()
		self.loss = loss

	def _train_batch(self, model, input_variables, input_lengths, target_variables):
		loss = self.loss

		decoder_outputs = model(input_variables, input_lengths, target_variables)

		for (step, step_output) in enumerate(decoder_outputs):
			batch_size = target_variables.size(0)
			loss.eval_batch(step_output.contiguous().view(batch_size, -1), target_variables[:, step + 1])

		model.zero_grad()
		loss.backward()
		self.optimizer.step()

		return loss.get_loss()


	def train(self, train_dataloader, model, lr, num_epochs):
		self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

		for epoch in range(num_epochs):
			model.train(True)
			
			epoch_loss = 0
			steps_for_epoch = len(train_dataloader.dataloader)
			steps = 0

			for (batch_idx, data) in enumerate(train_dataloader.dataloader):
				input_variables, input_lengths, target_variables, target_lengths = data

				input_variables = U.var(torch.from_numpy(input_variables).float())
				target_variables = U.var(torch.from_numpy(target_variables).long())

				input_variables = input_variables.transpose(0,1)

				loss = self._train_batch(model, input_variables, input_lengths, target_variables)
				epoch_loss += loss

				steps += 1
				print("avg loss %f" % (epoch_loss/steps))



