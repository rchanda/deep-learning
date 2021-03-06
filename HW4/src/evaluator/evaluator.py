import data.utils as U
import torch

class Evaluator():
	def __init__(self, criterion):
		super(Evaluator, self).__init__()
		self.criterion = criterion


	def _eval_batch(self, model, input_variables, input_lengths, target_variables):
		decoder_outputs, ret_dict = model(input_variables, input_lengths, target_variables)
		acc_loss = self.criterion(decoder_outputs.contiguous(), target_variables[1:,:].contiguous())
		acc_loss = acc_loss.view(target_variables.size(0)-1, target_variables.size(1))
		acc_loss = acc_loss.sum(0).mean()
		return acc_loss.data.item()

	def evaluate(self, model, dev_dataloader):
		model.eval()

		num_batches = len(dev_dataloader.dataloader)
		epoch_loss = 0

		for (batch_idx, data) in enumerate(dev_dataloader.dataloader):
			input_variables, input_lengths, target_variables, target_lengths = data

			input_variables = U.var(torch.from_numpy(input_variables).float())
			target_variables = U.var(torch.from_numpy(target_variables).long())

			input_variables = input_variables.transpose(0,1)
			target_variables = target_variables.transpose(0,1)

			batch_loss = self._eval_batch(model, input_variables, input_lengths, target_variables)
			epoch_loss += batch_loss

		return (epoch_loss/num_batches)
