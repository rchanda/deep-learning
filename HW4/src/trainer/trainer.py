import data.utils as U

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import itertools

from evaluator.evaluator import Evaluator

class Trainer:
    def __init__(self, criterion):
        super(Trainer, self).__init__()
        self.criterion = criterion
        self.evaluator = Evaluator(criterion)


    def _train_batch(self, model, input_variables, input_lengths, target_variables):
        batch_size = target_variables.size(0)
        
        decoder_outputs = model(input_variables, input_lengths, target_variables)
        acc_loss = 0.0
        
        for (step, step_output) in enumerate(decoder_outputs):
            acc_loss += self.criterion(step_output, target_variables[:, step + 1])
        
        acc_loss /= (batch_size*1.0)
        
        self.optimizer.zero_grad()
        acc_loss.backward()

        params = itertools.chain.from_iterable([group['params'] for group in self.optimizer.param_groups])
        torch.nn.utils.clip_grad_norm_(params, max_norm=self.max_grad_norm)
        self.optimizer.step()

        return acc_loss.data.item()


    def train(self, train_dataloader, dev_dataloader, model, lr, num_epochs):
        self.max_grad_norm = 5
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for epoch in range(num_epochs):
            model.train(True)

            epoch_loss = 0
            num_batches = len(train_dataloader.dataloader)

            for (batch_idx, data) in enumerate(train_dataloader.dataloader):
                input_variables, input_lengths, target_variables, target_lengths = data

                input_variables = U.var(torch.from_numpy(input_variables).float())
                target_variables = U.var(torch.from_numpy(target_variables).long())

                input_variables = input_variables.transpose(0,1)

                batch_loss = self._train_batch(model, input_variables, input_lengths, target_variables)
                epoch_loss += batch_loss
                
                if batch_idx % 50 == 0:
                    print("batch %d avg_loss %f" % (batch_idx, epoch_loss/(batch_idx+1)))
                
            print("epoch %d train_epoch_loss %f" % (epoch, epoch_loss/num_batches))

            self.evaluator.evaluate(model, dev_dataloader)

            if epoch % 2 == 0:
                U.checkpoint(epoch, model)


