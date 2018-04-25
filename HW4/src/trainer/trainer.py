import data.utils as U
import constants as C

import torch
import torch.nn as nn
from torch.autograd import Variable


class Trainer:
    def __init__(self):
        super(Trainer, self).__init__()
        
    def _train_batch(self, model, input_variables, input_lengths, target_variables):
        batch_size = target_variables.size(0)
        
        decoder_outputs = model(input_variables, input_lengths, target_variables)
        acc_loss = 0.0
        
        for (step, step_output) in enumerate(decoder_outputs):
            acc_loss += self.criterion(step_output.contiguous().view(batch_size, -1), target_variables[:, step + 1])
        
        acc_loss /= batch_size
        
        self.optimizer.zero_grad()
        acc_loss.backward()
        self.optimizer.step()

        return acc_loss.data[0]


    def train(self, train_dataloader, model, lr, num_epochs):
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.NLLLoss(size_average=False, ignore_index=C.PAD_TOKEN_IDX)
        
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
           
            print("epoch %d epoch_loss %f" % (epoch, epoch_loss/num_batches))


