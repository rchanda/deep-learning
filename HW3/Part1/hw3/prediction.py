import numpy as np
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import random

from torch.optim import Adam
from torch.autograd import Variable


class RLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(RLSTM, self).__init__()
        self.encoder = nn.Embedding(vocab_size, embed_size)
        self.rnns = nn.ModuleList([
            nn.LSTM(input_size=embed_size, hidden_size=hidden_size),
            nn.LSTM(input_size=hidden_size, hidden_size=hidden_size),
            nn.LSTM(input_size=hidden_size, hidden_size=embed_size)
        ])
        self.decoder = nn.Linear(embed_size, vocab_size)

    def forward(self, inputs, forward=0):
        h = inputs  # (n, t)
        h = self.encoder(h)  # (n, t, c)
        states = []
        for rnn in self.rnns:
            h, state = rnn(h)
            states.append(state)
        h = self.decoder(h)
        logits = h

        if forward > 0:
            outputs = []
            h = torch.max(logits[:, -1:, :], dim=2)[1] + 1
            for i in range(forward):
                h = self.embedding(h)
                for j, rnn in enumerate(self.rnns):
                    h, state = rnn(h, states[j])
                    states[j] = state
                h = self.projection(h)
                outputs.append(h)
                h = torch.max(h, dim=2)[1] + 1
            logits = torch.cat([logits] + outputs, dim=1)
        return logits


def dataset_path(name):
    return os.path.join(
        os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__))),
        'dataset',
        name)


def prediction(inp):
    inp = inp.T
    embed_size = 400
    hidden_size = 100

    model = RLSTM(33278, embed_size, hidden_size)

    if torch.cuda.is_available():
        model = model.cuda()

    filepath = os.path.abspath(os.path.join(__file__, '../4-model.pkl'))
    model.load_state_dict(
        torch.load(
            filepath,
            map_location=lambda storage,
            loc: storage))

    model.eval()

    inputs = torch.from_numpy(inp).long()

    if torch.cuda.is_available():
        inputs = inputs.cuda()

    inputs = Variable(inputs)
    outputs = model(inputs)
    outputs = outputs.data.cpu().numpy()
    outputs = outputs[-1, :]
    print(inputs.shape, outputs.shape)
    return outputs
