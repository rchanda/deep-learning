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
            nn.LSTM(input_size=embed_size, hidden_size=hidden_size, batch_first=True),
            nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True),
            nn.LSTM(input_size=hidden_size, hidden_size=embed_size, batch_first=True)
        ])
        self.decoder = nn.Linear(embed_size, vocab_size)

        self.decoder.weight = self.encoder.weight

    def forward(self, inputs, forward=0, stochastic=False):
        h = inputs  # (n, t)
        h = self.encoder(h)  # (n, t, c)
        states = []
        for rnn in self.rnns:
            h, state = rnn(h)
            states.append(state)
        h = self.decoder(h)
        if stochastic:
            gumbel = Variable(sample_gumbel(shape=h.size(), out=h.data.new()))
            h += gumbel
        logits = h

        if forward > 0:
            outputs = []
            h = torch.max(logits[:, -1:, :], dim=2)[1]
            for i in range(forward):
                h = self.encoder(h)
                for j, rnn in enumerate(self.rnns):
                    h, state = rnn(h, states[j])
                    states[j] = state
                h = self.decoder(h)
                if stochastic:
                    gumbel = Variable(
                        sample_gumbel(
                            shape=h.size(),
                            out=h.data.new()))
                    h += gumbel
                outputs.append(h)
                h = torch.max(h, dim=2)[1]
            logits = torch.cat([logits] + outputs, dim=1)
        return logits


def sample_gumbel(shape, eps=1e-10, out=None):
    U = out.resize_(shape).uniform_() if out is not None else torch.rand(shape)
    return - torch.log(eps - torch.log(U + eps))


def dataset_path(name):
    return os.path.join(
        os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__))),
        'dataset',
        name)


def generation(inp, forward):
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
    model.eval()
    logits = model(inputs, forward=20, stochastic=True)
    classes = torch.max(logits, dim=2)[1].data.cpu().numpy()
    return classes[:, -forward:]
