import data.utils as U

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import itertools
from data.language import Lang
from data.dataloader import SpeechDataLoader
from data.dataset import SpeechDataset

class Predictor:
    def __init__(self, model):
        super(Predictor, self).__init__()
        self.model = model


    def predict(self, test_dataloader):
        self.model.eval()

        num_batches = len(test_dataloader.dataloader)

        for (batch_idx, data) in enumerate(test_dataloader.dataloader):
            input_variables, input_lengths, target_variables, target_lengths = data

            input_variables = U.var(torch.from_numpy(input_variables).float())
            target_variables = U.var(torch.from_numpy(target_variables).long())

            input_variables = input_variables.transpose(0,1)

            batch_size = target_variables.size(0)
            decoder_outputs, ret_dict = self.model(input_variables, input_lengths, target_variables)
            print(ret_dict['sequence'])


def _test():
    lang = Lang()
    trans = U.tokenizeTranscripts('train')
    lang.init_lang(trans)
    output_size = lang.num_items

    batch_size = 4
    test_dataset = SpeechDataset(lang, 'test')
    test_dataloader = SpeechDataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    las = torch.load('../models1/0model.pt', map_location=lambda storage, loc: storage)
    las.teacher_forcing_ratio = 0.0
    las = las.cuda()
    predictor = Predictor(las)
    
    predictor.predict(test_dataloader)


if __name__ == "__main__":
    _test()

