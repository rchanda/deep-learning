import data.utils as U

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import itertools
from data.language import Lang
from data.dataloader import SpeechDataLoader
from data.dataset import SpeechDataset
import pdb

from models.encoder import EncoderRNN
from models.decoder import DecoderRNN
from models.las import LAS

class Predictor:
    def __init__(self, model, lang):
        super(Predictor, self).__init__()
        self.model = model
        self.lang = lang


    def predict(self, test_dataloader, outFile):
        self.model.eval()

        num_batches = len(test_dataloader.dataloader)
        step = 0
        outFile.write("Id,Predicted\n")

        for (batch_idx, data) in enumerate(test_dataloader.dataloader):
            input_variables, input_lengths, target_variables, target_lengths = data

            input_variables = U.var(torch.from_numpy(input_variables).float())
            target_variables = U.var(torch.from_numpy(target_variables).long())

            input_variables = input_variables.transpose(0,1)

            batch_size = target_variables.size(0)
            decoder_outputs, ret_dict = self.model(input_variables, input_lengths, target_variables)

            for i in range(batch_size):
                length = ret_dict['length'][i]-1

                tgt_id_seq = [ret_dict['sequence'][di][i].item() for di in range(length)]
                tgt_seq = self.lang.indices2items(tgt_id_seq)
                outFile.write("%d,%s\n" % (step, ''.join(tgt_seq)))
                step += 1

            print("%d Batch Prediction Completed" % (batch_idx))


def _test():
    #U.set_random_seeds(1)

    lang = Lang()
    trans = U.tokenizeTranscripts('train')
    lang.init_lang(trans)
    output_size = lang.num_items

    batch_size = 32
    print("Starting .. ..")

    num_layers = 3
    hidden_size = 256

    input_size = 40
    key_size = 128
    value_size = 128
    bidirectional = True
    p = 3

    embedding_size = 128
    max_len = 496

    encoder = EncoderRNN(input_size, hidden_size, key_size, value_size, num_layers, bidirectional, p)
    decoder = DecoderRNN(output_size, embedding_size, hidden_size, key_size, value_size, num_layers, max_len)

    teacher_forcing_ratio = 1.0
    las = LAS(encoder, decoder, teacher_forcing_ratio)

    model = torch.load('../data/saved_models_0.001/6model.pt', map_location=lambda storage, loc: storage)
    las.load_state_dict(model.state_dict())
    las.teacher_forcing_ratio = 0.0

    #las = torch.load('../data/saved_models_0.0001/7model.pt', map_location=lambda storage, loc: storage)
    #las.teacher_forcing_ratio = 0.0

    if U.use_cuda():
        las = las.cuda()

    # Prediction

    test_dataset = SpeechDataset(lang, 'test')
    test_dataloader = SpeechDataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    predictor = Predictor(las, lang)
    
    outFile = open('predictions.txt', 'w')
    predictor.predict(test_dataloader, outFile)


if __name__ == "__main__":
    _test()


