import data.utils as U
import constants as C

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

from loss.loss import CrossEntropyLoss3D



class Predictor:
    def __init__(self, model, lang, criterion):
        super(Predictor, self).__init__()
        self.model = model
        self.lang = lang
        self.criterion = criterion


    def dump_target_sequences(self, sequences, lengths, outFile, batch_idx):
        batch_size = len(lengths)
        step = batch_size*batch_idx

        for i in range(batch_size):
            length = lengths[i]

            tgt_id_seq = [sequences[di,i,0].item() for di in range(length)]
            tgt_seq = self.lang.indices2items(tgt_id_seq)
            outFile.write("%d,%s\n" % (step, ''.join(tgt_seq)))
            step += 1


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
            target_variables = target_variables.transpose(0,1)
            # T X O

            batch_size = target_variables.size(0)

            self.model.teacher_forcing_ratio = 0.0
            decoder_outputs, ret_dict = self.model(input_variables, input_lengths, target_variables)
            # T X B X O

            target_lengths = ret_dict['length']
            target_sequences = torch.stack(ret_dict['sequence'])
            # T X B X 1
            target_sequences = target_sequences.squeeze(2)
            
            self.model.teacher_forcing_ratio = 1.0
            decoder_outputs, ret_dict = self.model(input_variables, input_lengths, target_sequences)

            acc_loss = self.criterion(decoder_outputs.contiguous(), target_sequences[:,1:].contiguous())
            acc_loss = acc_loss.view(target_variables.size(0), target_sequences.size(1))
            acc_loss = acc_loss.sum(0)

            print(acc_loss)
            self.dump_target_sequences(target_sequences, target_lengths, outFile, batch_idx)

            print("%d Batch Prediction Completed" % (batch_idx))

        outFile.close()


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

    model = torch.load('saved_models/4model.pt', map_location=lambda storage, loc: storage)
    las.load_state_dict(model.state_dict())

    #las = torch.load('../data/saved_models_0.0001/7model.pt', map_location=lambda storage, loc: storage)
    #las.teacher_forcing_ratio = 0.0

    if U.use_cuda():
        las = las.cuda()

    # Prediction

    test_dataset = SpeechDataset(lang, 'test')
    test_dataloader = SpeechDataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    criterion = CrossEntropyLoss3D(reduce=False, ignore_index=C.PAD_TOKEN_IDX)
    predictor = Predictor(las, lang, criterion)
    
    outFile = open('predictions.txt', 'w')
    predictor.predict(test_dataloader, outFile)


if __name__ == "__main__":
    _test()


