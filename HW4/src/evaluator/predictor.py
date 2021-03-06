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

import sys

class Predictor:
    def __init__(self, model, lang, criterion):
        super(Predictor, self).__init__()
        self.model = model
        self.lang = lang
        self.criterion = criterion
        self.num_random_samples = 100


    def dump_target_sequences(self, sequences, lengths, outFile, step):
        batch_size = len(lengths)
                
        for i in range(batch_size):
            length = lengths[i]
            
            tgt_id_seq = [sequences[di,i].item() for di in range(length)]

            tgt_seq = self.lang.indices2items(tgt_id_seq)
            outFile.write("%d,%s\n" % (step, ''.join(tgt_seq)))
            step += 1

        return step


    def predict(self, test_dataloader, outFile):
        self.model.eval()
        step = 0
        num_batches = len(test_dataloader.dataloader)
        outFile.write("Id,Predicted\n")

        for (batch_idx, data) in enumerate(test_dataloader.dataloader):
            input_variables, input_lengths, target_variables, target_lengths = data

            input_variables = U.var(torch.from_numpy(input_variables).float())
            target_variables = U.var(torch.from_numpy(target_variables).long())

            input_variables = input_variables.transpose(0,1)
            target_variables = target_variables.transpose(0,1)
            # T X O

            batch_size = target_variables.size(1)
            max_len = self.model.decoder.max_len
            #print(max_len)

            target_sequence_min_loss = [float('inf')] * batch_size
            target_sequences_final = U.var(torch.zeros(max_len, batch_size))
            target_lengths_final = [0] * batch_size

            for r in range(self.num_random_samples):
                self.model.teacher_forcing_ratio = 0.0
                decoder_outputs, ret_dict = self.model(input_variables, input_lengths, target_variables)
                # T X B X O

                target_lengths = ret_dict['length']
                target_sequences = torch.stack(ret_dict['sequence'])
                # T X B X 1
                target_sequences = target_sequences.squeeze(2)
                
                self.model.teacher_forcing_ratio = 1.0
                decoder_outputs, ret_dict = self.model(input_variables, input_lengths, target_sequences)
                
                acc_loss = self.criterion(decoder_outputs.contiguous(), target_sequences[1:,:].contiguous())
                acc_loss = acc_loss.view(target_sequences.size(0)-1, target_sequences.size(1))
                acc_loss = acc_loss.sum(0)/target_sequences.size(1)
            
                for b in range(batch_size):
                    if acc_loss[b] < target_sequence_min_loss[b]:
                        target_sequences_final[:,b] = target_sequences[:,b]
                        target_lengths_final[b] = target_lengths[b]


            step = self.dump_target_sequences(target_sequences_final, target_lengths, outFile, step)

            print("%d %d Batch Prediction Completed" % (batch_idx, step))

        outFile.close()


def _test(run):
    U.set_random_seeds(11785)

    lang = Lang()
    trans = U.tokenizeTranscripts('train')
    lang.init_lang(trans)
    output_size = lang.num_items

    batch_size = 1
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

    model = torch.load('../saved_models/'+run+'-model.pt', map_location=lambda storage, loc: storage)
    las.load_state_dict(model.state_dict())

    if U.use_cuda():
        las = las.cuda()

    # Prediction

    test_dataset = SpeechDataset(lang, 'test')
    test_dataloader = SpeechDataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    criterion = CrossEntropyLoss3D(reduce=False, ignore_index=C.PAD_TOKEN_IDX)
    predictor = Predictor(las, lang, criterion)
    
    outFile = open('../predictions-'+run+'.txt', 'w')
    predictor.predict(test_dataloader, outFile)


if __name__ == "__main__":
    run = sys.argv[1]
    print(run)
    _test(run)


