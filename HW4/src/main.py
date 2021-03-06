import data.utils as U
import constants as C

from data.language import Lang
from data.dataloader import SpeechDataLoader
from data.dataset import SpeechDataset

from models.encoder import EncoderRNN
from models.decoder import DecoderRNN
from models.las import LAS

from trainer.trainer import Trainer
from loss.loss import CrossEntropyLoss3D
import torch.nn as nn
import torch

if __name__ == "__main__":
    #U.set_random_seeds(1)

    lang = Lang()
    trans = U.tokenizeTranscripts('train')
    lang.init_lang(trans)
    output_size = lang.num_items

    batch_size = 32
    print("Starting .. ..")
    train_dataset = SpeechDataset(lang, 'train') # TODO - Change to train
    train_dataloader = SpeechDataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    dev_dataset = SpeechDataset(lang, 'dev')
    dev_dataloader = SpeechDataLoader(dev_dataset, batch_size=batch_size, shuffle=True)

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
    print(las)
    las = torch.load('saved_models/8model.pt', map_location=lambda storage, loc: storage)

    if U.use_cuda():
        las = las.cuda()
    
    num_epochs = 15
    lr = 0.0001

    criterion = CrossEntropyLoss3D(reduce=False, ignore_index=C.PAD_TOKEN_IDX)
    trainer = Trainer(criterion)
    trainer.train(train_dataloader, dev_dataloader, las, lr, num_epochs)
