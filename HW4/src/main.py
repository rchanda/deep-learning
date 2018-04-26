import data.utils as U
import constants as C

from data.language import Lang
from data.dataloader import SpeechDataLoader
from data.dataset import SpeechDataset

from models.encoder import EncoderRNN
from models.decoder import DecoderRNN
from models.las import LAS

from trainer.trainer import Trainer

if __name__ == "__main__":
    U.set_random_seeds(1)

    lang = Lang()
    trans = U.tokenizeTranscripts('train') #train
    lang.init_lang(trans)
    output_size = lang.num_items

    batch_size = 32
    print("Starting .. ..")
    train_dataset = SpeechDataset(lang, 'train')
    print("Dataset Loading Completed")
    train_dataloader = SpeechDataLoader(train_dataset, batch_size=batch_size)
    print("DataLoader Compeleted")

    num_layers = 3
    hidden_size = 128

    input_size = 40
    key_size = 128
    value_size = 128
    bidirectional = True
    p = 3

    embedding_size = 128

    encoder = EncoderRNN(input_size, hidden_size, key_size, value_size, num_layers, bidirectional, p)
    decoder = DecoderRNN(output_size, embedding_size, hidden_size, key_size, value_size, num_layers)

    teacher_forcing_ratio = 1.0
    las = LAS(encoder, decoder, teacher_forcing_ratio)

    if U.is_cuda():
        las = las.cuda()

    num_epochs = 15
    lr = 0.001

    trainer = Trainer()
    trainer.train(train_dataloader, las, lr, num_epochs)
