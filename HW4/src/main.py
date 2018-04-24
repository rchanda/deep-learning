
import data.utils as U
from data.language import Lang
from data.dataloader import SpeechDataLoader
from data.dataset import SpeechDataset

from models.encoder import EncoderRNN
from models.decoder import DecoderRNN
from models.las import LAS

from trainer.trainer import Trainer
from loss.loss import NLLLoss



if __name__ == "__main__":
	lang = Lang()
	trans = U.tokenizeTranscripts('dev') #train
	lang.init_lang(trans)
	output_size = lang.num_items

	dev_dataset = SpeechDataset(lang, 'dev')
	dev_dataloader = SpeechDataLoader(dev_dataset, batch_size=100)

	num_layers = 3
	hidden_size = 256

	input_size = 40
	key_size = 128
	value_size = 128
	bidirectional = True
	p = 1

	embedding_size = 40

	encoder = EncoderRNN(input_size, hidden_size, key_size, value_size, num_layers, bidirectional, p)
	decoder = DecoderRNN(output_size, embedding_size, hidden_size, key_size, value_size, num_layers)

	teacher_forcing_ratio = 1.0
	las = LAS(encoder, decoder, teacher_forcing_ratio)

	num_epochs = 15
	lr = 0.01
	
	loss = NLLLoss(size_average=False)
	trainer = Trainer(loss=loss)
	trainer.train(dev_dataloader, las, lr, num_epochs)