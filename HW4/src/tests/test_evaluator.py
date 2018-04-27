import data.utils as U
import constants as C

from data.language import Lang
from data.dataloader import SpeechDataLoader
from data.dataset import SpeechDataset

from models.encoder import EncoderRNN
from models.decoder import DecoderRNN
from models.las import LAS

from evaluator.evaluator import Evaluator
import torch.nn as nn

if __name__ == "__main__":
	batch_size = 32

	lang = Lang()
	trans = U.tokenizeTranscripts('train')
	lang.init_lang(trans)
	output_size = lang.num_items

	dev_dataset = SpeechDataset(lang, 'dev')
	dev_dataloader = SpeechDataLoader(dev_dataset, batch_size=batch_size)

	num_layers = 3
	hidden_size = 256

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

	criterion = nn.CrossEntropyLoss(size_average=False, ignore_index=C.PAD_TOKEN_IDX)

	evaluator = Evaluator(criterion)
	evaluator.evaluate(las, dev_dataloader)