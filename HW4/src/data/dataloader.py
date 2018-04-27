import math
import numpy as np

from torch.utils.data import DataLoader

import constants as C
import data.utils as U
from data.language import Lang
from data.dataset import SpeechDataset

class SpeechDataLoader():

	def __init__(self, dataset, batch_size, shuffle):
		num_workers = 4
		pin_memory = True

		self.dataloader = DataLoader(dataset, shuffle=shuffle, batch_size=batch_size,
			collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory)
		print("DataLoader Loading Completed")


def pad_batch_data(batch_data):
	batch_lens = []
	max_batch_len = 0
	for data in batch_data:
		data_len = data.shape[0]
		batch_lens.append(data_len)
		max_batch_len = max(max_batch_len, data_len)

	batch_padded = []
	for data in batch_data:
		data_len = data.shape[0] 
		pad = max_batch_len - data_len

		if len(data.shape) == 1:
			data_padded = np.pad(data, [(0,pad)], 'constant', constant_values = C.PAD_TOKEN_IDX)
		else:
			data_padded = np.pad(data, [(0,pad), (0,0)], 'constant', constant_values = C.PAD_TOKEN_IDX)

		batch_padded.append(data_padded)
	
	batch_padded = np.asarray(batch_padded)
	batch_lens = np.asarray(batch_lens)

	return batch_padded, batch_lens


def collate_fn(data):
	data = sorted(data, key=lambda x: x[0].shape[0], reverse=True)
	[feats_batch, trans_batch] = zip(*data)

	feats_batch_padded, feats_batch_lens = pad_batch_data(list(feats_batch))
	trans_batch_padded, trans_batch_lens = pad_batch_data(list(trans_batch))

	return (feats_batch_padded, feats_batch_lens, trans_batch_padded, trans_batch_lens)


if __name__ == "__main__":
	lang = Lang()
	trans = U.tokenizeTranscripts('dev')
	lang.init_lang(trans)

	dev_dataset = SpeechDataset(lang, 'dev')
	dev_dataloader = SpeechDataLoader(dev_dataset, batch_size=100)

	for (batch_idx, data) in enumerate(dev_dataloader.dataloader):
		feats_batch_padded, feats_batch_lens, trans_batch_padded, trans_batch_lens = data
		print(batch_idx, feats_batch_padded.shape)




