import re
import numpy as np

import constants as C
import torch
from torch.autograd import Variable

USE_CUDA = torch.cuda.is_available()

def getTranscriptsPath(mode):
	transcripts_path = (C.DATA_PATH+"%s_transcripts.npy") % (mode)
	return transcripts_path


def tokenizeTranscripts(mode):
	transcripts_path = getTranscriptsPath(mode)
	transcripts = np.load(transcripts_path)

	data = []
	for string in transcripts:
		#string = string.encode('utf-8')
		string = re.sub(r"[^A-Z0-9 ,.']+", r" ", string)
		data.append(list(string))
	data = np.asarray(data)

	assert(transcripts.shape[0] == data.shape[0])
	return data


def create_mask(lengths_array):
	array_len = len(lengths_array)
	max_len = np.max(lengths_array)

	mask = torch.ByteTensor(array_len, max_len).fill_(1)
	for i, length in enumerate(lengths_array):
		mask[i,:length] = 0
	return mask


def var(tensor):
	if USE_CUDA:
		return Variable(tensor).cuda()
	else:
		return Variable(tensor)
