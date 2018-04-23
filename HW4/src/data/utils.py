import re
import numpy as np

import constants as C

def getTranscriptsPath(mode):
	transcripts_path = (C.DATA_PATH+"%s_transcripts.npy") % (mode)
	return transcripts_path


def tokenizeTranscripts(mode):
	transcripts_path = getTranscriptsPath(mode)
	transcripts = np.load(transcripts_path)

	data = []
	for string in transcripts:
		string = string.encode('utf-8')
		string = re.sub(r"[^A-Z0-9 ,.']+", r" ", string)
		data.append(list(string))
	data = np.asarray(data)

	assert(transcripts.shape[0] == data.shape[0])
	return data

