import re
import numpy as np
import os

import constants as C
import torch
from torch.autograd import Variable

import os

def getTranscriptsPath(mode):
    transcripts_path = (C.DATA_PATH+"%s_transcripts.npy") % (mode)
    return transcripts_path


def tokenizeTranscripts(mode):
    transcripts_path = getTranscriptsPath(mode)
    transcripts = np.load(transcripts_path)

    data = []
    for string in transcripts:
        #string = string.encode('utf-8')
        #string = re.sub(r"[^A-Z0-9 ,.']+", r" ", string)
        data.append(list(string))
    data = np.asarray(data)

    assert(transcripts.shape[0] == data.shape[0])
    return data


def create_mask(lengths_array):
    array_len = len(lengths_array)
    max_len = int(np.max(lengths_array))
    
    mask = torch.ones(array_len, max_len)
    
    if use_cuda():
        mask = mask.type(torch.cuda.ByteTensor)
    else:
        mask = mask.type(torch.ByteTensor)
    
    for i, length in enumerate(lengths_array):
        mask[i,:length] = 0
    return mask


def use_cuda():
    return torch.cuda.is_available()


def var(tensor):
    if use_cuda():
        return Variable(tensor).cuda()
    else:
        return Variable(tensor)


def set_random_seeds(seed):
    np.random.seed(0)
    torch.manual_seed(42)


def checkpoint(epoch, model):
    torch.save(model, os.path.join('saved_models/', str(epoch)+C.MODEL_NAME))


if __name__ == "__main__":
    lengths_array = [2,3,5]
    mask = create_mask(lengths_array)

    print(lengths_array, mask)

    attention = var(torch.randn(3,5))
    print(attention)
    attention.data.masked_fill_(mask, -float('inf'))
    print(attention)

    print(F.softmax(attention, dim=1))


