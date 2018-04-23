import numpy as np

import constants as C
import utils as U
from language import Lang

from torch.utils.data.dataset import Dataset

class SpeechDataset(Dataset):
    def __init__(self, lang, mode):
        features_path = (C.DATA_PATH+"%s.npy") % (mode)
        self.feats = np.load(features_path)

        self.trans = []
        if mode is not 'test':
            trans_tokenized = U.tokenizeTranscripts(mode)

            for trans_items in trans_tokenized:
                trans_indices = lang.items2indices(trans_items)
                self.trans.append(trans_indices)
            self.trans = np.asarray(self.trans)

        assert(self.feats.shape[0] == self.trans.shape[0])
        self.len = self.feats.shape[0]

    def __getitem__(self, index):
        trans_x = np.append([C.SOS_TOKEN_IDX], self.trans[index])
        trans_y = np.append(self.trans[index], [C.EOS_TOKEN_IDX])
        return (self.feats[index], trans_x, trans_y)

    def __len__(self):
        return self.len


if __name__ == "__main__":
    lang = Lang()

    trans = U.tokenizeTranscripts('dev')
    lang.init_lang(trans)

    dataset = SpeechDataset(lang, 'dev')
    assert(len(dataset) == 1139)
    print(dataset[0])
    print(''.join(lang.indices2items(dataset[0][1])))
