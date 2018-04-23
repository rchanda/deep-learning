
import data.utils as U
from data.language import Lang
from data.dataloader import SpeechDataLoader
from data.dataset import SpeechDataset

from trainer.trainer import Trainer




if __name__ == "__main__":
	lang = Lang()
	trans = U.tokenizeTranscripts('dev')
	lang.init_lang(trans)

	dev_dataset = SpeechDataset(lang, 'dev')
	dev_dataloader = SpeechDataLoader(dev_dataset, batch_size=100)
	trainer = Trainer(dev_dataloader)
	trainer.train()