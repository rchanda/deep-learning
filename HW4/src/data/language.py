import numpy as np

import constants as C
import data.utils as U
#item can be char or word

class Lang:
	def __init__(self):
		self.item2index = {C.SOS_TOKEN: C.SOS_TOKEN_IDX, C.EOS_TOKEN : C.EOS_TOKEN_IDX}
		self.index2item = {C.SOS_TOKEN_IDX : C.SOS_TOKEN, C.EOS_TOKEN_IDX : C.EOS_TOKEN, 
								C.PAD_TOKEN_IDX : C.PAD_TOKEN, C.UNK_TOKEN_IDX : C.UNK_TOKEN}
		self.num_items = 4


	def addItems(self, items):
		for item in items:
			if item not in self.item2index:
				self.item2index[item] = self.num_items
				self.index2item[self.num_items] = item
				self.num_items += 1


	def items2indices(self, items):
		indices = []
		for item in items:
			if item in self.item2index:
				indices.append(self.item2index[item])
			else:
				indices.append(C.UNK_TOKEN_IDX)
		indices = np.asarray(indices)
		return indices


	def indices2items(self, indices):
		items = []
		for index in indices:
			items.append(self.index2item[index])
		items = np.asarray(items)
		return items


	def init_lang(self, data):
		for items in data:
			self.addItems(items)



if __name__ == "__main__":
	lang = Lang()

	transcripts = U.tokenizeTranscripts('dev')
	lang.init_lang(transcripts)

	print(lang.num_items)
	print(lang.item2index)
	print(lang.index2item)

