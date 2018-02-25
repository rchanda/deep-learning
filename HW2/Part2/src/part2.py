import numpy as np
import torch
import torch.nn as nn

import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable


class SpeechDataset(Dataset):
	def __init__(self, phonemes, labels):
		self.phonemes = phonemes
		self.labels = labels
		self.length = len(phonemes)

	def __getitem__(self, index):
		return (self.phonemes[index], self.labels[index])

	def __len__(self):
		return self.length


def collate_fn(data):
    phonemes, labels = zip(*data)

    max_len = 0
    for phoneme in phonemes:
        max_len = max(max_len, len(phoneme))

    phonemes_padded = []
    masks = []

    for phoneme in phonemes:
        mask = [1.0 if i <len(phoneme) else 0.0 for i in range(max_len)]
        masks.append(mask)

        pad = max_len - len(phoneme)
        phoneme_padded = np.pad(phoneme, ((0, pad), (0,0)), 'constant')
        phonemes_padded.append(phoneme_padded.T)

    masks = torch.from_numpy(np.asarray(masks))
    labels = torch.from_numpy(np.asarray(labels)).long()
    phonemes = torch.from_numpy(np.asarray(phonemes_padded))

    return (phonemes, labels, masks)


use_cuda = torch.cuda.is_available()
pin_memory = use_cuda

dev_feats = np.load('data/dev-phonemes-feats.npy')
dev_labels = np.load('data/dev-phonemes-labels.npy')

train_feats = np.load('data/train-phonemes-feats.npy')
train_labels = np.load('data/train-phonemes-labels.npy')

test_feats = np.load('data/test-phonemes-feats.npy')


batch_size = 512
num_workers = 4
learning_rate = 0.001
num_epochs = 50


class CNN(nn.Module):

    def avgPool(self, x, mask):
        mul = mask.unsqueeze(1)*x.data.cpu().double()
        mul_sum = torch.sum(mul, 2)

        mask_sum = torch.sum(mask, 1)
        mask_sum[mask_sum==0]=1.0
        mask_sum = mask_sum.unsqueeze(1)

        result = (mul_sum / mask_sum).cuda()
        return Variable(result, requires_grad=True)

    def __init__(self):
        super(CNN, self).__init__()
        #self.dp1 = nn.Dropout(p=0.2)
        self.cn1 = nn.Conv1d(40, 46, 10, stride=1, padding=0)
        #self.dp2 = nn.Dropout(p=0.2)
        self.cn2 = nn.Conv1d(46, 46, 2, stride=1, padding=5)
        self.ln1 = nn.Linear(46, 46).cuda().double()

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                torch.nn.init.xavier_normal(m.weight)

    def forward(self, x, mask):
        out = x
        #out = self.dp1(out)
        out = F.relu(self.cn1(out))
        #out = self.dp2(out)
        out = F.relu(self.cn2(out))
        out = self.avgPool(out, mask).double()
        #print(type(out))
        out = self.ln1(out).double()
        #print(type(out))
        return out


dset_dev = SpeechDataset(dev_feats, dev_labels)
dev_loader = DataLoader(dset_dev, shuffle=True, batch_size=batch_size,
    collate_fn=collate_fn,
    num_workers=num_workers, pin_memory=pin_memory)


dset_train = SpeechDataset(train_feats, train_labels)
train_loader = DataLoader(dset_train, shuffle=False, batch_size=batch_size,
    collate_fn=collate_fn,
    num_workers=num_workers, pin_memory=pin_memory)


dset_test = SpeechDataset(test_feats, None)
test_loader = DataLoader(dset_test, shuffle=False, batch_size=batch_size,
    collate_fn=collate_fn,
    num_workers=num_workers, pin_memory=pin_memory)


cnn = CNN()

if use_cuda:
    cnn.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)


for epoch in range(num_epochs):
    train_loss = 0.0
    correct = 0
    total = 0

    cnn.train()

    for batch_idx, (phonemes, labels, masks) in enumerate(train_loader):
        if use_cuda:
            phonemes = phonemes.cuda()
            labels = labels.cuda()

        inputs = Variable(phonemes)
        labels = Variable(labels)

        optimizer.zero_grad()
        outputs = cnn(inputs, masks)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += predicted.eq(labels.data).cpu().sum()

        if (batch_idx % 100) == 0:
            print('Batch Index %d Loss %f Accuracy %f' % (batch_idx, train_loss/(batch_idx+1), 100.0*correct/total))


    print('Total Loss %f Accuracy %f' % (train_loss/len(dev_loader), 100.0*correct/total))
    torch.save(cnn.state_dict(), str(epoch)+'model.pkl')

    cnn.eval()
    dev_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (phonemes, labels, masks) in enumerate(dev_loader):
        if use_cuda:
            phonemes = phonemes.cuda()
            labels = labels.cuda()

        inputs = Variable(phonemes)
        labels = Variable(labels)

        outputs = cnn(inputs, masks)
        loss = criterion(outputs, labels)

        dev_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += predicted.eq(labels.data).cpu().sum()

        if (batch_idx % 100) == 0:
            print('Batch Index %d Loss %f Accuracy %f' % (batch_idx, dev_loss/(batch_idx+1), 100.0*correct/total))


    print('Total Loss %f Accuracy %f' % (dev_loss/len(dev_loader), 100.0*correct/total))

    f = open(str(epoch)+'output.txt', 'w')
    f.write('id,label\n')

    for index, (data, labels, mask) in enumerate(test_loader):
        data = Variable(data).cuda()

        output = cnn(data)
        _, predicted = torch.max(output.data, 1)
        f.write("%d,%d\n" % (index, predicted))
