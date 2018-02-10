
# coding: utf-8

# In[1]:


import numpy as np
import os

class WSJ():  
    def __init__(self):
        self.dev_set = None
        self.train_set = None
        self.test_set = None
  
    @property
    def dev(self):
        if self.dev_set is None:
            self.dev_set = load_raw(os.environ['WSJ_PATH'], 'dev')
        return self.dev_set

    @property
    def train(self):
        if self.train_set is None:
            self.train_set = load_raw(os.environ['WSJ_PATH'], 'train')
        return self.train_set
  
    @property
    def test(self):
        if self.test_set is None:
            self.test_set = (np.load(os.path.join(os.environ['WSJ_PATH'], 'test.npy'), encoding='bytes'), None)
        return self.test_set
    
def load_raw(path, name):
    return (
        np.load(os.path.join(path, '{}.npy'.format(name)), encoding='bytes'), 
        np.load(os.path.join(path, '{}_labels.npy'.format(name)), encoding='bytes')
    )


# In[2]:


os.environ['WSJ_PATH']='../data'
loader = WSJ()
trainX, trainY = loader.train
devX, devY = loader.dev
testX, testY = loader.test


# In[3]:


import torch
from torch.utils.data.dataset import Dataset
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn


# In[4]:


# Hyper Parameters 
k = 15
input_size = (2*k+1)*40
hidden_size = 200
num_classes = 138
num_epochs = 5
batch_size = 64
learning_rate = 0.001
momentum = 0.9


# In[27]:


class WSJ_Dataset(Dataset):
    
    def pad_utterances(self, X, k):
        padded_X = []
        n = len(X)

        for i in range(n):
            padded_X.append(np.pad(X[i], [(k,k),(0,0)], mode='constant'))
        
        return padded_X

    def __init__(self, X, Y, k):
        utterance_table = []
        frame_table = []
        
        for i in range(len(X)):
            n = len(X[i])
            utterance_table.extend([i]*n)
            frame_table.extend([i for i in range(n)])
        
        self.u_table = utterance_table
        self.f_table = frame_table
        
        assert(len(utterance_table) == len(frame_table))
        self.length = len(utterance_table)
        
        self.padded_X = self.pad_utterances(X, k)
        self.Y = Y
        self.k = k
        
    def __getitem__(self, index):
        u_index = self.u_table[index]
        f_index = self.f_table[index]
        
        utterance = self.padded_X[u_index]
        data = utterance[f_index:(f_index + 2*self.k+1)].flatten()
        assert(len(data)/40 == (2*self.k+1))
        
        if self.Y is None:
            label = 0
        else:
            label = self.Y[u_index][f_index]
        
        return (data, label)

    def __len__(self):
        return self.length


# In[7]:


# Neural Network Model (2 hidden layers)
class MLPNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.fc2 = nn.Linear(hidden_size, hidden_size) 
        self.fc3 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


# In[8]:


dset_train = WSJ_Dataset(trainX, trainY, k)
train_loader = torch.utils.data.DataLoader(dset_train, batch_size=batch_size, shuffle=True)


# In[9]:


dset_dev = WSJ_Dataset(devX, devY, k)
dev_loader = torch.utils.data.DataLoader(dset_dev, shuffle=True, batch_size=batch_size)


# In[28]:


dset_test = WSJ_Dataset(testX, testY, k)
test_loader = torch.utils.data.DataLoader(dset_test, shuffle=False)


# In[11]:


net = MLPNet(input_size, hidden_size, num_classes)

criterion = nn.CrossEntropyLoss()  
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
#optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)  

# In[13]:


#Train the model

for epoch in range(num_epochs):
    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = Variable(data), Variable(labels)
        
        optimizer.zero_grad()
        outputs = net(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if (batch_idx+1) % 100 == 0:
            print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' 
                   %(epoch+1, num_epochs, batch_idx+1, len(train_loader), loss.data[0]))

    torch.save(net.state_dict(), str(epoch)+'model.pkl')

    correct = 0
    total = 0

    for data, labels in dev_loader:
        data = Variable(data)
        output = net(data)
        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('Accuracy of the network on the dev set: %d %%' % (100 * correct / total))

    f = open(str(epoch)+'output.txt', 'w')
    f.write('id,label\n')

    for index, (data, labels) in enumerate(test_loader):
        data = Variable(data)
        output = net(data)
        _, predicted = torch.max(output.data, 1)
        f.write("%d,%d\n" % (index, predicted))

    f.close()

