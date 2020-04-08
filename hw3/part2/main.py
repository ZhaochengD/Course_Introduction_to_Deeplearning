#!/usr/bin/env python
# coding: utf-8

# In[64]:


import torch
import numpy as np
import pandas as pd
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from warpctc_pytorch import CTCLoss
from Levenshtein import distance
from ctcdecode import CTCBeamDecoder
from phoneme_list import PHONEME_MAP


# In[2]:


trainx = np.load('../data/wsj0_train' , allow_pickle=True)
trainy = np.load('../data/wsj0_train_merged_labels.npy', allow_pickle=True)
valx = np.load('../data/wsj0_dev.npy', allow_pickle=True) 
valy = np.load('../data/wsj0_dev_merged_labels.npy', allow_pickle=True)
testx = np.load('../data/wsj0_test', allow_pickle=True)


# In[65]:


BATCHSIZE = 3
NUMWORKER = 0
PHONEME_MAP = [' ']+PHONEME_MAP
OUTPATH = '../result/out.csv'


# In[66]:


len(PHONEME_MAP)


# ## 1. Preparing Dataloader

# In[5]:


class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, idx):
        if self.y is not None:  
            return torch.from_numpy(self.x[idx]).float(), torch.from_numpy(self.y[idx] + 1).float()
        else:
            return torch.from_numpy(self.x[idx]).float(), [0]

    def __len__(self):
        return self.x.shape[0]  # // 10


# In[6]:


def MyCollate(batch):
    bnum = len(batch)
    batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
    seqLen = batch[0][0].shape[0]
    batchLen = len(batch)
    channel = batch[0][0].shape[1]
    
    pack = np.zeros((seqLen, batchLen, channel))
    seqSize = np.zeros(batchLen)
    labelSize = np.zeros(batchLen)
    labels = []
    
    for index, (batchx, batchy) in enumerate(batch):
        pack[:batchx.shape[0], index, :] = batchx
        seqSize[index] = batchx.shape[0]
        labelSize[index] = len(batchy)
        labels.append(batchy)
    pack =  torch.from_numpy(pack).float()
    seqSize = torch.from_numpy(seqSize).int()
    labelSize = torch.IntTensor(labelSize)
    return pack, seqSize, labels, labelSize


# In[7]:


# , collate_fn=MyCollate, pin_memory=True
trainLoader = DataLoader(MyDataset(trainx, trainy), batch_size=BATCHSIZE, 
                          shuffle=True, num_workers=NUMWORKER, collate_fn=MyCollate, pin_memory=True)
devLoader = DataLoader(MyDataset(valx, valy), batch_size=BATCHSIZE, 
                        shuffle=True, num_workers=NUMWORKER, collate_fn=MyCollate, pin_memory=True)
testLoader = DataLoader(MyDataset(testx, None), batch_size=128, 
                        shuffle=False, collate_fn=MyCollate, num_workers=NUMWORKER, pin_memory=True)


# ## 2. Construct Model

# In[8]:


class BiLSTM(nn.Module): # expected (timelength, batch, channel=40)
    def __init__(self, inSize, hiSize):
        super(BiLSTM, self).__init__() #
        self.lstm = nn.LSTM(inSize, hiSize, num_layers=3, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hiSize * 2, hiSize) # 将双向dim降到单向dim
        self.relu = nn.ReLU()
        
    def forward(self, pack, seqSize):
        out = pack_padded_sequence(pack, seqSize)
        out, h = self.lstm(out)
        out, _ = pad_packed_sequence(out)
        out = self.linear(out)
        out = self.relu(out)
        return out


# In[9]:


class CNN(nn.Module): # expected (batch, channel=40, timelength)
    def __init__(self, inChannel, outChannel):
        super(CNN, self).__init__()
        self.bn1 = nn.BatchNorm1d(inChannel)

        self.conv1 = nn.Conv1d(inChannel, outChannel, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(outChannel)
        self.tanh = nn.Hardtanh(inplace=True)
        
        self.conv2 = nn.Conv1d(outChannel, outChannel, 3, padding=1)
        self.bn3 = nn.BatchNorm1d(outChannel)
        
    def forward(self, batch):
        out = self.bn1(batch)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.tanh(out)
        
        out = self.conv2(out)
        out = self.bn3(out)
        
        return out


# In[10]:


class Dense(nn.Module): # expected (batch, * , channel=40)
    def __init__(self, inSize, outSize):
        super(Dense, self).__init__()
        self.Linear1 = nn.Linear(inSize, outSize)
        
    def forward(self, x):
        out = self.Linear1(x)
        
        return out


# In[42]:


class MyCLDNN(nn.Module): # expected (batch, * , channel=40)
    def __init__(self, inSize, hiSize, outSize=47):
        super(MyCLDNN, self).__init__()
        self.CNN = CNN(inSize, hiSize)
        self.BiLSTM = BiLSTM(hiSize, hiSize)
        self.Dense = Dense(hiSize, outSize)
        
        self.RED = nn.Linear(inSize, hiSize) #这里对输入的x进行了升维

    def forward(self, x, seqSize):
        iniIdentity = self.RED(x)
        out = x.permute(1, 2, 0)
        out = self.CNN(out)
        out = out.permute(2, 0, 1)
        cnnIdentity = out
        out += iniIdentity
        out = self.BiLSTM(out, seqSize)
        out += cnnIdentity
        out = self.Dense(out)
        
        return out


# ## 3. Start Training

# In[79]:


class Network():
    def __init__(self, path = None, epoch = 5):
        
        self.cuda = torch.cuda.is_available()
        print(self.cuda)
        self.dev  = torch.device("cuda") if self.cuda else torch.device("cpu")
        print(self.dev)
        self.path = path

        self.trainLoader = trainLoader
        self.valLoader = devLoader
        self.testLoader = testLoader

        self.model = MyCLDNN(40, 256)
        
        print('in cuda', next(self.model.parameters()).is_cuda)
        if self.path:
            self.model.load_state_dict(torch.load(self.path))
  
        self.epoch = epoch
        self.optimizer = optim.Adam(self.model.parameters(), amsgrad=True)
        self.scheduler = ReduceLROnPlateau(self.optimizer)
        self.criterion = CTCLoss()
        
        self.lossList  = []
        self.errorList = []

        self.valLoss  = []
        self.valError = []
    
    def error(self, s1, s2):
        s1, s2 = s1.replace(' ', ''), s2.replace(' ', '')
        return distance(s1, s2)
    
    def decode(self, output, seqSize, beamWidth=40):
        decoder = CTCBeamDecoder(labels=PHONEME_MAP, blank_id=0, beam_width=beamWidth)
        output = output.permute(1, 0, 2)  # batch, seq_len, probs
        probs = F.softmax(output, dim=2).data.cpu()

        output, scores, timesteps, out_seq_len = decoder.decode(probs=probs,
                                                            seq_lens=torch.IntTensor(seqSize))
        decoded = []
        for i in range(output.size(0)):
            chrs = ""
            if out_seq_len[i, 0] != 0:
                chrs = "".join(PHONEME_MAP[o] for o in output[i, 0, :out_seq_len[i, 0]])
            decoded.append(chrs)
        return decoded
    
    def train(self, path):
        for epochIdx in range(self.epoch):
            self.model.train()
            tempLossList  = []
            tempErrorList = []
            for batchIdx, (batchx, seqSize, batchy, labelSize) in enumerate(self.trainLoader):
                batchx, batchy = Variable(batchx.to(self.dev)), Variable(torch.cat(batchy).int().to(self.dev))
                self.optimizer.zero_grad()
                output  = self.model(batchx, seqSize)
                loss = self.criterion(output, batchy, 
                                      Variable(seqSize.to(self.dev)), 
                                      Variable(labelSize))
                loss.backward(retain_graph=True)
                self.optimizer.step()
                tempLossList.append(loss.tolist())
                # tempErrorList.append(self.error(output, batchy))
                if batchIdx % 1 == 0:
                    break
                    print('----'+str(batchIdx)+'-----'+str(loss.tolist()))
            self.lossList.append(np.mean(tempLossList))
            self.errorList.append(np.sum(tempErrorList))
            self.validate()
            self.model.train()
            self.scheduler.step(self.valLoss[-1])
            print('validation', self.valError)
            self.saveModel(path+'_epoch_{}.pth'.format(epochIdx))
            print('--------------------'+str(epochIdx)+'-----'+str(np.mean(tempLossList)))

    def validate(self):
        self.model.eval()
        tempLossList  = []
        tempErrorList = []
        totalError = 0 
        for batchIdx, (batchx, seqSize, batchy, labelSize) in enumerate(self.valLoader):
            #if self.cuda:
            batchx, batchyloss = Variable(batchx.to(self.dev)), Variable(torch.cat(batchy).int().to(self.dev))
            output  = self.model(batchx, seqSize)
            loss = self.criterion(output, batchyloss, 
                                      Variable(seqSize.to(self.dev)), 
                                      Variable(labelSize.to(self.dev)))
            
            decoded = self.decode(output, seqSize)
            decodedy = ["".join(PHONEME_MAP[int(i)] for i in y) for y in batchy]
            print(len(decoded), len(decodedy))
            
            for l, m in zip(decodedy, decoded):
                e = self.error(l, m)
                totalError += e
                # tempErrorList.append(e * 100.0 / len(l))
            
            tempLossList.append(loss.tolist())
        self.valLoss.append(np.mean(tempLossList))
        self.valError.append(totalError / len(self.valLoader))

    def predict(self, path):
        with torch.no_grad():
            self.model.eval()
            # print('The validation error is ', self.valError)
            #modelIdx = self.valError.index(min(self.valError))
            #print('The validation error index is ', modelIdx) 
            #modelpath = path+'_epoch_{}.pth'.format(modelIdx)
            #print('The best model is {} and the best model is '.format(modelIdx) + modelpath)
            # self.model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
            result = []
            for batchIdx, (batchx, seqSize, _, _) in enumerate(self.testLoader):
                batchx = Variable(batchx.to(self.dev))
                output  = self.model(batchx.float(), seqSize) 
                decoded = self.decode(output, seqSize, beamWidth=100)
                print(len(result))
                result.extend(decoded)
            self.output = np.array(result)
            
    
    def saveModel(self, path):
        # path = 'mytraining'
        torch.save(self.model.state_dict(), path)


# In[77]:


def main():
    
    model = Network()
    model.train('../param/M1')

    model.predict('../param/M1')
    
    output = model.output
    
    return output


# In[78]:


a = main()


# In[47]:


b = zip([i for i in range(len(a))], a)
c = np.array(tuple(b))
df = pd.DataFrame(c, columns=['id', 'Predicted'])
df.to_csv(OUTPATH, index = False)


# ## 3. Preparing tools

# In[ ]:





# In[ ]:





# In[ ]:




