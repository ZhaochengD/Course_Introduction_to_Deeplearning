#!/usr/bin/env python
# coding: utf-8

# In[64]:


import torch
import numpy as np
import pandas as pd
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from torch.nn import CTCLoss
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


BATCHSIZE = 64
NUMWORKER = 4
HIDDEN = 256
EPOCH = 50
PHONEME_MAP = [' ']+PHONEME_MAP
OUTPATH = '../result/out.csv'


# In[5]:


class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.phonemes = sum([len(i) for i in y]) if y is not None else 0

    def __getitem__(self, idx):
        if self.y is not None:  
            return torch.from_numpy(self.x[idx]).float(), torch.from_numpy(self.y[idx] + 1).int()
        else:
            return torch.from_numpy(self.x[idx]).float(), np.array([0])

    def __len__(self):
        return self.x.shape[0]  # // 10


# In[6]:


def MyCollate(batch):
    batchLen = len(batch)
    batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
    seqLen = batch[0][0].shape[0]
    channel = batch[0][0].shape[1]
    
    pack = np.zeros((seqLen, batchLen, channel))
    seqSize = np.zeros(batchLen)
    labelSize = np.zeros(batchLen)
    labels = []
    
    for index, (batchx, batchy) in enumerate(batch):
        pack[:batchx.shape[0], index, :] = batchx
        seqSize[index] = batchx.shape[0]
        labelSize[index] = batchy.shape[0]
        labels.append(batchy)
    pack =  torch.from_numpy(pack).float()
    seqSize = torch.from_numpy(seqSize).int()
    labelSize = torch.IntTensor(labelSize)
    return pack, seqSize, labels, labelSize


# In[7]:
valSet = MyDataset(valx, valy)

# , collate_fn=MyCollate, pin_memory=True
trainLoader = DataLoader(MyDataset(trainx, trainy), batch_size=BATCHSIZE, 
                          shuffle=True, num_workers=NUMWORKER, collate_fn=MyCollate, pin_memory=True)
devLoader = DataLoader(valSet, batch_size=BATCHSIZE, 
                        shuffle=True, num_workers=NUMWORKER, collate_fn=MyCollate, pin_memory=True)
testLoader = DataLoader(MyDataset(testx, None), batch_size=1, 
                        shuffle=False, collate_fn=MyCollate, num_workers=NUMWORKER, pin_memory=True)


# ## 2. Construct Model

class Seqbn(nn.Module):
    def __init__(self, hiSize):
        super(Seqbn, self).__init__()
        self.bn = nn.BatchNorm1d(hiSize)
        
    def forward(self, batch):
        seqn, batchn = batch.size(0), batch.size(1)
        batch = batch.view(seqn * batchn, -1)
        batch = self.bn(batch)
        batch = batch.view(seqn, batchn, -1)
        return batch

class BiLSTM(nn.Module): # expected (timelength, batch, channel=40)
    def __init__(self, inSize, hiSize):
        super(BiLSTM, self).__init__() #
        self.lstm1 = nn.LSTM(inSize, hiSize, bidirectional=True)
        self.bn1 = Seqbn(hiSize * 2)
        self.lstm2 = nn.LSTM(inSize * 2, hiSize, bidirectional=True)
        self.bn2 = Seqbn(hiSize * 2)
        self.lstm3 = nn.LSTM(inSize * 2, hiSize, bidirectional=True)
        self.bn3 = Seqbn(hiSize * 2)
        self.lstm4 = nn.LSTM(inSize * 2, hiSize, bidirectional=True)
        self.bn4 = Seqbn(hiSize * 2)
        self.lstm5 = nn.LSTM(inSize * 2, hiSize, bidirectional=True)
        self.bn5 = Seqbn(hiSize * 2)
        self.lstm6 = nn.LSTM(inSize * 2, hiSize, bidirectional=True)
        self.bn6 = Seqbn(hiSize * 2)
        
    def forward(self, pack, seqSize):
        out = pack_padded_sequence(pack, seqSize)
        out, h = self.lstm1(out)
        out, _ = pad_packed_sequence(out)
        out = self.bn1(out)
        
        out = pack_padded_sequence(out, seqSize)
        out, h = self.lstm2(out)
        out, _ = pad_packed_sequence(out)
        out = self.bn2(out)
        
        out = pack_padded_sequence(out, seqSize)
        out, h = self.lstm3(out)
        out, _ = pad_packed_sequence(out)
        out = self.bn3(out)

        out = pack_padded_sequence(out, seqSize)
        out, h = self.lstm4(out)
        out, _ = pad_packed_sequence(out)
        out = self.bn4(out)

        out = pack_padded_sequence(out, seqSize)
        out, h = self.lstm5(out)
        out, _ = pad_packed_sequence(out)
        out = self.bn5(out)

        out = pack_padded_sequence(out, seqSize)
        out, h = self.lstm6(out)
        out, _ = pad_packed_sequence(out)
        out = self.bn6(out)
        
        return out

# In[9]:


class CNN(nn.Module): # expected (batch, channel=40, timelength)
    def __init__(self, inChannel, outChannel):
        super(CNN, self).__init__()
        self.bn1 = nn.BatchNorm1d(inChannel)

        self.conv1 = nn.Conv1d(inChannel, outChannel, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(outChannel)
        self.tanh1 = nn.Hardtanh()
        
        self.conv2 = nn.Conv1d(outChannel, outChannel, 3, padding=1)
        self.bn3 = nn.BatchNorm1d(outChannel)
        self.tanh2 = nn.Hardtanh()

#        self.conv3 = nn.Conv1d(inChannel, outChannel, 3, padding=1)
#        self.bn4 = nn.BatchNorm1d(outChannel)
#        self.tanh3 = nn.Hardtanh()
        
    def forward(self, batch):
        out = self.bn1(batch)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.tanh1(out)
        
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.tanh2(out)
        
#        out = self.conv3(out)
#        out = self.bn4(out)
#        out = self.tanh3(out)
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
        self.ReLU = nn.ReLU()
        self.BiLSTM = BiLSTM(hiSize, hiSize)
        self.Dense = Dense(hiSize * 2, outSize)
        
        self.RED = nn.Linear(inSize, hiSize) #这里对输入的x进行了升维

    def forward(self, x, seqSize):
        # iniIdentity = self.RED(x)
        out = x.permute(1, 2, 0)
        out = self.CNN(out)
        out = out.permute(2, 0, 1)
        # cnnIdentity = out
        # out1 = out.clone() + iniIdentity
        out = self.ReLU(out)
        out = self.BiLSTM(out, seqSize)
        # out2 = out1.clone() + cnnIdentity
        out = self.Dense(out)
        
        return out


# ## 3. Start Training

# In[79]:


class Network():
    def __init__(self, path = None, epoch = EPOCH):
        
        self.cuda = torch.cuda.is_available()
        print(self.cuda)
        self.dev  = torch.device("cuda") if self.cuda else torch.device("cpu")
        print(self.dev)
        self.path = path

        self.trainLoader = trainLoader
        self.valLoader = devLoader
        self.testLoader = testLoader

        self.model = MyCLDNN(40, HIDDEN).to(self.dev)
        
        print('in cuda', next(self.model.parameters()).is_cuda)
        if self.path:
            print('... loaded! ...')
            self.model.load_state_dict(torch.load(self.path))
  
        self.epoch = epoch
        self.optimizer = optim.Adam(self.model.parameters(), amsgrad=True)#, weight_decay=1.2e-6)
        # self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
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
        output = torch.transpose(output, 0, 1)  # batch, seq_len, probs
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
                torch.nn.utils.clip_grad_norm(self.model.parameters(), 0.25)
                self.optimizer.step()
                tempLossList.append(loss.tolist())
                # tempErrorList.append(self.error(output, batchy))
                if self.cuda:
                    torch.cuda.empty_cache()
                    del batchx
                    del batchy
                    del loss
                    del output
                    del seqSize
                    del labelSize
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

            for l, m in zip(decodedy, decoded):
                e = self.error(l, m)
                totalError += e
                # tempErrorList.append(e * 100.0 / len(l))
            
            tempLossList.append(loss.tolist())
            if self.cuda:
                torch.cuda.empty_cache()
                del batchx
                del batchy
                del loss
                del output
                del seqSize
                del labelSize
        self.valLoss.append(np.mean(tempLossList))
        self.valError.append(totalError * 100 / valSet.phonemes) 

    def predict(self, path):
        with torch.no_grad():
            self.model.eval()
            modelIdx = self.valError.index(min(self.valError))
            modelpath = path + '_epoch_{}.pth'.format(modelIdx)
            print('the best model is ' + modelpath)
            self.model.load_state_dict(torch.load(path))
            result = []
            for batchIdx, (batchx, seqSize, _, _) in enumerate(self.testLoader):
                batchx = Variable(batchx.to(self.dev))
                output  = self.model(batchx.float(), seqSize) 
                decoded = self.decode(output, seqSize, beamWidth=100)
                result.extend(decoded)
                if self.cuda:
                    torch.cuda.empty_cache()
                    del batchx
                    del output
                    del seqSize
            self.output = np.array(result)
            
    
    def saveModel(self, path):
        # path = 'mytraining'
        torch.save(self.model.state_dict(), path)


# In[77]:


def main():
    
    model = Network('../param/M3_epoch_23.pth')
    model.train('../param/M3')

    model.predict('../param/M4')
    
    output = model.output
    
    return output


# In[78]:


a = main()


# In[47]:


b = zip([i for i in range(len(a))], a)
c = np.array(tuple(b))
df = pd.DataFrame(c, columns=['id', 'Predicted'])
df.to_csv(OUTPATH, index = False)

