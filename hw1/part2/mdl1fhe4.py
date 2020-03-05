#!/usr/bin/env python
# coding: utf-8

# In[40]:


# x 1100, 1 -> 625 * 40 # 总共有1100个语音，每个语音的时间都不同（行是T），每个语音被拆分成40个（列是40）
# y 总共有1100个语音，每个语音都用自己的T个label(0-137)


# # 正式开始

# In[235]:


import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import pandas as pd
import sys

TRAINX = sys.argv[1]
TRAINY = sys.argv[2]
TESTX = sys.argv[3]
VALX = sys.argv[4]
VALY = sys.argv[5]

MODELID = int(sys.argv[6])
DATAID = int(sys.argv[7])
HIDDEN = int(sys.argv[8])
CONTEXTN = int(sys.argv[9])
BATCHN = int(sys.argv[10])
EPOCH = int(sys.argv[11])

PARAMPATH = sys.argv[12]
OUTPATH = sys.argv[13]
MODELPATH = sys.argv[14]

# In[205]:

trainx =  np.load(TRAINX , allow_pickle=True)
trainy =  np.load(TRAINY, allow_pickle=True)
testx =  np.load(TESTX, allow_pickle=True)
testy = []
for i in range(len(testx)):
    testy.append([0 for i in range(testx[i].shape[0])])
valx = np.load(VALX, allow_pickle=True) 
valy = np.load(VALY, allow_pickle=True)

def getData(x, y, k, dataIdx):
    if dataIdx == 1:
        class NLPDataset1(Dataset):
            def __init__(self, x, y, k):
                super().__init__()
                assert len(x) == len(y)
                
                self._x = np.pad(x[0], ((k,k),(0,0)), mode='constant')
                pointer = k
                self.index = [pointer + i for i in range(x[0].shape[0])]
                pointer += x[0].shape[0]
                
                for item in x[1:]:
                    pointer += 2*k
                    self.index.extend([pointer + i for i in range(item.shape[0])])
                    temp = np.pad(item, ((k,k),(0,0)), mode='constant')
                    self._x = np.concatenate((self._x, temp), axis=0)
                    pointer += item.shape[0]

                self._y = []
                for i in y:
                    for j in i:
                        self._y.append(j)
                self.k = k
                self._y = np.array(self._y)
            def __len__(self):
                return len(self._y)
              
            def __getitem__(self, idx):
                realId = self.index[idx]
                x_item = self._x.take(range(realId - self.k, realId + self.k + 1), mode='clip', axis=0).flatten()
                y_item = self._y[idx]
                return x_item, y_item
        dataset = NLPDataset1(x, y, k)
        
    if dataIdx == 2:
        class NLPDataset2(Dataset):
            def __init__(self, x, y, k):
                self.x = np.pad(np.vstack(x), ((k,k),(0,0)), mode='constant')
                self.k = k
                self.y = torch.from_numpy(np.vstack(np.array([np.expand_dims(i,axis=1) for i in y]))).reshape(self.x.shape[0] - 2 * self.k,)
            def __len__(self):
                return len(self.y)
              
            def __getitem__(self, idx):
                dataIdx = idx + self.k
                return self.x[dataIdx - self.k: dataIdx + self.k + 1, :].flatten(),self.y[idx]
        dataset = NLPDataset2(x, y, k)  
        
    if dataIdx == 3:
        class NLPDataset3(Dataset):
            def __init__(self, x, y, k):
                self.x = x
                self.k = k
                self.y = np.concatenate([i for i in y], axis = 0)
                realId = 0
                self.id2dict = {}
                for utterId in range(x.shape[0]):
                    for frameId in range(x[utterId].shape[0]):
                        self.id2dict[realId] = (utterId, frameId, x[utterId].shape[0])
                        realId += 1
            def __len__(self):
                return len(self.y)
              
            def __getitem__(self, idx):
                utterId, frameId, framen = self.id2dict[idx]
                frontpad = (frameId - self.k) < 0 
                endpad = (frameId + self.k + 1) > framen
                if (frontpad and endpad):
                    xitem = np.pad(self.x[utterId][:, :], ((self.k - frameId, frameId + self.k + 1 - framen),(0,0)), mode='constant').flatten()
                elif (frontpad and not endpad):
                    xitem = np.pad(self.x[utterId][:frameId+k+1, :], ((self.k - frameId, 0),(0,0)), mode='constant').flatten()
                elif (not frontpad and endpad):
                    xitem = np.pad(self.x[utterId][frameId-k:, :], ((0, frameId + self.k + 1 - framen),(0,0)), mode='constant').flatten()
                else:
                    xitem = self.x[utterId][frameId-k:frameId+k+1, :].flatten()
                return xitem, self.y[idx]
        dataset = NLPDataset3(x, y, k)                      
    return dataset


# In[177]:
def weightsInit(m):
    if type(m) in [nn.Linear]:
        nn.init.xavier_uniform_(m.weight)

def getModel(inSize, hiddenN, outSize, modelIdx):
    if modelIdx == 1:
        model = nn.Sequential(
                nn.Linear(inSize, hiddenN),
                nn.BatchNorm1d(hiddenN),
                nn.ReLU(),

                nn.Linear(hiddenN, hiddenN),
                nn.BatchNorm1d(hiddenN),
                nn.ReLU(),

                nn.Linear(hiddenN, hiddenN),
                nn.BatchNorm1d(hiddenN),
                nn.ReLU(),

                nn.Linear(hiddenN, hiddenN // 2),
                nn.BatchNorm1d(hiddenN // 2),
                nn.ReLU(),

                nn.Linear(hiddenN // 2, hiddenN // 2),
                nn.BatchNorm1d(hiddenN // 2),
                nn.ReLU(),

                nn.Linear(hiddenN // 2, hiddenN // 4),
                nn.BatchNorm1d(hiddenN // 4),
                nn.ReLU(),

                nn.Linear(hiddenN // 4, hiddenN // 4),
                nn.BatchNorm1d(hiddenN // 4),
                nn.ReLU(),

                nn.Linear(hiddenN // 4, outSize))
        
    if modelIdx == 2:
        model = nn.Sequential(
                nn.Linear(inSize, hiddenN),
                nn.BatchNorm1d(hiddenN),
                nn.LeakyReLU(),
		
                nn.Linear(hiddenN, hiddenN // 2),
                nn.LeakyReLU(),

                nn.Linear(hiddenN // 2, outSize))
        
    if modelIdx == 3:
        model = nn.Sequential(
                nn.Linear(inSize, 2048),
                nn.BatchNorm1d(2048),
                nn.RReLU(),

                nn.Linear(2048, 1024),
                nn.BatchNorm1d(1024),
                nn.RReLU(),

                nn.Linear(1024, 1024),
                nn.BatchNorm1d(1024),
                nn.RReLU(),
                
                nn.Linear(1024, 1024),
                nn.BatchNorm1d(1024),
                nn.RReLU(),

                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                nn.RReLU(),

                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.RReLU(),

                nn.Linear(256, outSize))
    return model


# In[210]:


class Network():
    def __init__(self, trainxArr, trainyArr, testxArr, testyArr, valxArr, valyArr, modelIdx, 
                 hiddenN, dataId = 1, k=12, gpu = 0, path = None, batchN = 32, epoch = 1000):
        
        self.cuda = torch.cuda.is_available()
        print(self.cuda)
        self.dev  = torch.device("cuda") if self.cuda else torch.device("cpu")
        print(self.dev)
        self.loaderArgs = {'num_workers': 4, 'pin_memory': True} if self.cuda else {}
        self.path = path
        self.trainx, self.inSize  = trainxArr, 40 * (2 * k + 1)
        self.trainy, self.outSize = trainyArr, 138
        self.valx = valxArr
        self.valy = valyArr
        self.testx = testxArr
        self.testy = testyArr
        
        self.trainset = getData(self.trainx, self.trainy, k, dataId)
        self.valset = getData(self.valx, self.valy, k, dataId)
        print(self.valset.y)
        self.testset = getData(self.testx, self.testy, k, dataId)

        self.trainLoader = DataLoader(self.trainset, batch_size=batchN, shuffle=True, num_workers=4)
        self.valLoader = DataLoader(self.valset, batch_size=batchN, shuffle=False, num_workers=4)
        self.testLoader = DataLoader(self.testset, batch_size=batchN, shuffle=False, num_workers=4)
        # if MODELPATH == 'false':
        self.model = getModel(self.inSize, hiddenN, self.outSize, modelIdx).to(self.dev)
	
        print('in cuda', next(self.model.parameters()).is_cuda)
        if self.path:
            self.model.load_state_dict(torch.load(self.path))
  
        self.epoch = epoch
        self.optimizer = optim.Adam(self.model.parameters(), amsgrad=True)
        self.scheduler = ReduceLROnPlateau(self.optimizer)
        #self.scheduler = ReduceLROnPlateau(self.optimizer, 'min')

        self.lossList  = []
        self.errorList = []

        self.valLoss  = []
        self.valError = []
    
    def error(self, output, labels):
        return (np.argmax(output.cpu().detach().numpy(), axis = 1) != labels.cpu().detach().numpy()).sum()

    def train(self, path):
        for epochIdx in range(self.epoch):
            self.model.train()
            if epochIdx == 9:
                self.scheduler = StepLR(self.optimizer, step_size=1, gamma=0.05)
            tempLossList  = []
            tempErrorList = []
            for batchIdx, (batchx, batchy) in enumerate(self.trainLoader):
                # print('shape', batchx.shape, batchy)
                if self.cuda:
                    batchx, batchy = Variable(batchx.to(self.dev)), Variable(batchy.to(self.dev))
                self.optimizer.zero_grad()
                output  = self.model(batchx.float())
                loss = F.cross_entropy(output, batchy)
                loss.backward(retain_graph=True)
                self.optimizer.step()
                tempLossList.append(loss.tolist())
                tempErrorList.append(self.error(output, batchy))
                if batchIdx % 200000 == 0:
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
        for batchIdx, (batchx, batchy) in enumerate(self.valLoader):
            if self.cuda:
                batchx, batchy = Variable(batchx.to(self.dev)), Variable(batchy.to(self.dev))
            output  = self.model(batchx.float())
            loss = F.cross_entropy(output, batchy)
            tempLossList.append(loss.tolist())
            tempErrorList.append(self.error(output, batchy))
        self.valLoss.append(np.mean(tempLossList))
        self.valError.append(1 - np.sum(tempErrorList) / len(self.valset.y))

    def predict(self, path):
        with torch.no_grad():
            self.model.eval()
            # print('The validation error is ', self.valError)
            #modelIdx = self.valError.index(min(self.valError))
            #print('The validation error index is ', modelIdx) 
            #modelpath = path+'_epoch_{}.pth'.format(modelIdx)
            #print('The best model is {} and the best model is '.format(modelIdx) + modelpath)
            self.model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
            result = []
            for batchIdx, (batchx, _) in enumerate(self.testLoader):
                batchx = Variable(batchx.to(self.dev))
                output  = self.model(batchx.float()) 
                result.extend(np.argmax(output.detach().numpy(), axis = 1).tolist())
            self.output = np.array(result)
    
    def saveModel(self, path):
        # path = 'mytraining'
        torch.save(self.model.state_dict(), path)


# In[214]:

def main():
    
    model = Network(trainx, trainy, testx, testy,
                    valx, valy, MODELID, HIDDEN, dataId=DATAID, k=CONTEXTN, batchN = BATCHN, epoch = EPOCH)
    if MODELPATH != 'false':
        model.predict(MODELPATH)
        output = model.output
    else:
        model.train(PARAMPATH)

        model.predict(PARAMPATH)
    
        output = model.output
    
    return output

# In[ ]:

a = main()

# In[227]:

b = zip([i for i in range(len(a))], a)
c = np.array(tuple(b))
df = pd.DataFrame(c, columns=['id', 'label'])
df.to_csv(OUTPATH, index = False)

