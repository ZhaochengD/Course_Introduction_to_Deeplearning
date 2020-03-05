#!/usr/bin/env python
# coding: utf-8

# In[91]:


import torch
import numpy as np
from numpy import linalg as LA
from PIL import Image
import glob
import cv2
from sklearn.metrics import roc_auc_score
import os
import pandas as pd
from torch.autograd import Variable
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch.optim as optim


# https://github.com/pytorch/vision/issues/157 how to load the data

# ### 1. This is the dataLoader part

# In[2]:


trainPath = '../data/train_data/large'

testClaPath = '../data/test_classification/medium/'
testVerPath = '../data/test_verification/'
testClaOrder = '../data/test_order_classification.txt'
testVerOrder = '../data/test_trials_verification_student.txt'

validClaPath = '../data/validation_classification/large/'
validVerPath = '../data/validation_verification/'
validVerOrder = '../data/validation_trials_verification.txt'


# In[217]:


OUTPATH = '../result/out.csv'


# In[226]:


def loadClaTest(testPath=testClaPath, testOrder=testClaOrder):
    orderLi = [testPath + i.strip() for i in open(testClaOrder)]
    return orderLi


# In[4]:


def loadVerTest(testPath=testVerPath, testOrder=testVerOrder):
    '''return the first image and second image'''
    orderLi = [i.strip().split(' ') for i in open(testOrder)]
    orderLi = [[testPath + i[0], testPath + i[1]] for i in orderLi]
    return orderLi


# In[5]:


def loadVerValid(testPath=validVerPath, testOrder=validVerOrder):
    '''return the first image , second image and related label'''
    orderLi = [i.strip().split(' ') for i in open(testOrder)]
    orderLi = [[testPath + i[0], testPath + i[1], int(i[2])] for i in orderLi]
    return orderLi


# In[227]:


def getDataLoader(mode):
    
    if mode == 'ctrain':
        dataTransform = transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(),
            #transforms.RandomRotation(10),
            #transforms.RandomAffine(5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        dataset = ImageFolder(trainPath, transform=dataTransform)
        dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=4)
        idx2class = {v: k for k, v in dataset.class_to_idx.items()}
        
    if mode == 'cvalid':
        dataTransform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        dataset = ImageFolder(validClaPath, transform=dataTransform)
        dataloader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=4)
        
    if mode == 'ctest':
        x = loadClaTest()
        class CVDatasetTE(Dataset):
            def __init__(self, x):
                self.x = x
                self.dataTransform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                
            def __len__(self):
                return len(self.x)
              
            def __getitem__(self, idx):
                return self.dataTransform(Image.open(self.x[idx]))
        dataset = CVDatasetTE(x)
        dataloader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=4)
    
    if mode == 'vvalid':
        im = loadVerValid()
        class CVDatasetTE(Dataset):
            def __init__(self, imli):
                self.imli = imli
                self.dataTransform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                
            def __len__(self):
                return len(self.imli)
              
            def __getitem__(self, idx):
                im1 = self.dataTransform(cv2.imread(self.imli[idx][0]))
                im2 = self.dataTransform(cv2.imread(self.imli[idx][1]))
                label = self.imli[idx][2]
                return im1, im2, label
            
        dataset = CVDatasetTE(im)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    
    if mode == 'vtest':
        im = loadVerTest()
        class CVDatasetTE(Dataset):
            def __init__(self, imli):
                self.imli = imli
                self.dataTransform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                
            def __len__(self):
                return len(self.imli)
              
            def __getitem__(self, idx):
                im1 = self.dataTransform(cv2.imread(self.imli[idx][0]))
                im2 = self.dataTransform(cv2.imread(self.imli[idx][1]))
                return im1, im2
            
        dataset = CVDatasetTE(im)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    if mode == 'ctrain':
        return dataloader, idx2class
    else:
        return dataloader


# ### 2. This is the model construction part

# In[7]:


def conv7x7(inChannels, outChannels=64, kernelSize=7, stride=2, dilation=1):
    return nn.Conv2d(inChannels, outChannels, kernelSize, stride=stride, 
                  padding=dilation, dilation=dilation, bias=False)


# In[8]:


def conv3x3(inChannels, outChannels, stride=1, dilation=1):
    return nn.Conv2d(inChannels, outChannels, kernel_size=3, stride=stride, 
                  padding=dilation, bias=False, dilation=dilation)


# In[9]:


def conv1x1(inChannels, outChannels, stride=1):
    return nn.Conv2d(inChannels, outChannels, kernel_size=1, stride=stride, bias=False)


# In[105]:


def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight.data)


# In[201]:


class BasicBlock(nn.Module):
    def __init__(self, inChannel, outChannel, stride = 1, isDownSample=False):
        super(BasicBlock, self).__init__()
        self.isDownSample = isDownSample
        
        self.conv1 = conv3x3(inChannel, outChannel, stride)
        self.norm1 = nn.BatchNorm2d(outChannel)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = conv3x3(outChannel, outChannel)
        self.norm2 = nn.BatchNorm2d(outChannel)
        
        if isDownSample:
            self.downconv = conv1x1(inChannel, outChannel, stride)
            self.downnorm = nn.BatchNorm2d(outChannel)
    
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu1(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        
        if self.isDownSample:
            identity = self.downconv(identity)
            identity = self.downnorm(identity)
        out += identity
        out = self.relu1(out)
        
        return out


# In[202]:


class Bottleneck(nn.Module):
    def __init__(self, inChannel, outChannel, stride = 1, isDownSample=False):
        super(Bottleneck, self).__init__()
        self.expansion = 4
        self.isDownSample = isDownSample
        self.conv1 = conv1x1(inChannel, outChannel)
        self.norm1 = nn.BatchNorm2d(outChannel)

        self.conv2 = conv3x3(outChannel, outChannel, stride)
        self.norm2 = nn.BatchNorm2d(outChannel)
        
        self.conv3 = conv1x1(outChannel, outChannel * self.expansion)
        self.norm3 = nn.BatchNorm2d(outChannel * self.expansion)
        self.relu1 = nn.ReLU(inplace=True)
        
        if isDownSample:
            self.downconv = conv1x1(inChannel, outChannel * self.expansion, stride)
            self.downnorm = nn.BatchNorm2d(outChannel * self.expansion)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu1(out)

        out = self.conv3(out)
        out = self.norm3(out)

        if self.isDownSample:
            identity = self.downconv(identity)
            identity = self.downnorm(identity)
        out += identity
        out = self.relu1(out)

        return out


# In[203]:


class MyResnet50(nn.Module):
    def __init__(self, inChannel, outClass, outFeat):
        super(MyResnet50, self).__init__()
        self.conv0 = conv3x3(inChannel, 64 * 4, stride = 1)
        
        self.block2 = Bottleneck(64 * 4, 64)
        self.block3 = Bottleneck(64 * 4, 64)
        self.block4 = Bottleneck(64 * 4, 64)
        
        self.block5 = Bottleneck(64 * 4, 128, stride = 2, isDownSample=True)
        self.block6 = Bottleneck(128 * 4, 128)
        self.block7 = Bottleneck(128 * 4, 128)
        self.block8 = Bottleneck(128 * 4, 128)
        
        self.block9 = Bottleneck(128 * 4, 256, stride = 2, isDownSample=True)
        self.block10 = Bottleneck(256 * 4, 256)
        self.block11 = Bottleneck(256 * 4, 256)
        self.block12 = Bottleneck(256 * 4, 256)
        self.block13 = Bottleneck(256 * 4, 256)
        self.block14 = Bottleneck(256 * 4, 256)
        
        self.block15 = Bottleneck(256 * 4, 512, stride = 2, isDownSample=True)
        self.block16 = Bottleneck(512 * 4, 512)
        self.block17 = Bottleneck(512 * 4, 512)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, outClass)
        
        self.cfc = nn.Linear(512 * 4, outFeat)
        self.crelu = nn.ReLU(inplace=True)

    def forward(self, x, ver=False):
        out = x
        out = self.conv0(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = self.block6(out)
        out = self.block7(out)
        out = self.block8(out)
        out = self.block9(out)
        out = self.block10(out)
        out = self.block11(out)
        out = self.block12(out)
        out = self.block13(out)
        out = self.block14(out)
        out = self.block15(out)
        out = self.block16(out)
        out = self.block17(out)

        out = self.avgpool(out)
        
        out = torch.squeeze(out)

        rout = self.fc(out)
        cout = self.cfc(out)
        # cout = self.crelu(cout)
        
        return rout, cout
        
class MyResnet101(nn.Module):
    def __init__(self, inChannel, outClass, outFeat):
        super(MyResnet101, self).__init__()
        self.conv0 = conv3x3(inChannel, 64 * 4, stride = 1)
        
        self.block2 = Bottleneck(64 * 4, 64)
        self.block3 = Bottleneck(64 * 4, 64)
        self.block4 = Bottleneck(64 * 4, 64)
        
        self.block5 = Bottleneck(64 * 4, 128, stride = 2, isDownSample=True)
        self.block6 = Bottleneck(128 * 4, 128)
        self.block7 = Bottleneck(128 * 4, 128)
        self.block8 = Bottleneck(128 * 4, 128)
        
        self.block9 = Bottleneck(128 * 4, 256, stride = 2, isDownSample=True)
        self.block10 = Bottleneck(256 * 4, 256)
        self.block11 = Bottleneck(256 * 4, 256)
        self.block12 = Bottleneck(256 * 4, 256)
        self.block13 = Bottleneck(256 * 4, 256)
        self.block14 = Bottleneck(256 * 4, 256)
        self.block15 = Bottleneck(256 * 4, 256)
        self.block16 = Bottleneck(256 * 4, 256)
        self.block17 = Bottleneck(256 * 4, 256)
        self.block18 = Bottleneck(256 * 4, 256)
        self.block19 = Bottleneck(256 * 4, 256)
        self.block20 = Bottleneck(256 * 4, 256)
        self.block21 = Bottleneck(256 * 4, 256)
        self.block22 = Bottleneck(256 * 4, 256)
        self.block23 = Bottleneck(256 * 4, 256)
        self.block24 = Bottleneck(256 * 4, 256)
        self.block25 = Bottleneck(256 * 4, 256)
        self.block26 = Bottleneck(256 * 4, 256)
        self.block27 = Bottleneck(256 * 4, 256)
        self.block28 = Bottleneck(256 * 4, 256)
        self.block29 = Bottleneck(256 * 4, 256)
        self.block30 = Bottleneck(256 * 4, 256)
        self.block31 = Bottleneck(256 * 4, 256)
        
        self.block32 = Bottleneck(256 * 4, 512, stride = 2, isDownSample=True)
        self.block33 = Bottleneck(512 * 4, 512)
        self.block34 = Bottleneck(512 * 4, 512)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, outClass)
        
        self.cfc = nn.Linear(512 * 4, outFeat)
        self.crelu = nn.ReLU(inplace=True)

    def forward(self, x, ver=False):
        out = x
        out = self.conv0(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = self.block6(out)
        out = self.block7(out)
        out = self.block8(out)
        out = self.block9(out)
        out = self.block10(out)
        out = self.block11(out)
        out = self.block12(out)
        out = self.block13(out)
        out = self.block14(out)
        out = self.block15(out)
        out = self.block16(out)
        out = self.block17(out)
        out = self.block18(out)
        out = self.block19(out)
        out = self.block20(out)
        out = self.block21(out)
        out = self.block22(out)
        out = self.block23(out)
        out = self.block24(out)
        out = self.block25(out)
        out = self.block26(out)
        out = self.block27(out)
        out = self.block28(out)
        out = self.block29(out)
        out = self.block30(out)
        out = self.block31(out)
        out = self.block32(out)
        out = self.block33(out)
        out = self.block34(out)

        out = self.avgpool(out)
        
        out = torch.squeeze(out)

        rout = self.fc(out)
        cout = self.cfc(out)
        # cout = self.crelu(cout)
        
        return rout, cout


# In[204]:


class MyResnet18(nn.Module):
    def __init__(self, inChannel, outClass, outFeat):
        super(MyResnet18, self).__init__()
        self.conv0 = conv3x3(inChannel, 64, stride = 1)
        
        self.block2 = BasicBlock(64, 64, stride = 1, isDownSample=False)
        self.block3 = BasicBlock(64, 64, stride = 1, isDownSample=False)
        
        self.block4 = BasicBlock(64, 128, stride = 2, isDownSample=True)
        self.block5 = BasicBlock(128, 128, stride = 1, isDownSample=False)
        
        self.block6 = BasicBlock(128, 256, stride = 2, isDownSample=True)
        self.block7 = BasicBlock(256, 256, stride = 1, isDownSample=False)
        
        self.block8 = BasicBlock(256, 512, stride = 2, isDownSample=True)
        self.block9 = BasicBlock(512, 512, stride = 1, isDownSample=False)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, outClass)
        
        self.cfc = nn.Linear(512, outFeat)
        self.crelu = nn.ReLU(inplace=True)

    def forward(self, x, ver=False):
        out = x
        out = self.conv0(out)
        
        out = self.block2(out)
        out = self.block3(out)

        out = self.block4(out)
        out = self.block5(out)

        out = self.block6(out)
        out = self.block7(out)

        out = self.block8(out)
        out = self.block9(out)

        out = self.avgpool(out)
        
        out = torch.squeeze(out)

        rout = self.fc(out)
        cout = self.cfc(out)
        # cout = self.crelu(cout)
        
        return rout, cout


# ### 3. The training part

# In[205]:


def get_auc(y_true, y_score):
    auc = roc_auc_score(y_true, y_score)
    return auc


# In[206]:


class CenterLoss(nn.Module):
    """
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes, feat_dim, device=torch.device('cpu')):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device
        
        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).to(self.device))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long().to(self.device)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = []
        for i in range(batch_size):
            value = distmat[i][mask[i]]
            value = value.clamp(min=1e-12, max=1e+12) # for numerical stability
            dist.append(value)
        dist = torch.cat(dist)
        loss = dist.mean()

        return loss


# In[230]:


class Network():
    def __init__(self, path = None, epoch = 30):
        
        self.cuda = torch.cuda.is_available()
        print(self.cuda)
        self.dev  = torch.device("cuda") if self.cuda else torch.device("cpu")
        print(self.dev)
        
        self.trainLoader, self.idx2class = getDataLoader('ctrain')
        self.valLoader = getDataLoader('cvalid')
        self.testLoader = getDataLoader('ctest')
        
        self.valLoader2 = getDataLoader('vvalid')
        self.testLoader2 = getDataLoader('vtest')
        
        self.testname = [i.strip() for i in open(testClaOrder)]
        
        self.featnum = 100
        self.model = MyResnet101(3, 4300, self.featnum).to(self.dev)#.apply(init_weights)
        
        #if TRAINEDMODEL != 'false':
            #self.path = TRAINEDMODEL

        print('in cuda', next(self.model.parameters()).is_cuda)
        
        if path:
            self.model.load_state_dict(torch.load(path))
        self.clossWeight = 1
        self.closs = CenterLoss(len(self.idx2class), self.featnum, self.dev)
        self.epoch = epoch
        self.roptimizer = optim.Adam(self.model.parameters(), amsgrad=True)
        self.coptimizer = optim.Adam(self.closs.parameters(), amsgrad=True)
        self.rscheduler = ReduceLROnPlateau(self.roptimizer)
        self.cscheduler = ReduceLROnPlateau(self.coptimizer)
        #self.scheduler = ReduceLROnPlateau(self.optimizer, 'min')

        self.lossList  = []
        self.errorList = []

        self.valLoss  = []
        self.valError = []
    
    def error(self, output, labels):
        return (np.argmax(output.cpu().detach().numpy(), axis = 1) != labels.cpu().detach().numpy()).sum()
    
    def cosSimilarity(self, v1, v2):
        return v1.dot(v2) / (LA.norm(v1) * LA.norm(v2))
    
    def train(self, path):
        for epochIdx in range(self.epoch):
            self.model.train()
            tempLossList  = []
            tempErrorList = []
            for batchIdx, (batchx, batchy) in enumerate(self.trainLoader):
                if self.cuda:
                    batchx, batchy = Variable(batchx.to(self.dev)), Variable(batchy.to(self.dev))
                self.roptimizer.zero_grad()
                self.coptimizer.zero_grad()
                
                output, feature  = self.model(batchx.float())
                rloss  = F.cross_entropy(output, batchy)
                closs = self.closs(feature, batchy)
                loss = rloss + self.clossWeight * closs
                
                loss.backward(retain_graph=True)
                self.roptimizer.step()
                for param in self.closs.parameters():
                    param.grad.data *= (1. / self.clossWeight)
                    self.coptimizer.step()
                tempLossList.append(loss.tolist())
                tempErrorList.append(self.error(output, batchy))
                if self.cuda:
                    torch.cuda.empty_cache() 
                    del batchx
                    del batchy
                    del loss
                    del rloss
                    del closs
                    del output
                    del feature
            self.lossList.append(np.mean(tempLossList))
            self.errorList.append(np.sum(tempErrorList))
            self.validate()
            self.model.train()
            self.rscheduler.step(self.valLoss[-1])
            self.cscheduler.step(self.valLoss[-1])
            print('validation', self.valError)
            self.saveModel(path+'_epoch_{}.pth'.format(epochIdx))
            print('--------------------'+str(epochIdx)+'-----'+str(np.mean(tempLossList)))

    def validate(self):
        self.model.eval()
        tempLossList  = []
        tempErrorList = []
        totalValid = 0
        for batchIdx, (batchx, batchy) in enumerate(self.valLoader):
            if self.cuda:
                batchx, batchy = Variable(batchx.to(self.dev)), Variable(batchy.to(self.dev))
                
            output, feature  = self.model(batchx.float())
            rloss  = F.cross_entropy(output, batchy)
            closs = self.closs(feature, batchy)
            loss = rloss + self.clossWeight * closs
        
            tempLossList.append(loss.tolist())
            tempErrorList.append(self.error(output, batchy))
            totalValid += len(batchy)
            if self.cuda:
                torch.cuda.empty_cache() 
                del batchx
                del batchy
                del loss
                del rloss
                del closs
                del output
                del feature

        self.valLoss.append(np.mean(tempLossList))
        self.valError.append(1 - np.sum(tempErrorList) / totalValid)

    def predict(self, path):
        with torch.no_grad():
            self.model.eval()
            print('The validation accuracy is ', self.valError)
            modelIdx = self.valError.index(min(self.valError))
            print('The validation error index is ', modelIdx) 
            modelpath = path+'_epoch_{}.pth'.format(modelIdx)
            print('The best model is {} and the best model is '.format(modelIdx) + modelpath)
            self.model.load_state_dict(torch.load(modelpath))
            result = []
            for batchIdx, batchx in enumerate(self.testLoader):
                batchx = Variable(batchx.to(self.dev))
                output, _ = self.model(batchx.float())
                result.extend(np.argmax(output.cpu().detach().numpy(), axis = 1).tolist())
                if self.cuda:
                    torch.cuda.empty_cache() 
                    del batchx
            self.output = np.array([int(self.idx2class[i]) for i in result])
            
            b = zip(self.testname , self.output)
            c = np.array(tuple(b))
            df = pd.DataFrame(c, columns=['Id', 'Category'])
            df.to_csv(OUTPATH, index = False)
    
    def verificationValidate(self, path):
        with torch.no_grad():
            self.model.eval()
            modelIdx = self.valError.index(min(self.valError))
            modelpath = path+'_epoch_{}.pth'.format(modelIdx)
            self.model.load_state_dict(torch.load(modelpath))
            result, label = [], []
            for batchIdx, (batchx1, batchx2, batchy) in enumerate(self.valLoader2):
                batchx1, batchx2 = Variable(batchx1.to(self.dev)), Variable(batchx2.to(self.dev))
                _, output1 = self.model(batchx1.float()) 
                _, output2  = self.model(batchx2.float())
                result.append(self.cosSimilarity(output1.cpu().detach().numpy(), output2.cpu().detach().numpy()))
                label.append(batchy)
                if batchIdx == 0:
                    break
            self.vscore = np.array(result)
            self.vlabel = np.array(label)
            return self.vscore
            
    def verificationTest(self, path):
        with torch.no_grad():
            self.model.eval()
            modelIdx = self.valError.index(min(self.valError)) 
            modelpath = path+'_epoch_{}.pth'.format(modelIdx)
            self.model.load_state_dict(torch.load(modelpath))
            result = []
            for batchIdx, (batchx1, batchx2) in enumerate(self.testLoader2):
                batchx1, batchx2 = Variable(batchx1.to(self.dev)), Variable(batchx2.to(self.dev))
                _, output1  = self.model(batchx1.float()) 
                _, output2  = self.model(batchx2.float())
                result.append(self.cosSimilarity(output1.cpu().detach().numpy(), output2.cpu().detach().numpy()))
                if batchIdx == 0:
                    break
            self.tscore = np.array(result)
            return self.tscore
    
    def saveModel(self, path):
        # path = 'mytraining'
        torch.save(self.model.state_dict(), path)


# In[231]:


def main():
    
    model = Network()
    
    #if MODELPATH != 'false':
    #    model.predict(MODELPATH)
     #   output = model.output
    #else:
    model.train('../param/M7')
    model.predict('../param/M7')
    #model.verificationValidate('../param/M1')
    #model.verificationTest('../param/M1')
    
    output = model.output
    #output2 = model.vscore
    #output3 = model.tscore
    return output#, output2, output3
    # return output2, output3


# In[229]:


a = main()


# In[ ]:




