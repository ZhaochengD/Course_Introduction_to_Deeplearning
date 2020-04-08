import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F

def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=False):
        super(BasicBlock, self).__init__()
        self.isDownSample = downsample
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        
        if self.isDownSample:
            self.downconv = conv1x1x1(inplanes, planes, stride)
            self.downnorm = nn.BatchNorm3d(planes)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.isDownSample:
            residual = self.downconv(residual)
            residual = self.downnorm(residual)

        out += residual
        out = self.relu(out)

        return out

class resnet10(nn.Module):
    def __init__(self, num_classes, depth, width, length):
        super(resnet10, self).__init__()
        self.depth = depth
        self.width = width
        self.length = length
        self.conv1 = nn.Conv3d(3, 8, kernel_size=7,stride=(1, 2, 2),padding=(3, 3, 3),bias=False)
        self.bn1 = nn.BatchNorm3d(8)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        
        self.block2 = BasicBlock(8, 16, stride=2, downsample=True)
        self.block3 = BasicBlock(16, 32, stride=2, downsample=True)
        self.block4 = BasicBlock(32, 64, stride=2, downsample=True)
        self.block5 = BasicBlock(64, 128, stride=2, downsample=True)
        last_duration = int(math.ceil(self.depth / 32))
        last_width = int(math.ceil(self.width / 64))
        last_length = int(math.ceil(self.length / 64))
        self.avgpool = nn.AvgPool3d((last_duration, last_width, last_length), stride=1)
        
        self.fc = nn.Linear(128, num_classes)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.maxpool1(out)
        
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = self.avgpool(out)
        out = torch.squeeze(out)
        out = self.fc(out)
        
        return out
    
    