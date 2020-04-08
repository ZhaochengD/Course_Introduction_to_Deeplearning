#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# In[2]:


mu = 1 # see the formula 2
s = 1 # see the formula 2


# In[3]:


class Queue():
    def __init__(self, max_size, n_classes):
        self.queue = np.zeros((max_size, n_classes),dtype = float).tolist()
        self.max_size = max_size
        self.median = None
        self.average = None
        self.exponential = None
        
    def enqueue(self, data):
        self.queue.insert(0,data)
        self.median = self._median()
        self.average = self._average()
        self.exponential = self._exponential()
        return True

    def dequeue(self):
        if len(self.queue)>0:
            return self.queue.pop()
        return ("Queue Empty!")

    def size(self):
        return len(self.queue)

    def printQueue(self):
        return self.queue
     
    def _average(self):
        return np.array(self.queue[:self.max_size]).mean(axis = 0)

    def _median(self):
        return np.median(np.array(self.queue[:self.max_size]), axis = 0)
    
    def _exponential(sedxlf):
        weights = np.exp(np.linspace(-1., 0., self.max_size))
        weights /= weights.sum()
        average = weights.reshape(1,self.max_size).dot( np.array(self.queue[:self.max_size]))
        return average.reshape(average.shape[1],)


# In[11]:


class VideoQueue():
    def __init__(self, frame_W, frame_H, frame_C, clf_len, det_len, max_len):
        self.vq = [np.zeros((frame_C, frame_W, frame_H)) for i in range(clf_len - 1)]
        self.clf_len = clf_len
        self.det_len = det_len
        self.max_len = max_len
        self.index = clf_len - 1
        
    def enqueue(self, new_frame):
        self.vq.append(new_frame)
        self.len += 1
    
    def get_data(self):
        det_data = np.array(self.vq[self.index - self.det_len, self.index])
        clf_data = np.array(self.vq[self.index - self.clf_len, self.index])
        self.index += 1
        self.vq = self.vq[self.index - self.det_len, :]
        return det_data, clf_data


# In[4]:


def calculate_wj(j):
    '''
    calculate formula 1
    '''
    return (1 / (1 + np.exp(-0.2*(j-9))))


# In[5]:


def post_processing(outputs_det, queue_det, strategy):
    '''
    inputs:
    outputs_det: detector raw output sigmoid prob
    queue_det: the queue for detector now
    strategy: decide which strategy used
    
    return: 
    max1: 
    '''
    queue_det.enqueue(outputs_det.tolist())
    if strategy == 'mean':
        detector = queue_det.average
    elif strategy == 'median':
        detector = queue_det.median
    elif strategy == 'median':
        detector = queue_det.exponential
    if detector >= 0.5:
        return 1
    else:
        return 0


# In[6]:


def single_time_activate(outputs_clf, j, probs_old, probs_new):
    '''
    inputs:
    outputs_clf: the probability classifier give and pass sigmoid
    j: see the paper Algotithm1 for j
    probs_old: see the paper for probs_j-1
    probs_new: new sigmoid output from classifier
    
    return:
    max1: maximum gesture, if no, this quals -1
    new_j: j+=1
    '''
    ACTIVE = True
    alpha = probs_old * (j-1)
    mean_prods = (alpha + calculate_wj(j) * probs_new) / j
    max1, max2 = mean_prods.argsort()[-2:][::1]
    if mean_prods[max1] - mean_prods[max2] >= tao_early:
        EARLY_DETECTION = True
        return max1, j+1
    else:
        return -1, j+1


# In[10]:


def get_cam_frame(cap):
    '''
    read a frame from video
    '''
    return cap.read().transpose((2, 0, 1))

