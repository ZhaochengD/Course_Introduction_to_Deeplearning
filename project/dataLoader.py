#!/usr/bin/env python
# coding: utf-8

# ## 读的基本函数

# In[80]:


import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import cv2
import numpy as np
import pdb


# In[69]:



def load_split_nvgesture(file_with_split = '../data/nvgesture_train_correct_cvpr2016.lst',list_split = list()):
    params_dictionary = dict()
    with open(file_with_split,'rb') as f:

        dict_name  = file_with_split[file_with_split.rfind('/')+1 :]
        dict_name  = dict_name[:dict_name.find('_')]

        for line in f:
            line = line.decode('utf-8')
            params = line.split(' ')
            params_dictionary = dict()
    
            params_dictionary['dataset'] = dict_name
            
            path = params[0].split(':')[1]
            for param in params[1:]:
                    parsed = param.split(':')
                    key = parsed[0]
                    if key == 'label':
                        # make label start from 0
                        label = int(parsed[1]) - 1 
                        params_dictionary['label'] = label
                    elif key in ('depth','color','duo_left'):
                        #othrwise only sensors format: <sensor name>:<folder>:<start frame>:<end frame>
                        sensor_name = key
                        #first store path
                        params_dictionary[key] = path + '/' + parsed[1]
                        #store start frame
                        params_dictionary[key+'_start'] = int(parsed[2])

                        params_dictionary[key+'_end'] = int(parsed[3])
            params_dictionary['duo_right'] = params_dictionary['duo_left'].replace('duo_left', 'duo_right')
            params_dictionary['duo_right_start'] = params_dictionary['duo_left_start']
            params_dictionary['duo_right_end'] = params_dictionary['duo_left_end']          

            params_dictionary['duo_disparity'] = params_dictionary['duo_left'].replace('duo_left', 'duo_disparity')
            params_dictionary['duo_disparity_start'] = params_dictionary['duo_left_start']
            params_dictionary['duo_disparity_end'] = params_dictionary['duo_left_end']                  

            list_split.append(params_dictionary)
 
    return list_split


# In[70]:


def load_data_from_file(example_config, sensor,image_width, image_height):
    path = '../data/' + example_config[sensor][1:] + ".avi"
    start_frame = example_config[sensor+'_start']
    end_frame = example_config[sensor+'_end']
    label = example_config['label']

    frames_to_load = range(start_frame, end_frame)
    chnum = 3 if sensor == "color" else 1
    video_container = np.zeros((image_height, image_width, chnum, 80), dtype = np.uint8)
    cap = cv2.VideoCapture(path)
    ret = 1
    frNum = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for indx, frameIndx in enumerate(frames_to_load):    
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame,(image_width, image_height))
            if sensor != "color":
                frame = frame[...,0]
                frame = frame[...,np.newaxis]
            video_container[..., indx] = frame
        else:
            print("Could not load frame")
            
    cap.release()

    return video_container, label


# In[72]:


sensors = ["color", "depth", "duo_left", "duo_right", "duo_disparity"]
file_lists = dict()
file_lists["test"] = "../data/nvgesture_test_correct_cvpr2016.lst"
file_lists["train"] = "../data/nvgesture_train_correct_cvpr2016.lst"
train_list = list()
test_list = list()

train_list = load_split_nvgesture(file_with_split = file_lists["train"],list_split = train_list)
test_list = load_split_nvgesture(file_with_split = file_lists["test"],list_split = test_list)


# In[86]:


len(train_list)


# ## 制作dataloader

# In[90]:


class NVDataset(Dataset):
    def __init__(self, sensorId, modeId, trainList, testList):
        self.sensorId = sensorId
        self.modeId = modeId
        self.trainList = trainList
        self.testList = testList
        self.preList = trainList if modeId == 'train' else testList
        
    def __getitem__(self, idx):
        x, y = load_data_from_file(example_config = self.preList[idx], 
                                      sensor = sensors[self.sensorId], image_width = 320, image_height = 240)
        return x.transpose((2,3, 0, 1)), y
    
    def __len__(self):
        return len(self.preList)


# In[96]:


def getLoader(sensorId, mode, batchSize):
    dataset = NVDataset(sensorId, mode, train_list, test_list)
    shuffle = False if mode == 'test' else True
    dataloader = DataLoader(dataset, batch_size=batchSize, shuffle=shuffle, num_workers=4)
    return dataloader


# In[ ]:




