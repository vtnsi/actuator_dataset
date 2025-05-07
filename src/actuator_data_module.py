# -*- coding: utf-8 -*-
"""

actuator data module

"""

from os import listdir
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset


'''
function for loading actuator data

Inputs:
    directories - list of directories to load
    sensor_names - list of sensors to extract from the files
    cutoff - point to trim the data
    standardize - option to standardize the data on each sequence.  Default
        is False.
    
Outputs:
    data - numerical data for each actuator sequence
    actuator_number - list of actuator numbers corresponding to the sequence
'''
def load_actuator_data(directories, sensor_names, cutoff=3000, standardize=False):
    
    # initialize to store data
    data = []
    actuator_number = []
    
    # iterate over directories
    for d in directories:
        
        # get files in d
        act_filenames = listdir(d)
        
        # iterate over files
        for filename in act_filenames:
            
            # read data
            temp = pd.read_csv(d + '/' + filename, sep='\t', usecols=sensor_names).iloc[:cutoff,:].values
            
            # standardize data
            if standardize:
                temp = (temp-np.mean(temp, axis=0))/np.std(temp, axis=0)
            
            data.append(temp)
            actuator_number.append(d[d.find('Act_')+len('Act_')])

    return np.array(data), actuator_number



'''
Custom dataset for pytorch

data - actuator data in a numpy array
label - actuator number in a list
label_dict - dictionary that defines conversion from label to classes
num_classes - number of classes
transpose
'''
class Actuator_Dataset(Dataset):
    
    def __init__(self, data, label, label_dict, num_classes):
        self.data = data
        self.label = label
        self.label_dict = label_dict
        self.num_classes = num_classes
        
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        
        X = torch.transpose(torch.from_numpy(self.data[idx]).to(torch.float), 0, 1)
    
        y = torch.zeros(self.num_classes)
        y[self.label_dict[self.label[idx]]] = 1
        
        return X, y
    






























