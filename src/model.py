# -*- coding: utf-8 -*-
"""

pytorch model

"""

import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, num_features, num_classes):
        super(CNN, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes

        self.conv1 = nn.Conv1d(self.num_features, 128, 8)
        self.conv2 = nn.Conv1d(128, 256, 8)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(4)

        self.linear1 = nn.LazyLinear(256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, self.num_classes)

    def forward(self, x):
        conv_output_1 = self.maxpool(self.relu(self.conv1(x)))
        conv_output_2 = self.maxpool(self.relu(self.conv2(conv_output_1))).flatten(1)
        linear_output_1 = self.relu(self.linear1(conv_output_2))
        linear_output_2 = self.relu(self.linear2(linear_output_1))
        linear_output_3 = self.linear3(linear_output_2)
        return linear_output_3
    
    def classify(self, x):
        logit = self.forward(x)
        return nn.functional.softmax(logit, dim=1)
    
    def save_model(self, save_filename):
         torch.save(self.state_dict(), save_filename)
         
    def load_model(self, load_filename):
         self.load_state_dict(torch.load(load_filename))

