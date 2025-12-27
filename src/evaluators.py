# -*- coding: utf-8 -*-
"""
evaluator class
"""

import numpy as np
import matplotlib.pyplot as plt

import torch
from sklearn.metrics import ConfusionMatrixDisplay


class Evaluator():
    def __init__(self, model):
       
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            print("Using cuda for evaluation")
            model.to('cuda')
        else:
            print("Using cpu for evaluation")
           
        self.model = model.eval()
   
   
    def evaluate(self, dataloader, metric):
        y, y_pred = self.get_y(dataloader)
        return metric(y, y_pred)
       
           
    def get_y(self, dataloader):
           
        y_list, y_pred = [], []
        for x, y in dataloader:
           
            if self.device == 'cuda':
                x = x.to('cuda')
               
            logit = self.model.classify(x)
            y_pred.append(torch.argmax(logit, dim=1).to('cpu').detach().numpy())
            y_list.append(torch.argmax(y, dim=1).detach().numpy())
               
        return np.concatenate(y_list), np.concatenate(y_pred)
   
   
    def confusion_matrix(self, dataloader, labels, save_name=None):
        y, y_pred = self.get_y(dataloader)
       
        fig, ax = plt.subplots(figsize=(4.5, 3))
        ConfusionMatrixDisplay.from_predictions(y, y_pred, display_labels=labels, ax=ax)
        fig.tight_layout()
        if save_name:
            plt.savefig(save_name)
