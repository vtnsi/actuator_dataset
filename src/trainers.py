# -*- coding: utf-8 -*-
"""

trainer class

"""

import matplotlib.pyplot as plt

import torch
import torch.nn as nn


class Trainer():
    def __init__(self,
                 epochs: int = 100, 
                 learning_rate: float = 0.0001,
                 print_every_n: int = 1):
    
        self.epochs = epochs
        self.lr = learning_rate
        self.print_every_n = print_every_n
        
        self.loss_fn = nn.CrossEntropyLoss()
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            print("Using cuda for training")
        else:
            print("Using cpu for training")
        
        
    def train_model(self, model, dataloader_train, dataloader_val = None):
        
        # setup optimizer
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        
        if dataloader_val is not None:
            self.validate = True
            self.val_loss = []
        
        # move model to gpu
        if self.device == 'cuda':
            model.to('cuda')
            
        self.train_loss = []
        
        for epoch in range(self.epochs):
            
            model.train(True)
            model, epoch_train_loss = self._train_step(model, dataloader_train)
            self.train_loss.append(epoch_train_loss)
            
            if self.validate:
                model.eval()
                model, epoch_val_loss = self._train_step(model, dataloader_val, train = False)
                self.val_loss.append(epoch_val_loss)
                
            if (epoch % self.print_every_n) == 0:
                print('EPOCH {}'.format(epoch))
                print('Training loss: {}'.format(epoch_train_loss))
                if self.validate:
                    print('Validation loss: {}'.format(epoch_val_loss))
            
        return model
    
    
    def _train_step(self, model, dataloader, train = True):
        batch_loss = 0
        for (x, y) in dataloader:
            
            if self.device == 'cuda':
                x, y = x.to('cuda'), y.to('cuda')
            
            outputs = model(x)
            loss = self.loss_fn(outputs, y)
            
            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            batch_loss += loss.item()
            
        return model, batch_loss
    
        
    
    
    def plot_loss(self, save_dir = None):
        '''
        function to plot the training and validation loss
        '''
        fig, ax = plt.subplots()
        ax.plot(self.train_loss, color='b', label = 'Training')
        if self.validate:
            ax.plot(self.val_loss, color='r', label = 'Validation')    
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        
        if self.validate:
            ax.set_title('Training and Validation Loss')
        else:
            ax.set_title('Training Loss')
        fig.legend()
        
        if save_dir:
            plt.savefig(save_dir)
            plt.close()
            
    