# -*- coding: utf-8 -*-
"""

pytorch training functions

"""


import torch
import numpy as np


'''
train function for an mlp
'''
def mlp_train_func(model, 
                   optimizer, 
                   loss_fn,
                   train_dataloader, 
                   val_dataloader, 
                   epochs=100,
                   print_every_n=1):
    
    
    # check if gpu available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using ' + str(device) + ' for training')
    
    # move model to gpu
    if torch.cuda.is_available():
        model.to(device)
    
    train_loss = []
    val_loss = []
    for epoch in range(epochs):
        
        model.train(True)
        
        running_loss = 0
        for (x, y) in train_dataloader:
            
            if torch.cuda.is_available():
                x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(x)
            
            loss = loss_fn(outputs, y.reshape(outputs.shape))
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        train_loss.append(running_loss)
        
        # evaluate on test set
        running_val_loss = 0
        for (x, y) in val_dataloader:
            
            if torch.cuda.is_available():
                x, y = x.to(device), y.to(device)
            
            test_outputs = model(x)
            running_val_loss += loss_fn(test_outputs, y.reshape(test_outputs.shape)).item()
            
        val_loss.append(running_val_loss)
        
        if (epoch % print_every_n) == 0:
            print('EPOCH {}'.format(epoch))
            print('Training loss: {}'.format(running_loss))
            print('Validation loss: {}'.format(running_val_loss))
            
    return model, train_loss, val_loss



'''
function to get y and y_pred from data loader
'''
def predict_func(model, dataloader):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    y_array = []
    y_pred_array = []
    for X, y in dataloader:
        if device == 'cuda':
            X = X.cuda()
            
        _, y_target = torch.max(y, dim=1)
        y_array.append(y_target.numpy())
            
        output = model(X)
        _, y_pred = torch.max(output, dim=1)
        y_pred_array.append(y_pred.cpu())

    return np.concatenate(y_array), np.concatenate(y_pred_array)