# -*- coding: utf-8 -*-
"""

script for training a CNN on the actuator data

damage type experiment

"""


import os
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from math import floor

from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from src.utils import write2file
from src.actuator_data_module import load_actuator_data, Actuator_Dataset
from src.pytorch_training_utils import mlp_train_func, predict_func
from src.model import CNN


# actuator directories 
Act_dir = ['Act_1', 'Act_2', 'Act_3 (Gear Damage)', 'Act_4 (Seal Defect)', 'Act_5', 'Act_6']
Act_dir = ['Data/' + s for s in Act_dir]


# list of feautres to extract and compress
sensor_names = ['Accel_1', 'Accel_2', 'Accel_3', 'Angle', 'Temp_1', 'Temp_2', 'PG_1', 'PG_2', 'Lim_2']


# label dictionary
label_dict = {'1': 0, '2': 0, '3': 1, '4': 2, '5': 0, '6': 0}
cf_labels = ['Undamaged', 'Gear Damage', 'Seal Defect']
num_classes = len(cf_labels)


# model train parameters
train_params = {'Batch Size': 32,
                'Learning Rate': 0.0001,
                'Epochs': 10000,
                'Print Every N': 100,
                'Train Proportion': 0.6,
                'Validation Proportion': 0.2,
                'Test Proportion': 0.2}

# model params
model_params = {'Conv Filters 1': 128,
                'Conv Filters 2': 256,
                'Linear Layers': 3}

# cutoff of features
cutoff = 1000

# experiment directory
exp_dir_name = 'CNN_damage_'




def main():
    
    # create directory to save exp results
    exp_dir = exp_dir_name + datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    os.mkdir(exp_dir)
    
    
    # save data
    write2file(exp_dir + '/train_params.txt', train_params)
    write2file(exp_dir + '/model_params.txt', model_params)
    
    
    # load and actuator data
    print('Loading data...')
    data, actuator_number = load_actuator_data(Act_dir, sensor_names, cutoff=cutoff, standardize=True)
    
    # create dataset
    act_dataset = Actuator_Dataset(data, actuator_number, label_dict, num_classes)
    
    N = len(act_dataset)
    train_num = floor(train_params['Train Proportion']*N)
    val_num = floor(train_params['Validation Proportion']*N)
    test_num = N - train_num - val_num
    
    
    # split into train, val, and test sets
    act_train, act_val, act_test = random_split(act_dataset, [train_num, val_num, test_num])
    
    # create dataloaders
    train_dataloader = DataLoader(act_train, batch_size=train_params['Batch Size'], shuffle=True)
    val_dataloader = DataLoader(act_val, batch_size=train_params['Batch Size'], shuffle=False)
    test_dataloader = DataLoader(act_test, batch_size=train_params['Batch Size'], shuffle=False)
    
    # initialize model
    model = CNN(data.shape[2], num_classes)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_params['Learning Rate'])    
    
    print('Training model...')
    model, train_loss, val_loss = mlp_train_func(model, 
                                                 optimizer, 
                                                 loss_fn,
                                                 train_dataloader, 
                                                 val_dataloader, 
                                                 epochs=train_params['Epochs'],
                                                 print_every_n=train_params['Print Every N'])
    
    # save model
    torch.save(model.state_dict(), exp_dir + '/model_weights.pth')
    
    # plot loss
    fig, ax = plt.subplots()
    ax.plot(train_loss, color='b', label='Train')
    ax.plot(val_loss, color='r', label='Validation')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    fig.legend()
    plt.savefig(exp_dir + '/loss.png')
    
    
    print(' ')
    print('Testing model...')
    y_test, y_pred = predict_func(model, test_dataloader)
    
    # accuracy
    test_accuracy = accuracy_score(y_test, y_pred)
    print('Test accuracy: ', test_accuracy)
    test_accuracy_df = pd.DataFrame({'Test_accuracy': test_accuracy}, index=[0])
    test_accuracy_df.to_csv(exp_dir + '/test_accuracy.csv')
    
    
    # confusion matrix
    fig, ax = plt.subplots(figsize=(12,12))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred,
                                            display_labels=cf_labels,
                                            xticks_rotation='vertical',
                                            ax=ax)   
    plt.savefig(exp_dir + '/confusion_matrix.png')
    
    
    
    
if __name__ == "__main__":
    main()