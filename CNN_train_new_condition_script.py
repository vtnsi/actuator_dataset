# -*- coding: utf-8 -*-
"""

script for training a CNN on the actuator data

new condition experiment

"""

import os
from datetime import datetime
import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score

from torch.utils.data import DataLoader, random_split

from src.utils import read_configs, save_configs
from src.actuator_data_module import load_actuator_data, Actuator_Dataset
from src.trainers import Trainer
from src.model import CNN
from src.evaluators import Evaluator


configs_filename = "configs/train_params.json"

# actuator directories
Act_dir = [
    "Act_1",
    "Act_2",
    "Act_3 (Gear Damage)",
    "Act_4 (Seal Defect)",
    "Act_5",
    "Act_6",
]
Act_dir = ["data/" + s for s in Act_dir]


# list of feautres to extract and compress
sensor_names = [
    "Accel_1",
    "Accel_2",
    "Accel_3",
    "Angle",
    "Temp_1",
    "Temp_2",
    "PG_1",
    "PG_2",
    "Lim_2",
]


# label dictionary
label_dict = {"1": 0, "2": 0, "3": 1, "4": 1, "5": 0, "6": 0}
train_test_dict = {
    "1": "train",
    "2": "train",
    "3": "train",
    "4": "test",
    "5": "test",
    "6": "test",
}
cf_labels = ["Undamaged", "Damaged"]
num_classes = len(cf_labels)


# model params
model_params = {"Conv Filters 1": 128, "Conv Filters 2": 256, "Linear Layers": 3}

# cutoff of features
cutoff = 1000

# experimetn directory
exp_dir_name = "results/CNN_new_condition_"


# train test split function for new condition experiment
def train_test_split(data, actuator_number, train_test_dict):
    train_data, train_act_number = [], []
    test_data, test_act_number = [], []

    for n in range(data.shape[0]):
        if train_test_dict[actuator_number[n]] == "train":
            train_data.append(data[n, :, :])
            train_act_number.append(actuator_number[n])
        else:
            test_data.append(data[n, :, :])
            test_act_number.append(actuator_number[n])

    return np.array(train_data), train_act_number, np.array(test_data), test_act_number


def main():
    # create directory to save exp results
    exp_dir = exp_dir_name + datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    os.mkdir(exp_dir)

    # get training parameters
    train_params = read_configs(configs_filename)
    save_configs(train_params, os.path.join(exp_dir, 'train_params.json'))

    # load and actuator data
    print("Loading data...")
    data, actuator_number = load_actuator_data(Act_dir, sensor_names, cutoff=cutoff, standardize=True)

    # create train test data sets
    train_data, train_act_number, test_data, test_act_number = train_test_split(data, actuator_number, train_test_dict)

    # create dataset
    train_dataset = Actuator_Dataset(train_data, train_act_number, label_dict, num_classes)
    act_test = Actuator_Dataset(test_data, test_act_number, label_dict, num_classes)

    # split into train, val, and test sets
    act_train, act_val = random_split(train_dataset, [train_params["Train Proportion"], 1-train_params["Train Proportion"]])

    # create dataloaders
    train_dataloader = DataLoader(act_train, batch_size=train_params["Batch Size"], shuffle=True)
    val_dataloader = DataLoader(act_val, batch_size=train_params["Batch Size"], shuffle=False)
    test_dataloader = DataLoader(act_test, batch_size=train_params["Batch Size"], shuffle=False)

    # initialize model
    print("Training model...")
    model = CNN(data.shape[2], num_classes)
    trainer = Trainer(train_params['Epochs'], train_params['Learning Rate'], train_params['Print Every N'])
    model = trainer.train_model(model, train_dataloader, val_dataloader)
    trainer.plot_loss(os.path.join(exp_dir,'loss.png'))
    model.save_model(os.path.join(exp_dir, 'model.pth'))

    # evaluate on validation set
    print('Evaluating...')
    evaluator = Evaluator(model)
    accuracy = evaluator.evaluate(test_dataloader, accuracy_score)
    df = pd.DataFrame({'Accuracy': accuracy}, index = [0])
    df.to_csv(os.path.join(exp_dir, 'results.csv'), index = False)
    evaluator.confusion_matrix(test_dataloader, cf_labels, os.path.join(exp_dir, 'confusion_matrix.png'))



if __name__ == "__main__":
    main()
