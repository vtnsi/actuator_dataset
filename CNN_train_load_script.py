# -*- coding: utf-8 -*-
"""

script for training a CNN on the actuator data

load classification experiment

"""

import os
from datetime import datetime
import pandas as pd

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
label_dict = {"1": 0, "2": 1, "3": 1, "4": 0, "5": 2, "6": 2}
cf_labels = ["Butterfly", "Ball", "No Load"]
num_classes = len(cf_labels)



# model params
model_params = {"Conv Filters 1": 128, "Conv Filters 2": 256, "Linear Layers": 3}

# cutoff of features
cutoff = 1000

# experiment directory
exp_dir_name = "results/CNN_load_"


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

    # create datasets
    act_dataset = Actuator_Dataset(data, actuator_number, label_dict, num_classes)
    act_train, act_val, act_test = random_split(act_dataset, [train_params["Train Proportion"], train_params["Validation Proportion"], train_params["Test Proportion"]])
    
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
