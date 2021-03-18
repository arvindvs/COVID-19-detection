import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch
import models
# imports:
# BaselineFCCOVIDDetector,
# BaselineConvCOVIDDetectorA,
# SimpleConvCOVIDDetector_2layer,
# SimpleConvCOVIDDetector_4layer,
# ConvCOVIDDetectorA,
# ConvCOVIDDetectorB,
# COVIDResNet,
# ConvCOVIDDetectorC
from torch.utils.data import Dataset, DataLoader
import torch
import torch.optim as optim
import numpy as np
from torch import nn
from dataLoader import COVIDDataset
from torchvision import transforms
import matplotlib.pyplot as plt
import os
import argparse
import pathlib

csv_file = 'data/processed_with_incomplete_classes/processed_metadata.csv'
root_dir = 'data/processed_with_incomplete_classes/images'
save_dir = 'saved_artifacts_confusion/'


batch_size=32
img_size=256
print_frequency = 100
save_frequency = 500



save_plot_path = os.path.join(save_dir, 'plots')
save_model_path = os.path.join(save_dir, 'models')
dataset = COVIDDataset(csv_file=csv_file, root_dir=data_dir, transform=transforms.Compose([
                                               transforms.Resize((img_size,img_size)),
                                               transforms.ToTensor()
                                           ]))
train_size = int(0.98 * len(dataset))
test_size = (len(dataset) - train_size)//2
val_size = len(dataset) - train_size - test_size
torch.manual_seed(42) # for reproducible test set
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
torch.manual_seed(torch.initial_seed())

dataloaders = {}
dataloaders['train'] = DataLoader(train_dataset, batch_size=batch_size,
                        shuffle=True, num_workers=0)
dataloaders['val'] = DataLoader(val_dataset, batch_size=batch_size,
                        shuffle=True, num_workers=0)
dataloaders['test'] = DataLoader(test_dataset, batch_size=batch_size,
                        shuffle=True, num_workers=0)


ckpt_path = 'saved_artifacts_networkC/models/ConvCOVIDDetectorC.ckpt'
  
model = torch.load(ckpt_path)




















