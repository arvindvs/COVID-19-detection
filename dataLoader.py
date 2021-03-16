#!/Users/madhu/anaconda3/bin/python

from __future__ import print_function, division
import os
import sys
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, utils
from scipy import misc
import glob
import imageio
from PIL import Image


class COVIDDataset(Dataset):
    """COVID-19 dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:

            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        assert len(self.frame) == len([name for name in os.listdir(self.root_dir) if os.path.isfile(os.path.join(self.root_dir, name))])

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.get_name(idx) + '.png')
        image = Image.fromarray(io.imread(img_name))
        label = int(self.get_label(idx))
        sample = {'image': image, 'label': label}

        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample

    def get_name(self, idx):
        name = self.frame.iloc[idx, 0]
        return name

    def get_label(self, idx):
        try:
            return self.frame.iloc[idx, 1]
        except IndexError:
            if 'COVID' in img: # COVID
                return 0
            elif 'Viral' in img: # PNEUMONIA
                return 1
            else: # NORMAL
                return 2
