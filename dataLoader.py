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

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.frame.iloc[idx, 0] + '.png')
        image = Image.fromarray(io.imread(img_name))
        label = int(self.get_label(idx))
        sample = {'image': image, 'label': label}

        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample

    def get_name(self, idx):
        name = self.frame.iloc[idx, 0]

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


# def get_images_from_folder(img_path):
#     all_images = []
#     labels = [] # 0 = covid, 1 = pneumonia, 2 = normal
#     img_names = os.listdir(img_path)
#     for img in img_names:
#         if 'COVID' in img:
#             labels.append(torch.tensor([0]))
#         elif 'Viral' in img:
#             labels.append(torch.tensor([1]))
#         else:
#             labels.append(torch.tensor([2]))
#         full_path = os.path.join(img_path, img)
#         img = imageio.imread(full_path)
#         new_img = transform.resize(img, (256,256))
#         new_img /= 255.0
#         img_tensor = torch.FloatTensor(new_img)
#         all_images.append(img_tensor)
#     return all_images, labels, img_names

#################################################################################################

# 
# def main():
#     # require that the user specify an input and output path
#     assert len(sys.argv) == 3
#     input_dir, output_dir = sys.argv[1:]
#
#     loaded_dataset, loaded_labels, loaded_names = get_images_from_folder(input_dir)
#     rescaled_images = []
#     for i in range(len(loaded_dataset)):
#         img = loaded_dataset[i]
#         new_img = transform.resize(img, (256,256))
#
#         new_img /= 255.0
#         processed_input = torch.tensor(new_img)
#         torch.save(processed_input, os.path.join(output_dir, os.path.splitext(loaded_names[i])[0] + '.pt'))
#
#
# if __name__=="__main__":
#     main()
