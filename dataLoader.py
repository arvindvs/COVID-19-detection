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
from torchvision import transforms, utils
from torchvision import datasets, transforms
from scipy import misc
import glob
import imageio
from PIL import Image


# img_dir = "/Users/madhu/Desktop/milestone_images/"

def get_images_from_folder(img_path):
    all_images = []
    labels = [] # 0 = covid, 1 = pneumonia, 2 = normal
    img_names = os.listdir(img_path)
    for img in img_names:
        if 'COVID' in img:
            labels.append(torch.tensor([0]))
        elif 'Viral' in img:
            labels.append(torch.tensor([1]))
        else:
            labels.append(torch.tensor([2]))
        full_path = os.path.join(img_path, img)
        img = imageio.imread(full_path)
        new_img = transform.resize(img, (256,256))
        new_img /= 255.0
        img_tensor = torch.FloatTensor(new_img)
        all_images.append(img_tensor)
    return all_images, labels, img_names

#################################################################################################


def main():
    # require that the user specify an input and output path
    assert len(sys.argv) == 3
    input_dir, output_dir = sys.argv[1:]

    loaded_dataset, loaded_labels, loaded_names = get_images_from_folder(input_dir)
    rescaled_images = []
    for i in range(len(loaded_dataset)):
        img = loaded_dataset[i]
        new_img = transform.resize(img, (256,256))

        new_img /= 255.0
        processed_input = torch.tensor(new_img)
        torch.save(processed_input, os.path.join(output_dir, os.path.splitext(loaded_names[i])[0] + '.pt'))


if __name__=="__main__":
    main()
