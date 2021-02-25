#!/Users/madhu/anaconda3/bin/python

from __future__ import print_function, division
import os
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


img_dir = "/Users/madhu/Desktop/milestone_images/" 

def get_images_from_folder(img_path): 
	all_images = [] 
	labels = [] # 0 = covid, 1 = pneumonia, 2 = normal 
	img_names = os.listdir(img_path)
	for img in img_names:
		if 'COVID' in img: 
			labels.append(0) 
		elif 'Viral' in img: 
			labels.append(1) 
		else: 
			labels.append(2)
		full_path = os.path.join(img_path, img)
		im = imageio.imread(full_path)
		im_pil = Image.fromarray(im)

		all_images.append(im_pil) 
	return all_images, labels

#################################################################################################
loaded_dataset, loaded_labels = get_images_from_folder(img_dir)  
rescaled_images = []
for image in loaded_dataset:  
	rescaled = transforms.RandomResizedCrop(256)
	#(x, y) want to be (256, 256) 
	#x , y * (y, 256) = x, 256 
	new_img = rescaled(image)
	rescaled_images.append(new_img) 