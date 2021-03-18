import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch

ckpt_path = 'data/ConvCOVIDDetectorC.ckpt' 
model = torch.load(ckpt_path, map_location=torch.device('cpu'))
