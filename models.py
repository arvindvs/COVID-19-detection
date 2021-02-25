import torch.nn as nn
import torch.nn.functional as F

class BaselineFCCOVIDDetector(nn.Module):
    def __init__(self):
        super(BaselineFCCOVIDDetector, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256*256, 256)
        self.fc2 = nn.Linear(256, 3)
        
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        out = F.log_softmax(x, dim=1)
        return out

class BaselineConvCOVIDDetector(nn.Module):
    def __init__(self):
        super(BaselineConvCOVIDDetector, self).__init__()
        self.conv = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128*128*16, 256)
        self.fc2 = nn.Linear(256, 3)
        
    
    def forward(self, x):
        x = self.conv(x)
        x = self.maxpool(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        out = F.log_softmax(x, dim=1)
        return out
