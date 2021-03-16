import torch.nn as nn
import torch.nn.functional as F

class ConvSkipBlock(nn.Module):
    def __init__(self, num_channels, hidden_channels, out_channels):
        super(ConvSkipBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, hidden_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(hidden_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(hidden_channels, num_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(num_channels, out_channels, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        x_tmp = x
        out = self.conv1(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = out + x_tmp
        out = self.relu(out)
        out = self.conv3(out)
        out = self.relu(out)
        return out

class ConvCOVIDDetector(nn.Module):
    def __init__(self, num_classes):
        super(ConvCOVIDDetector, self).__init__()
        self.conv_skip1 = ConvSkipBlock(1, 16, 32)
        self.conv_skip2 = ConvSkipBlock(32, 16, 64)
        self.conv_skip3 = ConvSkipBlock(64, 16, 32)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32*32*32, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        x = self.conv_skip1(x)
        x = self.maxpool(x)
        x = self.conv_skip2(x)
        x = self.maxpool(x)
        x = self.conv_skip3(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        out = F.log_softmax(x, dim=1)
        return out

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
