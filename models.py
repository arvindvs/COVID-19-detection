import torch.nn as nn
import torch.nn.functional as F
import torchvision


class COVIDResNet(nn.Module):
  def __init__(self, num_classes=3, in_channels=1):
    super(COVIDResNet, self).__init__()

    # Load a pretrained resnet model from torchvision.models in Pytorch
    self.model = torchvision.models.resnet50(pretrained=True)

    # Change the input layer to take Grayscale image, instead of RGB images. 
    # Hence in_channels is set as 1 or 3 respectively
    # original definition of the first layer on the ResNet class
    # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    # Change the output layer to output 3 classes instead of 1000 classes
    num_ftrs = self.model.fc.in_features
    self.model.fc = nn.Linear(num_ftrs, num_classes)


  def forward(self, x):
    return self.model(x)


class ConvSkipBlock(nn.Module):
    def __init__(self, num_channels, hidden_channels, out_channels, drop_prob=0):
        super(ConvSkipBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, hidden_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(hidden_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(hidden_channels, num_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(num_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.dropout = nn.Dropout(p=drop_prob)
    
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
        out = self.dropout(out)
        return out

class ConvCOVIDDetectorA(nn.Module):
    def __init__(self, num_classes):
        super(ConvCOVIDDetectorA, self).__init__()
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

class ConvCOVIDDetectorB(nn.Module):
    def __init__(self, num_classes):
        super(ConvCOVIDDetectorB, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()
        self.conv_skip1 = ConvSkipBlock(16, 16, 64, drop_prob=0.1)
        self.conv_skip2 = ConvSkipBlock(64, 32, 128, drop_prob=0.2)
        self.conv_skip3 = ConvSkipBlock(128, 64, 128, drop_prob=0.3)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128*32*32, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
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

class ConvCOVIDDetectorBSmall(nn.Module):
    def __init__(self, num_classes):
        super(ConvCOVIDDetectorBSmall, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()
        self.conv_skip1 = ConvSkipBlock(16, 16, 64, drop_prob=0.1)
        self.conv_skip2 = ConvSkipBlock(64, 32, 128, drop_prob=0.2)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128*64*64, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv_skip1(x)
        x = self.maxpool(x)
        x = self.conv_skip2(x)
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
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 3)
        
    
    def forward(self, x):
        # x: 256 x 256
        x = self.flatten(x)
        # 1 x 256*256 = 1 x 64k
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        out = F.log_softmax(x, dim=1)
        return out

class BaselineConvCOVIDDetector(nn.Module):
    def __init__(self):
        super(BaselineConvCOVIDDetector, self).__init__()
        self.conv = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.relu1 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128*128*16, 256)
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(256, 3)
        
    
    def forward(self, x):
        x = self.conv(x)
        x = self.maxpool(x)
        x = self.relu1(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu2(x)
        x = self.fc2(x)
        out = F.log_softmax(x, dim=1)
        return out
