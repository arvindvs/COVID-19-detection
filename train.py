from models import BaselineFCCOVIDDetector, BaselineConvCOVIDDetector
import torch
import torch.optim as optim
import numpy as np
from torch import nn



def train():
    model = BaselineConvCOVIDDetector()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    batch_size = 32
    num_steps = 50
    num_classes = 3

    for i in range(num_steps):
        imgs = torch.rand((batch_size, 3, 256, 256))
        labels = torch.tensor(np.random.choice(num_classes, batch_size))

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if i % 10 == 9:
            print(f'iter {i}: loss = {loss}')



if __name__ == '__main__':
    train()