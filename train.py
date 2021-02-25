from models import BaselineFCCOVIDDetector, BaselineConvCOVIDDetector
import torch
import torch.optim as optim
import numpy as np
from torch import nn
from dataLoader import get_images_from_folder



def train():
    model = BaselineConvCOVIDDetector()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    img_dir = "milestone_images"

    batch_size = 1
    num_epochs = 50
    num_steps = 50
    num_classes = 3
    
    imgs, labels, _ = get_images_from_folder(img_dir)
    m = len(imgs)
    for epoch in range(num_epochs):
        for i in range(len(imgs)):
            img = imgs[i]
            label = labels[i]
            img = torch.reshape(img, (1, 1, 256, 256))
            # img shape: 256 x 256

            # torch.rand((batch_size, 3, 256, 256))
            # labels = torch.tensor(np.random.choice(num_classes, batch_size))

            optimizer.zero_grad()
            outputs = model(img)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

            iter_num = epoch*m + i
            if iter_num % 10 == 9:
                print(f'iter {iter_num}: loss = {loss}')



if __name__ == '__main__':
    train()