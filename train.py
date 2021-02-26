from models import BaselineFCCOVIDDetector, BaselineConvCOVIDDetector
from torch.utils.data import Dataset, DataLoader
import torch
import torch.optim as optim
import numpy as np
from torch import nn
from dataLoader import COVIDDataset
from torchvision import transforms

csv_file='data/milestone_images/milestone_metadata.csv'
root_dir='data/milestone_images'

batch_size=1
img_size=256
num_epochs = 50
num_classes = 3
print_frequency=1


def train():
    dataset = COVIDDataset(csv_file=csv_file, root_dir=root_dir, transform=transforms.Compose([
                                               transforms.Resize((img_size,img_size)),
                                               transforms.ToTensor()
                                           ]))
    dataloader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=True, num_workers=0)

    model = BaselineConvCOVIDDetector()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(num_epochs):
        for i_batch, sample_batched in enumerate(dataloader):
            imgs = sample_batched['image'] # shape(batch_size, 1, img_size, img_size)
            labels = sample_batched['label'] # shape(batch_size,)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()


            if i_batch % print_frequency == 0:
                print(f'epoch {epoch}, iter {i_batch}: loss = {loss}')


if __name__ == '__main__':
    train()
