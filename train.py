from models import BaselineFCCOVIDDetector, BaselineConvCOVIDDetector
from torch.utils.data import Dataset, DataLoader
import torch
import torch.optim as optim
import numpy as np
from torch import nn
from dataLoader import COVIDDataset
from torchvision import transforms
import matplotlib.pyplot as plt

csv_file='data/milestone_metadata.csv'
root_dir='data/milestone_images'

batch_size=32
img_size=256
num_epochs = 50
num_classes = 3
print_frequency=2


def train():
    dataset = COVIDDataset(csv_file=csv_file, root_dir=root_dir, transform=transforms.Compose([
                                               transforms.Resize((img_size,img_size)),
                                               transforms.ToTensor()
                                           ]))

    train_size = int(0.9 * len(dataset))
    
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    dataloaders = {}

    dataloaders['train'] = DataLoader(train_dataset, batch_size=batch_size,
                        shuffle=True, num_workers=0)
    dataloaders['val'] = DataLoader(val_dataset, batch_size=batch_size,
                        shuffle=True, num_workers=0)

    model = BaselineFCCOVIDDetector()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    train_loss = [] 
    val_loss = []
    for epoch in range(num_epochs):
        model.train()
        for i_batch, sample_batched in enumerate(dataloaders['train']):
            imgs = sample_batched['image'] # shape(batch_size, 1, img_size, img_size)
            labels = sample_batched['label'] # shape(batch_size,)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            train_loss.append(loss)
            loss.backward()
            optimizer.step()


            if i_batch % print_frequency == 0:
                print(f'epoch {epoch}, iter {i_batch}: loss = {loss}')
        
        model.eval()
        total_correct = 0.0
        num_batches = 0.0
        for i_batch, sample_batched in enumerate(dataloaders['val']):
            imgs = sample_batched['image'] # shape(batch_size, 1, img_size, img_size)
            labels = sample_batched['label'] # shape(batch_size,)
            
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss.append(loss)
            preds = torch.argmax(outputs, dim=1)
            total_correct += torch.sum(preds == labels)
            num_batches += 1
        val_acc = total_correct/val_size
        print(f'epoch {epoch} val accuracy: {val_acc}')
    plt.plot(range(len(train_loss)), train_loss )  
    plt.title("Training set loss vs. iteration")
    plt.show()    
    plt.plot(range(len(val_loss)), val_loss )      
    plt.title("Validation set loss vs. iteration") 
    plt.show()



if __name__ == '__main__':
    train() 
    
