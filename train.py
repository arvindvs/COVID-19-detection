import models 
# imports:
# BaselineFCCOVIDDetector,
# BaselineConvCOVIDDetectorA,
# SimpleConvCOVIDDetector_2layer,
# SimpleConvCOVIDDetector_4layer,
# ConvCOVIDDetectorA,
# ConvCOVIDDetectorB,
# COVIDResNet,
# ConvCOVIDDetectorC
from torch.utils.data import Dataset, DataLoader
import torch
import torch.optim as optim
import numpy as np
from torch import nn
from dataLoader import COVIDDataset
from torchvision import transforms
import matplotlib.pyplot as plt
import os
import argparse
import pathlib

# csv_file = 'data/milestone/milestone_metadata.csv'
# root_dir = 'data/milestone/milestone_images'
# save_dir = 'saved_artifacts/'


batch_size=32
img_size=256
# num_epochs = 50
# num_classes = 3
print_frequency = 100
save_frequency = 500



def train(csv_file, data_dir, save_dir, num_classes, num_epochs):
    save_plot_path = os.path.join(save_dir, 'plots')
    save_model_path = os.path.join(save_dir, 'models')
    if torch.cuda.is_available():  
        print("Running on GPU.")
        dev = "cuda:0" 
    else:  
        print("Running on CPU.")
        dev = "cpu"
    device = torch.device(dev)
    dataset = COVIDDataset(csv_file=csv_file, root_dir=data_dir, transform=transforms.Compose([
                                               transforms.Resize((img_size,img_size)),
                                               transforms.ToTensor()
                                           ]))
    os.makedirs(save_plot_path, exist_ok=True)
    os.makedirs(save_model_path, exist_ok=True)

    train_size = int(0.98 * len(dataset))
    test_size = (len(dataset) - train_size)//2
    val_size = len(dataset) - train_size - test_size

    torch.manual_seed(42) # for reproducible test set
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
    torch.manual_seed(torch.initial_seed())

    dataloaders = {}
    dataloaders['train'] = DataLoader(train_dataset, batch_size=batch_size,
                        shuffle=True, num_workers=0)
    dataloaders['val'] = DataLoader(val_dataset, batch_size=batch_size,
                        shuffle=True, num_workers=0)
    dataloaders['test'] = DataLoader(test_dataset, batch_size=batch_size,
                        shuffle=True, num_workers=0)

    model = models.SimpleConvCOVIDDetector_4layer(num_classes=num_classes)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-5)
    train_loss = [] 
    val_loss = []
    val_accs = []
    iteration = 0
    print(f'Training model {model.__class__.__name__}')
    for epoch in range(num_epochs):
        model.train()
        total_correct = 0.0
        for i_batch, sample_batched in enumerate(dataloaders['train']):
            iteration += 1
            imgs = sample_batched['image'].to(device) # shape(batch_size, 1, img_size, img_size)
            labels = sample_batched['label'].to(device) # shape(batch_size,)
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_correct += torch.sum(torch.argmax(outputs, dim=1) == labels)
            loss.backward()
            optimizer.step()

            if iteration % print_frequency == 0:
                print(f'epoch {epoch}, iter {i_batch}: train loss = {loss}')
                train_loss.append(loss)
            if iteration % save_frequency == 0:
                print(f'epoch {epoch}, iter {i_batch}: saving model to {save_model_path}')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                }, os.path.join(save_model_path, model.__class__.__name__ + '.ckpt'))
        print(f'epoch {epoch}: train accuracy = {total_correct/train_size}')

        total_correct = 0.0
        val_losses = []
        with torch.no_grad():
            model.eval()
            for i_batch, sample_batched in enumerate(dataloaders['val']):
                imgs = sample_batched['image'] # shape(batch_size, 1, img_size, img_size)
                labels = sample_batched['label'] # shape(batch_size,)
                imgs = imgs.to(device)
                labels = labels.to(device)

                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_losses.append(loss.item())
                preds = torch.argmax(outputs, dim=1)
                total_correct += torch.sum(preds == labels)
        val_loss.append(np.mean(val_losses))
        val_acc = total_correct/val_size
        val_accs.append(val_acc)
        print(f'epoch {epoch}: val loss = {val_loss[-1]}, val accuracy = {val_acc}')
        
        # Plotting
        plt.plot(np.arange(len(train_loss))*print_frequency, train_loss, marker='o')  
        plt.title("Training set loss vs. iteration")
        plt.xlabel('iter')
        plt.ylabel('loss')
        plt.savefig(os.path.join(save_plot_path, 'train_loss_' + model.__class__.__name__ + '.png'))    
        plt.clf()
        plt.plot(range(epoch+1), val_loss, marker='o')  
        plt.xlabel('epoch')
        plt.ylabel('avg loss')
        plt.title("Average validation set loss vs. epoch") 
        plt.savefig(os.path.join(save_plot_path, 'val_loss_' + model.__class__.__name__ + '.png'))
        plt.clf()
        plt.plot(range(epoch+1), val_accs, marker='o')  
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.title("Validation accuracy vs. epoch") 
        plt.savefig(os.path.join(save_plot_path, 'val_acc_' + model.__class__.__name__ + '.png'))
        plt.clf()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-c', '--csv', type=pathlib.Path, default='data/milestone/milestone_metadata.csv', help='path to metadata csv')
    parser.add_argument('-d', '--data_dir', type=pathlib.Path, default='data/milestone/milestone_images', help='path to data directory containing images')
    parser.add_argument('-s', '--save_dir', type=pathlib.Path, default='saved_artifacts/', help='path to save model artifacts (plots, ckpts)')
    parser.add_argument('-n', '--num_classes', type=int, default=16, help='num classes to predict')
    parser.add_argument('-e', '--num_epochs', type=int, default=30, help='num epochs to train')
    args = parser.parse_args()

    train(args.csv, args.data_dir, args.save_dir, args.num_classes, args.num_epochs) 
    
