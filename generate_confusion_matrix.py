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
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from dataLoader import COVIDDataset
from sklearn.metrics import confusion_matrix


checkpoint_path = os.path.expanduser("~/Downloads/ConvCOVIDDetectorC.ckpt")
output_path = os.path.expanduser("~/Downloads/confusion_matrix.png")

data_dir = os.path.expanduser("~/Downloads/processed_with_incomplete_classes/images")
csv_file = os.path.expanduser("~/Downloads/processed_with_incomplete_classes/processed_metadata.csv")

img_size=256
batch_size=500
num_classes=16
num_batches=1

use_percentage=True
show_numbers=True

string_labels = [
    'COVID',
    'Pneumonia',
    'Normal',
    'Atelectasis',
    'Consolidation',
    'Infiltration',
    'Pneumothorax',
    'Edema',
    'Emphysema',
    'Fibrosis',
     'Effusion',
     'Pleural_Thickening',
     'Cardiomegaly',
     'Nodule',
     'Mass',
     'Hernia',
]


def main():
    model = models.ConvCOVIDDetectorC(num_classes)
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu'))['model_state_dict'])
    dataset = COVIDDataset(csv_file=csv_file, root_dir=data_dir, transform=transforms.Compose([
                                               transforms.Resize((img_size,img_size)),
                                               transforms.ToTensor()
                                           ]))

    images, labels = load_batch(dataset)

    outputs = torch.argmax(model(images), dim=1)

    cm = confusion_matrix(
        [string_labels[l] for l in labels],
        [string_labels[o] for o in outputs],
        labels=string_labels
    )

    if use_percentage:
        data_size = len(images)
        cm = np.multiply(np.divide(cm, data_size), 100)

    df_cm = pd.DataFrame(cm, index=string_labels, columns=string_labels)
    plt.figure(figsize = (10,10))
    sn.heatmap(df_cm, annot=show_numbers)
    plt.savefig(os.path.join(output_path))




def load_batch(dataset):
    train_size = int(0.98 * len(dataset))
    test_size = (len(dataset) - train_size)//2
    val_size = len(dataset) - train_size - test_size

    torch.manual_seed(42) # for reproducible test set
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
    torch.manual_seed(torch.initial_seed())

    data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    batch = next(iter(data_loader))
    return batch['image'], batch['label']


if __name__=="__main__":
    main()
