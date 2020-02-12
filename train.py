import os

import torch
from torch import nn
from torch.optim import lr_scheduler, Adam
from torch.utils.data import Dataset, DataLoader

from src.config import DATASET_PATH
from src.dataset import SpeechDataset
from src.model import SpectralUNet
from src.train import train_model_multihead

print('Creating datasets')
datasets = {x: SpeechDataset(os.path.join(DATASET_PATH, x)) 
              for x in ['train', 'val']}
dataloaders = {x: DataLoader(datasets[x], batch_size=32, shuffle=True, 
                             num_workers=2)
              for x in ['train', 'val']}
dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = SpectralUNet().to(device)

criterion_mask = nn.MSELoss()
criterion_class = nn.BCEWithLogitsLoss()
optimizer = Adam(model.parameters(), lr=0.0001)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
train_model_multihead(dataloaders, model, criterion_mask, criterion_class, optimizer, exp_lr_scheduler, 
                                     device, dataset_sizes, num_epochs=25)
