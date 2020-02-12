import copy
import time
import os

import numpy as np
import torch
import tqdm

from .config import MODEL_DIR


def train_model_multihead(dataloaders, model, criterion_mask, criterion_class, optimizer, 
                          scheduler, device, dataset_sizes, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = np.inf

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_mse = 0
            running_acc = 0

            # Iterate over data.
            for inputs, targets_mask, targets_class in tqdm(dataloaders[phase]):
                inputs = inputs.to(device, dtype=torch.float)
                targets_mask = targets_mask.to(device, dtype=torch.float)
                targets_class = targets_class.to(device, dtype=torch.float)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs_mask, outputs_class = model(inputs)
                    loss = criterion_mask(outputs_mask, targets_mask) + \
                           criterion_class(outputs_class, targets_class)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_mse += torch.sum(torch.pow(outputs_mask - targets_mask, 2))
                running_acc += torch.sum((outputs_class > 0.5).double() == targets_class)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_acc.double() / dataset_sizes[phase]
            epoch_mse = running_mse.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f} MSE: {:.4f}'.format(
                phase, epoch_loss, epoch_acc, epoch_mse))

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, f'unet_{epoch_loss:.2f}.pth'))
    print(f'Saved fitted model to {MODEL_DIR}')
    return model