import os
import sys
import time
import torch
import numpy as np


__all__ = ['Progbar', 'Trainer']


class Progbar:
    def __init__(self, steps, width=50):
        self.steps = steps
        self.width = width
        self.since = time.time()

    def update(self, step, prefix='', suffix='', end='\r'):
        interval = time.time() - self.since
        progress = round(self.width * (step + 1) / self.steps)
        progressbar = '|' + '#' * progress + ' ' * (self.width - progress) + '|'
        print(prefix, progressbar, '{:.2f}s'.format(interval), suffix, sep=' - ', end=end, flush=True)
        self.since += interval


class Trainer:
    def __init__(self, model, criterion, optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

    def fit(self, dataloader, epochs=1, initial_epoch=0, device='cpu', checkpoint=None):
        best_accuracy = 0.

        for epoch in range(initial_epoch, epochs):
            print('Epoch: {}/{}'.format(epoch + 1, epochs))

            for phase in ['train', 'val']:
                if phase == 'val':
                    self.model.eval()
                else:
                    self.model.train()

                steps = len(dataloader[phase])
                progbar = Progbar(steps, width=50)
                total_loss, total_accuracy = 0., 0.

                for step, (inputs, labels) in enumerate(dataloader[phase]):
                    inputs, labels = inputs.to(device), labels.to(device)

                    self.optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        predict = outputs.max(dim=1)[1]
                        loss = self.criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    total_loss += loss.item()
                    total_accuracy += (predict == labels).sum().item() / labels.size(0)

                    batch_loss = total_loss / (step + 1)
                    batch_accuracy = total_accuracy / (step + 1)

                    prefix = '[{0:{2}d}/{1}]'.format(step + 1, steps, len(str(steps)))
                    suffix = '{2}loss: {0:.4f} - {2}acc: {1:.4f}'.format(batch_loss, batch_accuracy, 'val_' if phase  == 'val' else '')
                    progbar.update(step, prefix, suffix, '\r' if (step + 1) < steps else '\n')

                epoch_loss = total_loss / steps
                epoch_accuracy = total_accuracy / steps

                if phase == 'val' and epoch_accuracy > best_accuracy:
                    best_accuracy = epoch_accuracy

                if phase == 'val' and type(checkpoint) is str and os.path.isdir(checkpoint):
                    state_dict = {
                        'epoch': epoch + 1,
                        'model': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict()
                    }
                    state_dict_name = 'epoch_{0:0>{3}d}-loss_{1:.4f}-acc_{2:.4f}.pth.tar'.format(epoch + 1, epoch_loss, epoch_accuracy, len(str(epochs)))
                    torch.save(state_dict, os.path.join(checkpoint, state_dict_name))

        print('Best accuracy is {:.2f}%'.format(best_accuracy * 100))
