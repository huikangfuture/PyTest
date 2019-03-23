import os
import sys
import torch
import numpy as np

from time import time
from copy import deepcopy


class Progbar:
    def __init__(self, total, width=30):
        self.total = total
        self.width = width
        self.since = time()

    def update(self, index, prefix='', suffix='', end='\r'):
        interval = time() - self.since
        progress = round(self.width * (index + 1) / self.total)
        progressbar = '[' + '>' * progress + '-' * (self.width - progress) + ']'
        print(prefix, progressbar, '{:.2f}s'.format(interval), suffix, sep=' - ', end=end, flush=True)
        self.since += interval


class Trainer:
    def __init__(self, model, criterion, optimizer, scheduler=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

    def fit(self, dataloader, epochs=1, initial_epoch=0, device='cpu'):
        best_corrects = 0
        for epoch in range(initial_epoch, epochs):
            print('Epoch: {}/{}'.format(epoch + 1, epochs))

            for phase, loader in dataloader.items():
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()

                steps = len(loader)
                progbar = Progbar(steps)
                total_loss, total_corrects = 0, 0

                for i, (inputs, labels) in enumerate(loader):
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    self.optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = self.criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    total_loss += loss.item()
                    total_corrects += torch.sum(preds == labels).item() / labels.size(0)

                    batch_loss = total_loss / (i + 1)
                    batch_corrects = total_corrects / (i + 1)

                    prefix = '[{0:{2}d}/{1}]'.format(i + 1, steps, len(str(steps)))
                    suffix = '{2}loss: {0:.4f} - {2}acc: {1:.4f}'.format(batch_loss, batch_corrects, '' if phase  == 'train' else phase + '_')
                    progbar.update(i, prefix, suffix, '\r' if (i + 1) < steps else '\n')

                epoch_loss = total_loss / steps
                epoch_corrects = total_corrects / steps

                if phase == 'val' and self.scheduler is not None:
                    self.scheduler.step(epoch_loss)

                if phase == 'val' and epoch_corrects > best_corrects:
                    best_corrects = epoch_corrects
                    self.best_state = deepcopy({
                        'epoch': epoch + 1,
                        'model': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict()
                    })

        print('Best val acc: {:.2f}%'.format(best_corrects * 100))
