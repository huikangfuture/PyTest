import os
import sys
import torch
import numpy as np

from time import time
from copy import deepcopy


class Progbar:
    def __init__(self, steps, width=50):
        self.steps = steps
        self.width = width
        self.since = time()

    def update(self, step, prefix='', suffix='', end='\r'):
        interval = time() - self.since
        progress = round(self.width * (step + 1) / self.steps)
        progressbar = '[' + '>' * progress + '-' * (self.width - progress) + ']'
        print(prefix, progressbar, '{:.2f}s'.format(interval), suffix, sep=' - ', end=end, flush=True)
        self.since += interval


class Trainer:
    def __init__(self, model, criterion, optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

    def fit(self, dataloader, epochs=1, initial_epoch=0, device='cpu'):
        best_acc = 0
        for epoch in range(initial_epoch, epochs):
            print('Epoch: {}/{}'.format(epoch + 1, epochs))

            for phase, loader in dataloader.items():
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()

                steps = len(loader)
                progbar = Progbar(steps)
                total_loss, total_acc = 0, 0

                for step, (inputs, labels) in enumerate(loader):
                    inputs, labels = inputs.to(device), labels.to(device)

                    self.optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        _, pred = torch.max(outputs, 1)
                        loss = self.criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    acc = torch.sum(pred == labels).item() / labels.size(0)
                    total_loss, total_acc = total_loss + loss.item(), total_acc + acc
                    batch_loss, batch_acc = total_loss / (step + 1), total_acc / (step + 1)

                    prefix = '[{0:{2}d}/{1}]'.format(step + 1, steps, len(str(steps)))
                    suffix = '{2}loss: {0:.4f} - {2}acc: {1:.4f}'.format(batch_loss, batch_acc, '' if phase  == 'train' else phase + '_')
                    progbar.update(step, prefix, suffix, '\r' if (step + 1) < steps else '\n')

                epoch_loss, epoch_acc = total_loss / steps, total_acc / steps

                if 'val' not in loader or phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    self.best_state = deepcopy({
                        'epoch': epoch + 1,
                        'model': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict()
                    })

        print('Best accuracy is {:.2f}%'.format(best_acc * 100))
