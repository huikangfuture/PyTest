import os
import sys
import torch
import numpy as np

from tqdm import trange
from copy import deepcopy


class Trainer:
    def __init__(self, model, criterion, optimizer, scheduler=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

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
                progbar = trange(steps, ncols=100, ascii=True)
                total_loss, total_acc = 0, 0

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
                    total_acc += torch.sum(preds == labels).item() / labels.size(0)

                    batch_loss = total_loss / (i + 1)
                    batch_acc = total_acc / (i + 1)

                    progbar.set_postfix_str('loss: {:.4f}, acc: {:.4f}'.format(batch_loss, batch_acc))
                    progbar.update()

                progbar.close()

                epoch_loss = total_loss / steps
                epoch_acc = total_acc / steps

                if phase == 'val' and self.scheduler is not None:
                    self.scheduler.step(epoch_loss)

                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_state = deepcopy({
                        'epoch': epoch + 1,
                        'model': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict()
                    })

        print('Best val acc: {:.2f}%'.format(best_acc * 100))
