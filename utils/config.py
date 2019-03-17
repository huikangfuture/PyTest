import os
import sys
import glob
import torch
import warnings
import functools


class Config:
    def __init__(self, **kwargs):
        self.data = {
            'mnist': os.path.expanduser('~/.torch/datasets/MNIST/processed'),
            'dogcat': os.path.expanduser('D:/Workspace/Resources/dogs-vs-cats'),
            'cifar10': os.path.expanduser('~/.torch/datasets/cifar-10-batches-py')
        }

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.checkpoint = os.path.join(os.getcwd(), 'checkpoints')

        self.epochs = 10
        self.initial_epoch = 0

        self.lr = 1e-4
        self.step_lr = 10
        self.batch_size = 32

        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                warnings.warn('Config has no attr of \'{}\''.format(k))

    def __str__(self):
        info = ['{}: {}\n'.format(k, v) for k, v in vars(self).items()]
        return functools.reduce(lambda x, y: x + y, info)


if __name__ == '__main__':
    print(Config())
