import os
import sys
import glob
import warnings
import functools


__all__ = ['Config']


class Config:
    def __init__(self, **kwargs):
        self.device = 'cpu'
        self.datasets = {
            'mnist': os.path.join('C:/Users/Hui/.torch/datasets'),
            'dogcat': os.path.join('D:/Workspace/Resources/dogs-vs-cats'),
            'cifar10': os.path.join('C:/Users/Hui/.torch/datasets/cifar-10-batches-py')
        }
        self.checkpoint = os.path.join(os.getcwd(), 'checkpoints')

        self.epochs = 10
        self.initial_epoch = 0

        self.lr = 1e-5
        self.step_lr = 10
        self.batch_size = 64

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
