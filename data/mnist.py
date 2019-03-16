import os
import sys
import torch
import pickle
import numpy as np
from PIL import Image


class MNIST(torch.utils.data.Dataset):
    def __init__(self, root, phase='train', split=0.7, transform=None):
        self.transform = transform

        data = 'test.pt' if phase == 'test' else 'training.pt'
        images, labels = torch.load(os.path.join(root, data))

        segmnt = int(len(labels) * split)

        if phase == 'test':
            self.images, self.labels = images, labels
        elif phase == 'train':
            self.images, self.labels = images[:segmnt], labels[:segmnt]
        else:
            self.images, self.labels = images[segmnt:], labels[segmnt:]

    def __getitem__(self, index):
        image = self.images[index].numpy()
        label = self.labels[index].item()

        if self.transform is not None:
            image = self.transform(Image.fromarray(image, mode='L'))

        return image, label

    def __len__(self):
        return len(self.labels)
