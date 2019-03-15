import os
import sys
import torch
import pickle
import numpy as np
from PIL import Image


class CIFAR10(torch.utils.data.Dataset):
    def __init__(self, root, phase='train', split=0.7, transform=None):
        self.transform = transform

        if phase == 'test':
            batch_list = ['test_batch']
        else:
            batch_list = ['data_batch_{}'.format(i) for i in range(1, 6)]

        images, labels = [], []
        for name in batch_list:
            path = os.path.join(root, name)
            with open(path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
            images.append(entry['data'])
            labels.extend(entry['labels'])
        images = np.vstack(images).reshape(-1, 3, 32, 32).transpose((0, 2, 3, 1))

        segmnt = int(len(labels) * split)

        if phase == 'test':
            self.images, self.labels = images, labels
        elif phase == 'train':
            self.images, self.labels = images[:segmnt], labels[:segmnt]
        else:
            self.images, self.labels = images[segmnt:], labels[segmnt:]

    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]

        if self.transform is not None:
            image = self.transform(Image.fromarray(image))

        return image, label

    def __len__(self):
        return len(self.labels)
