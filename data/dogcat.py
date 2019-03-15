import os
import sys
import glob
import torch
import numpy as np
from PIL import Image


class DogCat(torch.utils.data.Dataset):
    def __init__(self, root, phase='train', split=0.7, transform=None):
        self.transform = transform

        images = os.path.join(root, 'test' if phase == 'test' else 'train', '*.jpg')
        images = sorted(glob.glob(images), key=lambda path: int(os.path.split(path)[1].split('.')[-2]))

        segmnt = int(len(images) * split)

        if phase == 'test':
            self.images = images
        elif phase == 'train':
            self.images = images[:segmnt]
        else:
            self.images = images[segmnt:]

    def __getitem__(self, index):
        image = Image.open(self.images[index])
        label = os.path.split(image)[1].split('.')[0]

        if self.transform is not None:
            image = self.transform(image)
        else:
            image = np.array(image)

        if label.isdigit():
            label = int(label)
        else:
            label = 1 if 'dog' in label else 0

        return image, label

    def __len__(self):
        return len(self.images)
