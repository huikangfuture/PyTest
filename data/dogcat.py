import os
import sys
import torch
import numpy as np
from PIL import Image


class DogCat(torch.utils.data.Dataset):
    def __init__(self, root, phase='train', transform=None):
        if transform is None:
            if phase == 'train':
                self.transform = torchvision.transforms.Compose([
                    torchvision.transforms.Resize(224),
                    torchvision.transforms.CenterCrop(224),
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.ToTensor()
                ])
            else:
                self.transform = torchvision.transforms.Compose([
                    torchvision.transforms.Resize(224),
                    torchvision.transforms.CenterCrop(224),
                    torchvision.transforms.ToTensor()
                ])
        else:
            self.transform = transform

        self.root = os.path.join(root, 'test' if phase == 'test' else 'train')
        images = sorted(os.listdir(self.root), key=lambda val: int(val.split('.')[-2]))

        if phase == 'test':
            self.images = images
        elif phase == 'train':
            self.images = images[:int(len(images) * 0.7)]
        else:
            self.images = images[int(len(images) * 0.7):]

        self.phase = phase

    def __getitem__(self, index):
        image = self.images[index]

        if self.phase == 'test':
            label = int(image.split('.')[0])
        else:
            label = 1 if 'dog' in image else 0

        image = self.transform(Image.open(os.path.join(self.root, image)))
        return image, label

    def __len__(self):
        return len(self.images)
