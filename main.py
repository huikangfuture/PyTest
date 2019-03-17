import os
import sys

import cv2
import torch

import numpy as np
import torchvision as tv

import data
import utils
import models


transform = {
    'train': tv.transforms.Compose([
        tv.transforms.RandomResizedCrop(224),
        tv.transforms.RandomHorizontalFlip(),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': tv.transforms.Compose([
        tv.transforms.Resize(224),
        tv.transforms.CenterCrop(224),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

config = utils.Config()
dataset = {x: data.DogCat(config.data['dogcat'], phase=x, transform=transform[x]) for x in ['train', 'val']}
dataloader = {x: torch.utils.data.DataLoader(dataset[x], batch_size=32, shuffle=True) for x in ['train', 'val']}

model = models.resnet50(classes=2)
# model = tv.models.resnet18(pretrained=True)
# model.fc = torch.nn.Linear(model.fc.in_features, 2)

model = model.to(config.device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)

trainer = utils.Trainer(model, criterion, optimizer)
trainer.fit(dataloader, epochs=10, device=config.device)
