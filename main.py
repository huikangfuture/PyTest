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
dataloader = {x: torch.utils.data.DataLoader(dataset[x], batch_size=config.batch_size, shuffle=True) for x in ['train', 'val']}

model = tv.models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, 2)

model = model.to(config.device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.RMSprop(model.parameters(), lr=config.lr, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.step_lr, gamma=0.1)

trainer = utils.Trainer(model, criterion, optimizer)
trainer.fit(dataloader, epochs=config.epochs, initial_epoch=config.initial_epoch, device=config.device)
