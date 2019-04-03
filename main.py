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
dataloader = {x: torch.utils.data.DataLoader(dataset[x], batch_size=16, shuffle=True) for x in ['train', 'val']}

model = tv.models.densenet121(pretrained=True)
model.classifier = torch.nn.Linear(model.classifier.in_features, 2)

model = model.to(config.device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.RMSprop([
    {'params': model.features.parameters()},
    {'params': model.classifier.parameters(), 'lr': 1e-3, 'momentum': 0.9}
], lr=0, momentum=0)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

trainer = utils.Trainer(model, criterion, optimizer, scheduler)
trainer.fit(dataloader, epochs=100, initial_epoch=0, device=config.device)
