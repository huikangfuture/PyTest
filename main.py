import os
import sys

import cv2
import torch
import torchvision

import numpy as np
import matplotlib.pyplot as plt

import data
import utils
import models


transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

config = utils.Config()
dataset = {x: data.CIFAR10(config.data['cifar10'], phase=x, transform=transform) for x in ['train', 'val']}
dataloader = {x: torch.utils.data.DataLoader(dataset[x], batch_size=config.batch_size, shuffle=True) for x in ['train', 'val']}

model = models.VGG('D')
model = model.to(config.device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.RMSprop(model.parameters(), lr=config.lr, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.step_lr, gamma=0.1)

trainer = utils.Trainer(model, criterion, optimizer)
trainer.fit(dataloader, epochs=config.epochs, initial_epoch=config.initial_epoch, device=config.device)
