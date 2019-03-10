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


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# config = utils.Config()
# dataset = {x: data.DogCat(config.dataset, phase=x) for x in ['train', 'val']}
# dataloader = {x: torch.utils.data.DataLoader(dataset[x], batch_size=config.batch_size, shuffle=True) for x in ['train', 'val']}

# base = torchvision.models.resnet50(pretrained=True)
# for param in base.parameters():
#     param.requires_grad = False
# base.fc = torch.nn.Linear(base.fc.in_features, 2)

# model = base.to(device)
# criterion = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.RMSprop(model.parameters(), lr=config.lr, momentum=0.9)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.step_lr, gamma=0.1)

# checkpoint = torch.load(os.path.join(config.checkpoint, 'epoch_18-loss_0.0380-acc_0.9868.pth.tar'))
# model.load_state_dict(checkpoint['model'])
# optimizer.load_state_dict(checkpoint['optimizer'])
# print(checkpoint['epoch'])

# trainer = utils.Trainer(model, criterion, optimizer)
# trainer.fit(dataloader, epochs=config.epochs, initial_epoch=config.initial_epoch, device=device, checkpoint=config.checkpoint)


model_test = models.alexnet(pretrained=True)
