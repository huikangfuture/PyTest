```py
import os
import sys

import time
import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

import cv2
import torchvision


BASE_PATH = os.path.join('D:/Workspace/Projects/Tensor')
DATA_PATH = os.path.join('D:/Workspace/Resources/cats-vs-dogs')


def train(model, criterion, optimizer, scheduler, epochs, dataloader, ckpt_path, device):
    best_accuracy = 0.
    best_model_state_dict = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        print('Epoch: {}/{}'.format(epoch + 1, epochs))

        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()

            steps = len(dataloader[phase])
            total_loss, total_accuracy = 0., 0.
            for i, (inputs, labels) in enumerate(dataloader[phase]):
                since = time.time()
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, pred = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                total_loss += loss.item()
                total_accuracy += torch.sum(pred == labels).item() / labels.size(0)

                batch_loss = total_loss / (i + 1)
                batch_accuracy = total_accuracy / (i + 1)
                batch_progress = '[{:{}d}/{}] - {:.2f}s'.format(i + 1, len(str(steps)), steps, time.time() - since)
                print('{0} - {3}loss: {1:.4f} - {3}acc: {2:.4f}'.format(batch_progress, batch_loss, batch_accuracy, 'val_' if phase  == 'val' else ''))

            epoch_loss = total_loss / len(dataloader[phase])
            epoch_accuracy = total_accuracy / len(dataloader[phase])

            if phase == 'val' and ckpt_path:
                ckpt = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
                ckpt_name = 'epoch_{:0>{}d}-loss_{:.4f}-acc_{:.4f}.pth.tar'.format(epoch + 1, len(str(epochs)), epoch_loss, epoch_accuracy)
                torch.save(ckpt, os.path.join(ckpt_path, ckpt_name))

            if phase == 'val' and epoch_accuracy > best_accuracy:
                best_accuracy = epoch_accuracy
                best_model_state_dict = copy.deepcopy(model.state_dict())

    print('Best accuracy is {:.2f}%'.format(best_accuracy * 100))
    model.load_state_dict(best_model_state_dict)
    return model
```
```py
train = False

proj_root = os.path.join('D:/Workspace/Projects/Tensor')
data_root = os.path.join('D:/Workspace/Resources/cats-vs-dogs')

base = keras.applications.InceptionResNetV2(include_top=False, input_shape=(224, 224, 3), pooling='avg')
base.trainable = False
# base.summary()

model = keras.models.Sequential()
model.add(base)
model.add(keras.layers.Reshape((1, 1, -1)))
model.add(keras.layers.Conv2D(2, 1, activation='relu'))
model.add(keras.layers.Flatten())
model.add(keras.layers.Softmax())
# model.summary()

if train:
    train_generator = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        rescale=1./255
    ).flow_from_directory(
        os.path.join(data_root, 'train'),
        target_size=(224, 224),
        class_mode='categorical',
        batch_size=64
    )

    validation_generator = keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255
    ).flow_from_directory(
        os.path.join(data_root, 'validation'),
        target_size=(224, 224),
        class_mode='categorical',
        batch_size=64
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint(os.path.join(proj_root, 'models', 'cats_vs_dogs_{epoch:02d}_{val_loss:.4f}.h5'), monitor='val_loss'),
        # keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    ]

    model.compile(optimizer=keras.optimizers.Adam(lr=1e-3), loss='categorical_crossentropy', metrics=['acc'])
    model.fit_generator(train_generator, steps_per_epoch=40, epochs=99, verbose=1, callbacks=callbacks, validation_data=validation_generator, validation_steps=20)
else:
    test_generator = keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255
    ).flow_from_directory(
        os.path.join(data_root, 'test'),
        target_size=(224, 224),
        class_mode='categorical',
        batch_size=64,
        shuffle=True
    )
    (test_data, test_labels) = test_generator.next()

    image = cv2.imread(os.path.join(proj_root, 'data', 'cat.png'))
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
    image = image[..., ::-1].astype('float32') / 255

    model.load_weights(os.path.join(proj_root, 'models', 'cats_vs_dogs_07_0.0431.h5'))
    predict = model.predict(image.reshape((1,) + image.shape))
    print('This is a dog!' if np.argmax(predict, axis=1).item() else 'This is a cat!', predict)
    plt.imshow(image)
    plt.show()
```
```py
class Linear(nn.Module):
    def __init__(self):
        super(Linear, self).__init__()
        self.hidden = nn.Linear(1, 10)
        self.output = nn.Linear(10, 1)

    def forward(self, x):
        x = self.hidden(x)
        x = self.output(F.relu(x))
        return x

net = Linear()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.005)

for epoch in range(100):
    torch.manual_seed(0)
    x = torch.linspace(-1, 1, 100, requires_grad=True).view(-1, 1)
    y = x**2 + 0.1*torch.rand(x.size())

    optimizer.zero_grad()

    o = net(x)
    loss = criterion(o, y)
    loss.backward()
    optimizer.step()

    print('Epoch: {:3d} - Loss: {:.2f}'.format(epoch, loss.item()))

    plt.cla()
    plt.plot(x.data.numpy(), y.data.numpy(), 'b.')
    plt.plot(x.data.numpy(), o.data.numpy(), 'r-')
    plt.pause(0.001)
```
```py
dataset = {x: torchvision.datasets.ImageFolder(os.path.join(DATA_PATH, x), transform=transform) for x in ['train', 'val']}
dataloader = {x: torch.utils.data.DataLoader(dataset[x], batch_size=64, shuffle=True) for x in ['train', 'val']}

testset = torchvision.datasets.ImageFolder(os.path.join(DATA_PATH, 'test'), transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)
```
