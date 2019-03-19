import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


__all__ = ['resnet50', 'resnet101']


class ResLayer(nn.Module):
    expansion = 4

    def __init__(self, in_features, init_features, stride=1):
        super(ResLayer, self).__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(in_features, init_features, kernel_size=1, bias=False),
            nn.BatchNorm2d(init_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(init_features, init_features, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(init_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(init_features, init_features * self.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(init_features * self.expansion)
        )

        self.shortcut = None
        if stride != 1 or in_features != init_features * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_features, init_features * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(init_features * self.expansion)
            )

    def forward(self, x):
        out = self.residual(x)
        x = x if self.shortcut is None else self.shortcut(x)
        out = out + x
        out = F.relu(out, inplace=True)
        return out


class ResBlock(nn.Sequential):
    def __init__(self, layers, in_features, init_features, stride=1):
        super(ResBlock, self).__init__()

        for i in range(layers):
            layer = ResLayer(in_features=in_features, init_features=init_features, stride=stride)
            self.add_module('layer{}'.format(i + 1), layer)
            in_features = init_features * ResLayer.expansion


class ResNet(nn.Module):
    def __init__(self, config, init_features=64, classes=1000):
        super(ResNet, self).__init__()

        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, init_features, kernel_size=7, stride=2, padding=3, bias=False)), # /2
            ('norm0', nn.BatchNorm2d(init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)) # /2
        ]))

        for i, layers in enumerate(config):
            block = ResBlock(layers=layers, in_features=init_features, init_features=64 * 2 ** i, stride=2 if i > 0 else 1)
            self.features.add_module('block{}'.format(i + 1), block)
            init_features = 64 * 2 ** i * ResLayer.expansion

        self.features.add_module('pool5', nn.AdaptiveAvgPool2d((1, 1)))
        self.classifier = nn.Linear(init_features, classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def resnet50(**kwargs):
    model = ResNet((3, 4, 6, 3), **kwargs)
    return model


def resnet101(**kwargs):
    model = ResNet((3, 4, 23, 3), **kwargs)
    return model
