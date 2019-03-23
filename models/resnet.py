import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


__all__ = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']


def conv1x1(in_features, out_features, stride=1):
    return nn.Conv2d(in_features, out_features, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_features, out_features, stride=1):
    return nn.Conv2d(in_features, out_features, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_features, init_features, stride=1, shortcut=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_features, init_features, stride)
        self.norm1 = nn.BatchNorm2d(init_features)
        self.conv2 = conv3x3(init_features, init_features)
        self.norm2 = nn.BatchNorm2d(init_features)
        self.shortcut = shortcut

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = F.relu(out, inplace=True)

        out = self.conv2(out)
        out = self.norm2(out)

        x = x if self.shortcut is None else self.shortcut(x)

        out = out + x
        out = F.relu(out, inplace=True)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_features, init_features, stride=1, shortcut=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(in_features, init_features)
        self.norm1 = nn.BatchNorm2d(init_features)
        self.conv2 = conv3x3(init_features, init_features, stride)
        self.norm2 = nn.BatchNorm2d(init_features)
        self.conv3 = conv1x1(init_features, init_features * self.expansion)
        self.norm3 = nn.BatchNorm2d(init_features * self.expansion)
        self.shortcut = shortcut

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = F.relu(out, inplace=True)

        out = self.conv2(out)
        out = self.norm2(out)
        out = F.relu(out, inplace=True)

        out = self.conv3(out)
        out = self.norm3(out)

        x = x if self.shortcut is None else self.shortcut(x)

        out = out + x
        out = F.relu(out, inplace=True)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, classes=1000):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.norm1 = nn.BatchNorm2d(64)

        self.planes = 64
        self.layer1 = self.make_layers(block, 64,  layers[0])
        self.layer2 = self.make_layers(block, 128, layers[1], stride=2)
        self.layer3 = self.make_layers(block, 256, layers[2], stride=2)
        self.layer4 = self.make_layers(block, 512, layers[3], stride=2)

        self.fc = nn.Linear(512 * block.expansion, classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layers(self, block, init_features, blocks, stride=1):
        shortcut = None
        if stride != 1 or self.planes != init_features * block.expansion:
            shortcut = nn.Sequential(
                conv1x1(self.planes, init_features * block.expansion, stride),
                nn.BatchNorm2d(init_features * block.expansion)
            )

        layers = []
        layers.append(block(self.planes, init_features, stride, shortcut))
        self.planes = init_features * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.planes, init_features))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.relu(x, inplace=True)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def resnet18(**kwargs):
    model = ResNet(BasicBlock, (2, 2, 2, 2), **kwargs)
    return model


def resnet34(**kwargs):
    model = ResNet(BasicBlock, (3, 4, 6, 3), **kwargs)
    return model


def resnet50(**kwargs):
    model = ResNet(Bottleneck, (3, 4, 6, 3), **kwargs)
    return model


def resnet101(**kwargs):
    model = ResNet(Bottleneck, (3, 4, 23, 3), **kwargs)
    return model


def resnet152(**kwargs):
    model = ResNet(Bottleneck, (3, 8, 36, 3), **kwargs)
    return model
