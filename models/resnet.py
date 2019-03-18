import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict


__all__ = ['resnet50', 'resnet101']


def conv1x1(inputs, outputs, stride=1):
    return nn.Conv2d(inputs, outputs, kernel_size=1, stride=stride, bias=False)


def conv3x3(inputs, outputs, stride=1):
    return nn.Conv2d(inputs, outputs, kernel_size=3, stride=stride, padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inputs, planes, stride=1, shortcut=None):
        super(Bottleneck, self).__init__()
        self.residual = nn.Sequential(
            conv1x1(inputs, planes),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            conv3x3(planes, planes, stride),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            conv1x1(planes, planes * self.expansion),
            nn.BatchNorm2d(planes * self.expansion)
        )
        self.shortcut = shortcut

    def forward(self, x):
        out = self.residual(x)
        x = x if self.shortcut is None else self.shortcut(x)
        out = out + x
        out = nn.ReLU(inplace=True)(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, classes=1000):
        super(ResNet, self).__init__()
        self.inputs = 64
        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)), # /2
            ('norm1', nn.BatchNorm2d(64)),
            ('relu1', nn.ReLU(inplace=True)),
            ('pool1', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)), # /2
            ('conv2', self.make_layers(block, 64, layers[0], stride=1)),
            ('conv3', self.make_layers(block, 128, layers[1], stride=2)), # /2
            ('conv4', self.make_layers(block, 256, layers[2], stride=2)), # /2
            ('conv5', self.make_layers(block, 512, layers[3], stride=2)), # /2
            ('pool2', nn.AdaptiveAvgPool2d((1, 1)))
        ]))

        self.classifier = nn.Linear(512 * block.expansion, classes)

    def make_layers(self, block, planes, blocks, stride=1):
        shortcut = None
        if stride != 1 or self.inputs != planes * block.expansion:
            shortcut = nn.Sequential(
                conv1x1(self.inputs, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inputs, planes, stride, shortcut))
        self.inputs = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inputs, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x


def resnet50(**kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model
