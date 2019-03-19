import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict


__all__ = ['densenet121']


class DenseLayer(nn.Sequential):
    def __init__(self, inplanes, growth_rate, bn_size):
        super(DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(inplanes))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv1', nn.Conv2d(inplanes, bn_size * growth_rate, kernel_size=1, bias=False))
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False))

    def forward(self, x):
        out = super(DenseLayer, self).features(x)
        return torch.cat((x, out), 1) # size no change


class DenseBlock(nn.Sequential):
    def __init__(self, inplanes, growth_rate, bn_size, layers):
        super(DenseBlock, self).__init__()
        for i in range(layers):
            layer = DenseLayer(inplanes + i * growth_rate, growth_rate, bn_size)
            self.add_module('layer{}'.format(i + 1), layer)


class Transition(nn.Sequential):
    def __init__(self, inplanes, planes):
        super(Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(inplanes))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(inplanes, planes, kernel_size=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2)) # /2


class DenseNet(nn.Module):
    def __init__(self, config, inplanes=64, growth_rate=32, bn_size=4, classes=1000):
        super(DenseNet, self).__init__()
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, inplanes, kernel_size=7, stride=2, padding=3, bias=False)), # /2
            ('norm0', nn.BatchNorm2d(inplanes)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)) # /2
        ]))

        for i, layers in enumerate(config):
            block = DenseBlock(inplanes, growth_rate, bn_size, layers)
            self.features.add_module('block{}'.format(i + 1), block)
            inplanes = inplanes + layers * growth_rate
            if i < len(config) - 1:
                trans = Transition(inplanes, inplanes // 2)
                self.features.add_module('trans{}'.format(i + 1), trans)
                inplanes = inplanes // 2

        self.features.add_module('norm5', nn.BatchNorm2d(inplanes))

        self.classifier = nn.Linear(inplanes, classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def densenet121(**kwargs):
    model = DenseNet([6, 12, 24, 16], **kwargs)
    return model
