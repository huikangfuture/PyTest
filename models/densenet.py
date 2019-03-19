import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict


class DenseLayer(nn.Sequential):
    def __init__(self, inputs, growth_rate, bn_size):
        super(DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(inputs))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv1', nn.Conv2d(inputs, bn_size * growth_rate, kernel_size=1, bias=False))
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False))

    def forward(self, x):
        out = super(DenseLayer, self).features(x)
        return torch.cat((x, out), 1)


class DenseBlock(nn.Sequential):
    def __init__(self, layers, inputs, growth_rate, bn_size):
        super(DenseBlock, self).__init__()
        for i in range(layers):
            layer = DenseLayer(inputs + i * growth_rate, growth_rate, bn_size)
            self.add_module('layer{}'.format(i + 1), layer)


class Transition(nn.Sequential):
    def __init__(self, inputs, outputs):
        super(Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(inputs))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(inputs, outputs, kernel_size=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    def __init__(self, config, growth_rate=32, bn_size=4, classes=1000):
        super(DenseNet, self).__init__()
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, 2 * growth_rate, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(2 * growth_rate)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        ]))

        inputs = 64
        for i, layers in enumerate(config):
            block = DenseBlock(layers, inputs, growth_rate, bn_size)
            self.features.add_module('block{}'.format(i + 1), block)
            inputs = inputs + layers * growth_rate
            if i < len(config) - 1:
                trans = Transition(inputs, inputs // 2)
                self.features.add_module('trans{}'.format(i + 1), trans)
                inputs = inputs // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(inputs))

        # Linear layer
        self.classifier = nn.Linear(inputs, classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        out = self.classifier(out)
        return out
