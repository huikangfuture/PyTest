import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo


__all__ = ['resnet50', 'resnet101']


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, shortcut=None):
        super(Bottleneck, self).__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes * self.expansion)
        )
        self.shortcut = shortcut

    def forward(self, x):
        out = self.residual(x)
        x = x if self.shortcut is None else self.shortcut(x)
        out = out + x
        out = F.relu(out, inplace=True)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, classes=1000):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False), # /2
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1), # /2
            self.make_layers(block, 64, layers[0], stride=1),
            self.make_layers(block, 128, layers[1], stride=2), # /2
            self.make_layers(block, 256, layers[2], stride=2), # /2
            self.make_layers(block, 512, layers[3], stride=2), # /2
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(512 * block.expansion, classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layers(self, block, planes, blocks, stride=1):
        shortcut = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            shortcut = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, shortcut))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

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
