import torch.nn as nn
import torch.utils.model_zoo as model_zoo


__all__ = ['resnet50', 'resnet101']


def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, channels, stride=1, shortcut=None):
        super(Bottleneck, self).__init__()
        self.residual = nn.Sequential(
            conv1x1(in_channels, channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            conv3x3(channels, channels, stride),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
            conv1x1(channels, channels * self.expansion),
            nn.BatchNorm2d(channels * self.expansion)
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
        self.in_channels = 64
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False), # /2
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # /2
        )
        self.layer2 = self.make_layers(block, 64,  layers[0], stride=1)
        self.layer3 = self.make_layers(block, 128, layers[1], stride=2) # /2
        self.layer4 = self.make_layers(block, 256, layers[2], stride=2) # /2
        self.layer5 = self.make_layers(block, 512, layers[3], stride=2) # /2
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layers(self, block, channels, blocks, stride=1):
        shortcut = None
        if stride != 1 or self.in_channels != channels * block.expansion:
            shortcut = nn.Sequential(
                conv1x1(self.in_channels, channels * block.expansion, stride),
                nn.BatchNorm2d(channels * block.expansion)
            )

        layers = []
        layers.append(block(self.in_channels, channels, stride, shortcut))
        self.in_channels = channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet50(**kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model
