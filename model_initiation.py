from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from get_data_iter import cutout_batch


def model_init(data_name, model_name):
    if (data_name == 'cifar10'):
        return Net_cifar10(model_name)
    elif (data_name == 'cifar100'):
        return Net_cifar100(model_name)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        bn1 = nn.BatchNorm2d(planes)
        conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        bn2 = nn.BatchNorm2d(planes)

        self.conv_bn1 = nn.Sequential(OrderedDict([('conv', conv1), ('bn', bn1)]))
        self.conv_bn2 = nn.Sequential(OrderedDict([('conv', conv2), ('bn', bn2)]))
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if stride != 1:
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(x[:, :, ::2, ::2],
                                    (0, 0, 0, 0, (planes - in_planes) // 2,
                                     planes - in_planes - (planes - in_planes) // 2), "constant", 0))
            else:
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(x[:, :, :, :],
                                    (0, 0, 0, 0, (planes - in_planes) // 2,
                                     planes - in_planes - (planes - in_planes) // 2), "constant", 0))

    def forward(self, x):
        out = F.relu(self.conv_bn1(x))
        out = self.conv_bn2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, depth=20, num_classes=10, cfg=None, cutout=True):
        super(ResNet, self).__init__()
        if cfg is None:
            cfg = [16, 16, 32, 64]
        num_blocks = []
        if depth == 20:
            num_blocks = [3, 3, 3]
        elif depth == 32:
            num_blocks = [5, 5, 5]
        elif depth == 44:
            num_blocks = [7, 7, 7]
        elif depth == 56:
            num_blocks = [9, 9, 9]
        elif depth == 110:
            num_blocks = [18, 18, 18]
        block = BasicBlock
        self.num_classes = num_classes
        self.num_blocks = num_blocks
        self.cutout = cutout
        self.cfg = cfg
        self.in_planes = 16
        conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        bn1 = nn.BatchNorm2d(16)
        self.conv_bn = nn.Sequential(OrderedDict([('conv', conv1), ('bn', bn1)]))
        self.layer1 = self._make_layer(block, cfg[1], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, cfg[2], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, cfg[3], num_blocks[2], stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(cfg[-1], num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(len(strides)):
            layers.append(('block_%d' % i, block(self.in_planes, planes, strides[i])))
            self.in_planes = planes
        return nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        if self.training and self.cutout:
            with torch.no_grad():
                x = cutout_batch(x, 16)
        out = F.relu(self.conv_bn(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def Net_cifar10(model_name):
    if model_name == 'resnet56':
        return ResNet(depth=56, num_classes=10)
    elif model_name == 'resnet20':
        return ResNet(depth=20, num_classes=10)
    elif model_name == 'resnet32':
        return ResNet(depth=32, num_classes=10)
    elif model_name == 'resnet44':
        return ResNet(depth=44, num_classes=10)


def Net_cifar100(model_name):
    if model_name == 'resnet56':
        return ResNet(depth=56, num_classes=100)
    elif model_name == 'resnet20':
        return ResNet(depth=20, num_classes=100)
    elif model_name == 'resnet32':
        return ResNet(depth=32, num_classes=100)
    elif model_name == 'resnet44':
        return ResNet(depth=44, num_classes=100)
