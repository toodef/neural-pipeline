"""
This module created AlbUNet: U-Net with ResNet encoder. This model writed by Alexander Buslaev and spoiled by me.

This model can be constructed with 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152' encoders.

For create model just call ``resnet<number>`` method
"""

import torch
import torch.nn as nn
import math
import torch.nn.functional as F

from torch.utils import model_zoo

__all__ = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class UnetDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2)
        return self.layer(x)


class ConvBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, dec, enc):
        x = torch.cat([dec, enc], dim=1)
        return self.seq(x)


class AlbUNet(torch.nn.Module):
    def __init__(self, base_model: torch.nn.Module, num_classes: int, weights_url: str = None):
        super().__init__()
        if not hasattr(self, 'decoder_block'):
            self.decoder_block = UnetDecoderBlock
        if not hasattr(self, 'bottleneck_type'):
            self.bottleneck_type = ConvBottleneck

        if weights_url is not None:
            print("Model weights inited by url")

            pretrained_weights = model_zoo.load_url(weights_url)
            model_state_dict = base_model.state_dict()
            pretrained_weights = {k: v for k, v in pretrained_weights.items() if k in model_state_dict}
            base_model.load_state_dict(pretrained_weights)

        filters = [64, 64, 128, 256, 512]

        self.bottlenecks = nn.ModuleList([self.bottleneck_type(f * 2, f) for f in reversed(filters[:-1])])
        self.decoder_stages = nn.ModuleList([self.get_decoder(filters, idx) for idx in range(1, len(filters))])

        self.encoder_stages = nn.ModuleList([self.get_encoder(base_model, idx) for idx in range(len(filters))])

        self.last_upsample = self.decoder_block(filters[0], filters[0])
        self.final = self.make_final_classifier(filters[0], num_classes)

    def forward(self, x):
        # you better run this in debugger and see what happens.
        # initial ideas are U-Net and LinkNet. see papers or blogposts for additional information
        enc_results = []
        for stage in self.encoder_stages:
            x = stage(x)
            enc_results.append(x.clone())

        for idx, bottleneck in enumerate(self.bottlenecks):
            rev_idx = - (idx + 1)
            x = self.decoder_stages[rev_idx](x)
            x = bottleneck(x, enc_results[rev_idx - 1])

        x = self.last_upsample(x)
        f = self.final(x)
        return f

    @staticmethod
    def make_final_classifier(in_filters, num_classes):
        return nn.Sequential(
            nn.Conv2d(in_filters, num_classes, 3, padding=1)
        )

    @staticmethod
    def get_encoder(encoder, layer):
        """
        encoder layers are different sized features from different net depth
        """
        if layer == 0:
            return nn.Sequential(
                encoder.conv1,
                encoder.bn1,
                encoder.relu)
        elif layer == 1:
            return nn.Sequential(
                encoder.maxpool,
                encoder.layer1)
        elif layer == 2:
            return encoder.layer2
        elif layer == 3:
            return encoder.layer3
        elif layer == 4:
            return encoder.layer4

    def get_decoder(self, filters, layer):
        return self.decoder_block(filters[layer], filters[max(layer - 1, 0)])


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, in_channels=3):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


def resnet18(classes_num: int, in_channels: int, pretrained: bool = True):
    """
    Constructs a AlbUNet with ResNet-18 encoder.

    :param classes_num: number of classes (number of masks in output)
    :param in_channels: number of input channels
    :param pretrained: If True, returns a model with encoder pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], in_channels)
    return AlbUNet(model, classes_num, weights_url=model_urls['resnet18'] if pretrained else None)


def resnet34(classes_num: int, in_channels: int, pretrained: bool = True):
    """
    Constructs a AlbUNet with ResNet-34 encoder.

    :param classes_num: number of classes (number of masks in output)
    :param in_channels: number of input channels
    :param pretrained: If True, returns a model with encoder pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], in_channels)
    return AlbUNet(model, classes_num, weights_url=model_urls['resnet34'] if pretrained else None)


def resnet50(classes_num: int, in_channels: int, pretrained: bool = True):
    """
    Constructs a AlbUNet with ResNet-50 encoder.

    :param classes_num: number of classes (number of masks in output)
    :param in_channels: number of input channels
    :param pretrained: If True, returns a model with encoder pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], in_channels)
    return AlbUNet(model, classes_num, weights_url=model_urls['resnet50'] if pretrained else None)


def resnet101(classes_num: int, in_channels: int, pretrained: bool = True):
    """
    Constructs a AlbUNet with ResNet-101 encoder.

    :param classes_num: number of classes (number of masks in output)
    :param in_channels: number of input channels
    :param pretrained: If True, returns a model with encoder pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], in_channels)
    return AlbUNet(model, classes_num, weights_url=model_urls['resnet101'] if pretrained else None)


def resnet152(classes_num: int, in_channels: int, pretrained: bool = True):
    """
    Constructs a AlbUNet with ResNet-152 encoder.

    :param classes_num: number of classes (number of masks in output)
    :param in_channels: number of input channels
    :param pretrained: If True, returns a model with encoder pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], in_channels)
    return AlbUNet(model, classes_num, weights_url=model_urls['resnet152'] if pretrained else None)
