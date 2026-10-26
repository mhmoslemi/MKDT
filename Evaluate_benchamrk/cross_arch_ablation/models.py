"""BN ResNet depth extension, local to this ablation to preserve old run hashes."""
from torch import nn

import common
import networks
import train_ssl

original_builder = common.build_encoder


def build_encoder(model, size):
    if model not in ('ResNet34', 'ResNet50'):
        return original_builder(model, size)
    block = networks.BasicBlock if model == 'ResNet34' else networks.Bottleneck
    encoder = networks.ResNet(block, [3, 4, 6, 3], channel=3, num_classes=1, norm='batchnorm')
    dimension = encoder.classifier.in_features
    for module in encoder.modules():
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    for module in encoder.modules():
        if isinstance(module, networks.BasicBlock):
            nn.init.zeros_(module.bn2.weight)
        elif isinstance(module, networks.Bottleneck):
            nn.init.zeros_(module.bn3.weight)
    encoder.classifier = nn.Identity()
    assert dimension == (512 if model == 'ResNet34' else 2048)
    assert not any(isinstance(m, (nn.GroupNorm, nn.InstanceNorm2d)) for m in encoder.modules())
    return encoder, dimension


def install():
    common.build_encoder = build_encoder
    train_ssl.build_encoder = build_encoder
