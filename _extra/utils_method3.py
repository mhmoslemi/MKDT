"""Method 3: shared-noise continuation matching through complete finite SSL rollouts."""

import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from networks_method3 import MLP, ConvNet, LeNet, AlexNet, AlexNetBN, VGG11, VGG11BN, ResNet18, ResNet18BN_AP, ResNet18BN, SSLModel


def get_dataset(dataset, data_path):
    if dataset == 'MNIST':
        channel = 1
        im_size = (28, 28)
        num_classes = 10
        mean = [0.1307]
        std = [0.3081]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = datasets.MNIST(data_path, train=True, download=True, transform=transform) # no augmentation
        dst_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)
        class_names = [str(c) for c in range(num_classes)]

    elif dataset == 'FashionMNIST':
        channel = 1
        im_size = (28, 28)
        num_classes = 10
        mean = [0.2861]
        std = [0.3530]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = datasets.FashionMNIST(data_path, train=True, download=True, transform=transform) # no augmentation
        dst_test = datasets.FashionMNIST(data_path, train=False, download=True, transform=transform)
        class_names = dst_train.classes

    elif dataset == 'SVHN':
        channel = 3
        im_size = (32, 32)
        num_classes = 10
        mean = [0.4377, 0.4438, 0.4728]
        std = [0.1980, 0.2010, 0.1970]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = datasets.SVHN(data_path, split='train', download=True, transform=transform)  # no augmentation
        dst_test = datasets.SVHN(data_path, split='test', download=True, transform=transform)
        class_names = [str(c) for c in range(num_classes)]

    elif dataset == 'CIFAR10':
        channel = 3
        im_size = (32, 32)
        num_classes = 10
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = datasets.CIFAR10(data_path, train=True, download=True, transform=transform) # no augmentation
        dst_test = datasets.CIFAR10(data_path, train=False, download=True, transform=transform)
        class_names = dst_train.classes

    elif dataset == 'CIFAR100':
        channel = 3
        im_size = (32, 32)
        num_classes = 100
        mean = [0.5071, 0.4866, 0.4409]
        std = [0.2673, 0.2564, 0.2762]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = datasets.CIFAR100(data_path, train=True, download=True, transform=transform) # no augmentation
        dst_test = datasets.CIFAR100(data_path, train=False, download=True, transform=transform)
        class_names = dst_train.classes

    elif dataset == 'TinyImageNet':
        channel = 3
        im_size = (64, 64)
        num_classes = 200
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        data = torch.load(os.path.join(data_path, 'tinyimagenet.pt'), map_location='cpu')

        class_names = data['classes']

        images_train = data['images_train']
        labels_train = data['labels_train']
        images_train = images_train.detach().float() / 255.0
        labels_train = labels_train.detach()
        for c in range(channel):
            images_train[:,c] = (images_train[:,c] - mean[c])/std[c]
        dst_train = TensorDataset(images_train, labels_train)  # no augmentation

        images_val = data['images_val']
        labels_val = data['labels_val']
        images_val = images_val.detach().float() / 255.0
        labels_val = labels_val.detach()

        for c in range(channel):
            images_val[:, c] = (images_val[:, c] - mean[c]) / std[c]

        dst_test = TensorDataset(images_val, labels_val)  # no augmentation

    else:
        exit('unknown dataset: %s'%dataset)


    testloader = torch.utils.data.DataLoader(dst_test, batch_size=256, shuffle=False, num_workers=0)
    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader



class TensorDataset(Dataset):
    def __init__(self, images, labels): # images: n x c x h x w tensor
        self.images = images.detach().float()
        self.labels = labels.detach()

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.images.shape[0]



def get_default_convnet_setting():
    net_width, net_depth, net_act, net_norm, net_pooling = 128, 3, 'relu', 'instancenorm', 'avgpooling'
    return net_width, net_depth, net_act, net_norm, net_pooling



def get_network(model, channel, num_classes, im_size=(32, 32), seed=None, device=None):
    if seed is None:
        seed = int(time.time() * 1000) % 100000
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    net_width, net_depth, net_act, net_norm, net_pooling = get_default_convnet_setting()

    if model == 'MLP':
        net = MLP(channel=channel, num_classes=num_classes, im_size=im_size)
    elif model == 'ConvNet':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'LeNet':
        net = LeNet(channel=channel, num_classes=num_classes)
    elif model == 'AlexNet':
        net = AlexNet(channel=channel, num_classes=num_classes)
    elif model == 'AlexNetBN':
        net = AlexNetBN(channel=channel, num_classes=num_classes)
    elif model == 'VGG11':
        net = VGG11( channel=channel, num_classes=num_classes)
    elif model == 'VGG11BN':
        net = VGG11BN(channel=channel, num_classes=num_classes)
    elif model == 'ResNet18':
        net = ResNet18(channel=channel, num_classes=num_classes)
    elif model == 'ResNet18BN_AP':
        net = ResNet18BN_AP(channel=channel, num_classes=num_classes)
    elif model == 'ResNet18BN':
        net = ResNet18BN(channel=channel, num_classes=num_classes)

    elif model == 'ConvNetD1':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=1, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetD2':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=2, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetD3':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=3, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetD4':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=4, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)

    elif model == 'ConvNetW32':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=32, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetW64':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=64, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetW128':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=128, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetW256':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=256, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)

    elif model == 'ConvNetAS':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act='sigmoid', net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetAR':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act='relu', net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetAL':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act='leakyrelu', net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetASwish':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act='swish', net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetASwishBN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act='swish', net_norm='batchnorm', net_pooling=net_pooling, im_size=im_size)

    elif model == 'ConvNetNN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm='none', net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetBN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm='batchnorm', net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetLN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm='layernorm', net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetIN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm='instancenorm', net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetGN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm='groupnorm', net_pooling=net_pooling, im_size=im_size)

    elif model == 'ConvNetNP':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling='none', im_size=im_size)
    elif model == 'ConvNetMP':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling='maxpooling', im_size=im_size)
    elif model == 'ConvNetAP':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling='avgpooling', im_size=im_size)

    else:
        net = None
        exit('unknown model: %s'%model)

    device = torch.device(device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu'))
    net = net.to(device)

    return net



def get_time():
    return str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))


def init_ssl(net, images_train, args):
    embed = net.module.embed if isinstance(net, nn.DataParallel) else net.embed
    was_training = net.training
    net.eval()
    with torch.no_grad():
        feature_dim = embed(images_train[:1].to(args.device)).shape[1]
    net.train(was_training)
    projector = nn.Sequential(nn.Linear(feature_dim, args.projection_dim), nn.ReLU(inplace=False), nn.Linear(args.projection_dim, args.projection_dim)).to(args.device)
    optimizer = torch.optim.SGD(list(net.parameters()) + list(projector.parameters()), lr=args.lr_net, momentum=0.9, weight_decay=0.0005)
    return projector, optimizer


def ssl_loss(z_1, z_2, args):
    if args.ssl_method.lower() == 'simclr':
        z = torch.cat([F.normalize(z_1, dim=1, eps=1e-8), F.normalize(z_2, dim=1, eps=1e-8)], dim=0)
        logits = z @ z.T / args.temperature
        logits = logits.masked_fill(torch.eye(len(z), dtype=torch.bool, device=z.device), -float('inf'))
        targets = (torch.arange(len(z), device=z.device) + len(z_1)) % len(z)
        return F.cross_entropy(logits, targets)
    if args.ssl_method.lower() in ['barlowtwins', 'barlow_twins']:
        z_1 = (z_1 - z_1.mean(dim=0)) * torch.rsqrt(z_1.var(dim=0, unbiased=False) + 1e-4)
        z_2 = (z_2 - z_2.mean(dim=0)) * torch.rsqrt(z_2.var(dim=0, unbiased=False) + 1e-4)
        correlation = z_1.T @ z_2 / len(z_1)
        diagonal = correlation.diagonal()
        return (diagonal - 1).square().sum() + args.barlow_lambda * (correlation - torch.diag(diagonal)).square().sum()
    raise ValueError('Unknown SSL method: %s' % args.ssl_method)


def epoch_ssl(trainloader, net, projector, optimizer, args):
    net.train()
    projector.train()
    embed = net.module.embed if isinstance(net, nn.DataParallel) else net.embed
    loss_total, count = 0.0, 0
    parameters = list(net.parameters()) + list(projector.parameters())
    for datum in trainloader:
        images = datum[0].float().to(args.device)
        if len(images) < 2:
            continue
        view_a = DiffAugment(images, args.ssl_aug_strategy, param=args.dsa_param)
        view_b = DiffAugment(images, args.ssl_aug_strategy, param=args.dsa_param)
        loss = ssl_loss(projector(embed(view_a)), projector(embed(view_b)), args)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 5.0, error_if_nonfinite=True)
        optimizer.step()
        loss_total += loss.item() * len(images)
        count += len(images)
    if count == 0:
        raise ValueError('SSL evaluation needs at least two images per training batch.')
    return loss_total / count


def evaluate_synset_SSL(it_eval, net, images_train, dst_train, testloader, args):
    net = net.to(args.device)
    images_train = images_train.detach().to(args.device)
    embed = net.module.embed if isinstance(net, nn.DataParallel) else net.embed
    net.eval()
    with torch.no_grad():
        feature_dim = embed(images_train[:1]).shape[1]
        num_classes = net(images_train[:1]).shape[1]
    projector, optimizer = init_ssl(net, images_train, args)
    dst_ssl = torch.utils.data.TensorDataset(images_train.cpu())
    trainloader_ssl = torch.utils.data.DataLoader(dst_ssl, batch_size=args.batch_train, shuffle=True, num_workers=0)
    start = time.time()
    loss_ssl = 0.0
    for ep in range(args.epoch_eval_train):
        loss_ssl = epoch_ssl(trainloader_ssl, net, projector, optimizer, args)
        if ep + 1 == max(1, args.epoch_eval_train // 2):
            for group in optimizer.param_groups:
                group['lr'] *= 0.1

    for parameter in net.parameters():
        parameter.requires_grad_(False)
    net.eval()
    num_labeled = max(1, int(len(dst_train) * args.label_percentage / 100))
    indices = np.random.RandomState(args.seed + it_eval).permutation(len(dst_train))[:num_labeled]
    trainloader_linear = torch.utils.data.DataLoader(torch.utils.data.Subset(dst_train, indices), batch_size=args.batch_linear, shuffle=True, num_workers=0)
    linear = nn.Linear(feature_dim, num_classes).to(args.device)
    optimizer_linear = torch.optim.SGD(linear.parameters(), lr=args.lr_linear, momentum=0.9, weight_decay=0.0005)

    def epoch_linear(loader, training):
        linear.train(training)
        total_loss, correct, count = 0.0, 0, 0
        for images, labels in loader:
            images, labels = images.float().to(args.device), labels.long().to(args.device)
            with torch.no_grad():
                features = embed(images)
            with torch.set_grad_enabled(training):
                predictions = linear(features)
                loss = F.cross_entropy(predictions, labels)
                if training:
                    optimizer_linear.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer_linear.step()
            total_loss += loss.item() * len(images)
            correct += (predictions.argmax(dim=1) == labels).sum().item()
            count += len(images)
        return total_loss / count, correct / count

    loss_train, acc_train = 0.0, 0.0
    for ep in range(args.epoch_linear_train):
        loss_train, acc_train = epoch_linear(trainloader_linear, True)
        if ep + 1 == max(1, args.epoch_linear_train // 2):
            for group in optimizer_linear.param_groups:
                group['lr'] *= 0.1
    _, acc_test = epoch_linear(testloader, False)
    print('%s Evaluate_SSL_%02d: method = %s ssl epoch = %04d linear epoch = %04d labeled = %.2f%% train time = %d s ssl loss = %.6f train loss = %.6f train acc = %.4f, test acc = %.4f' % (get_time(), it_eval, args.ssl_method, args.epoch_eval_train, args.epoch_linear_train, args.label_percentage, int(time.time() - start), loss_ssl, loss_train, acc_train, acc_test), flush=True)
    return net, acc_train, acc_test


def get_eval_pool(eval_mode, model, model_eval):
    if eval_mode == 'M': # multiple architectures
        model_eval_pool = ['MLP', 'ConvNet', 'LeNet', 'AlexNet', 'VGG11', 'ResNet18']
    elif eval_mode == 'B':  # multiple architectures with BatchNorm for DM experiments
        model_eval_pool = ['ConvNetBN', 'ConvNetASwishBN', 'AlexNetBN', 'VGG11BN', 'ResNet18BN']
    elif eval_mode == 'W': # ablation study on network width
        model_eval_pool = ['ConvNetW32', 'ConvNetW64', 'ConvNetW128', 'ConvNetW256']
    elif eval_mode == 'D': # ablation study on network depth
        model_eval_pool = ['ConvNetD1', 'ConvNetD2', 'ConvNetD3', 'ConvNetD4']
    elif eval_mode == 'A': # ablation study on network activation function
        model_eval_pool = ['ConvNetAS', 'ConvNetAR', 'ConvNetAL', 'ConvNetASwish']
    elif eval_mode == 'P': # ablation study on network pooling layer
        model_eval_pool = ['ConvNetNP', 'ConvNetMP', 'ConvNetAP']
    elif eval_mode == 'N': # ablation study on network normalization layer
        model_eval_pool = ['ConvNetNN', 'ConvNetBN', 'ConvNetLN', 'ConvNetIN', 'ConvNetGN']
    elif eval_mode == 'S': # itself
        model_eval_pool = [model]
    elif eval_mode == 'SS':  # itself
        model_eval_pool = [model]
    else:
        model_eval_pool = [model_eval]
    return model_eval_pool


class ParamDiffAug():
    def __init__(self):
        self.aug_mode = 'S' #'multiple or single'
        self.prob_flip = 0.5
        self.ratio_scale = 1.2
        self.ratio_rotate = 15.0
        self.ratio_crop_pad = 0.125
        self.ratio_cutout = 0.5 # the size would be 0.5x0.5
        self.brightness = 1.0
        self.saturation = 2.0
        self.contrast = 0.5


def set_seed_DiffAug(param):
    if param.latestseed == -1:
        return
    else:
        torch.random.manual_seed(param.latestseed)
        param.latestseed += 1


def DiffAugment(x, strategy='', seed=-1, param=None):
    if strategy is None or strategy.lower() in ['', 'none']:
        return x
    if param is None:
        param = ParamDiffAug()
    param.Siamese = seed != -1
    param.latestseed = seed
    policies = strategy.split('_')
    if param.aug_mode == 'M':
        for policy in policies:
            for function in AUGMENT_FNS[policy]:
                x = function(x, param)
    elif param.aug_mode == 'S' and param.Siamese:
        set_seed_DiffAug(param)
        policy = policies[torch.randint(len(policies), (1,)).item()]
        for function in AUGMENT_FNS[policy]:
            x = function(x, param)
    elif param.aug_mode == 'S':
        # Each image draws its own policy, rather than sharing a batch-level choice.
        choices = torch.randint(len(policies), (len(x),), device=x.device)
        result = x.clone()
        for index, policy in enumerate(policies):
            positions = torch.where(choices == index)[0]
            if len(positions) == 0:
                continue
            augmented = x[positions]
            for function in AUGMENT_FNS[policy]:
                augmented = function(augmented, param)
            result[positions] = augmented
        x = result
    else:
        raise ValueError('Unknown augmentation mode: %s' % param.aug_mode)
    return x.contiguous()


# We implement the following differentiable augmentation strategies based on the code provided in https://github.com/mit-han-lab/data-efficient-gans.
def affine_sample(images, theta):
    # Bilinear interpolation written as gathers and constant weights, so image double backward remains available.
    count, channels, height, width = images.shape
    grid = F.affine_grid(theta.to(device=images.device, dtype=images.dtype), images.shape, align_corners=False)
    x = ((grid[..., 0] + 1) * width - 1) * 0.5
    y = ((grid[..., 1] + 1) * height - 1) * 0.5
    x0, y0 = x.floor().long(), y.floor().long()
    dx, dy = (x - x0).unsqueeze(1), (y - y0).unsqueeze(1)
    flat = images.reshape(count, channels, height * width)

    def gather(px, py):
        valid = ((px >= 0) & (px < width) & (py >= 0) & (py < height)).unsqueeze(1)
        indices = (py.clamp(0, height - 1) * width + px.clamp(0, width - 1)).reshape(count, 1, height * width).expand(-1, channels, -1)
        return flat.gather(2, indices).reshape(count, channels, height, width) * valid

    return gather(x0, y0) * (1 - dx) * (1 - dy) + gather(x0 + 1, y0) * dx * (1 - dy) + gather(x0, y0 + 1) * (1 - dx) * dy + gather(x0 + 1, y0 + 1) * dx * dy


def rand_scale(x, param):
    set_seed_DiffAug(param)
    sx = torch.rand(len(x), device=x.device) * (param.ratio_scale - 1 / param.ratio_scale) + 1 / param.ratio_scale
    set_seed_DiffAug(param)
    sy = torch.rand(len(x), device=x.device) * (param.ratio_scale - 1 / param.ratio_scale) + 1 / param.ratio_scale
    theta = x.new_zeros(len(x), 2, 3)
    theta[:, 0, 0], theta[:, 1, 1] = sx, sy
    if param.Siamese:
        theta = theta[:1].expand(len(x), -1, -1)
    return affine_sample(x, theta)


def rand_rotate(x, param):
    set_seed_DiffAug(param)
    angles = (torch.rand(len(x), device=x.device) - 0.5) * 2 * param.ratio_rotate * np.pi / 180
    theta = x.new_zeros(len(x), 2, 3)
    theta[:, 0, 0], theta[:, 0, 1] = angles.cos(), -angles.sin()
    theta[:, 1, 0], theta[:, 1, 1] = angles.sin(), angles.cos()
    if param.Siamese:
        theta = theta[:1].expand(len(x), -1, -1)
    return affine_sample(x, theta)


def rand_flip(x, param):
    prob = param.prob_flip
    set_seed_DiffAug(param)
    randf = torch.rand(x.size(0), 1, 1, 1, device=x.device)
    if param.Siamese: # Siamese augmentation:
        randf[:] = randf[0]
    return torch.where(randf < prob, x.flip(3), x)


def rand_brightness(x, param):
    ratio = param.brightness
    set_seed_DiffAug(param)
    randb = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.Siamese:  # Siamese augmentation:
        randb[:] = randb[0]
    x = x + (randb - 0.5)*ratio
    return x


def rand_saturation(x, param):
    ratio = param.saturation
    x_mean = x.mean(dim=1, keepdim=True)
    set_seed_DiffAug(param)
    rands = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.Siamese:  # Siamese augmentation:
        rands[:] = rands[0]
    x = (x - x_mean) * (rands * ratio) + x_mean
    return x


def rand_contrast(x, param):
    ratio = param.contrast
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    set_seed_DiffAug(param)
    randc = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.Siamese:  # Siamese augmentation:
        randc[:] = randc[0]
    x = (x - x_mean) * (randc + ratio) + x_mean
    return x


def rand_crop(x, param):
    # The image is padded on its surrounding and then cropped.
    ratio = param.ratio_crop_pad
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    set_seed_DiffAug(param)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    set_seed_DiffAug(param)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
    if param.Siamese:  # Siamese augmentation:
        translation_x[:] = translation_x[0]
        translation_y[:] = translation_y[0]
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
        indexing='ij',
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
    return x


def rand_cutout(x, param):
    ratio = param.ratio_cutout
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    set_seed_DiffAug(param)
    offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
    set_seed_DiffAug(param)
    offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
    if param.Siamese:  # Siamese augmentation:
        offset_x[:] = offset_x[0]
        offset_y[:] = offset_y[0]
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
        indexing='ij',
    )
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    x = x * mask.unsqueeze(1)
    return x


AUGMENT_FNS = {
    'color': [rand_brightness, rand_saturation, rand_contrast],
    'crop': [rand_crop],
    'cutout': [rand_cutout],
    'flip': [rand_flip],
    'scale': [rand_scale],
    'rotate': [rand_rotate],
}


def sample_real_images(dataset, batch_size, rng, device):
    indices = rng.choice(len(dataset), size=batch_size, replace=False)
    return torch.stack([dataset[int(index)][0] for index in indices]).to(device=device, dtype=torch.float32)


def sample_synthetic_images(images, batch_size, rng):
    indices = torch.as_tensor(rng.choice(len(images), size=batch_size, replace=False), device=images.device, dtype=torch.long)
    return images[indices]


def make_diff_augmenter(strategy, param):
    devices = list(range(torch.cuda.device_count()))

    def augment_images(images, seed):
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(int(seed))
            return DiffAugment(images, strategy, seed=-1, param=param)

    return augment_images


def functional_ssl_step(learner, parameters, momentum, images, seeds, augmenter, args, create_graph):
    view_a, view_b = augmenter(images, seeds[0]), augmenter(images, seeds[1])
    projected = functional_call(learner, parameters, (torch.cat([view_a, view_b], dim=0),), strict=True)
    projected_a, projected_b = projected.chunk(2, dim=0)
    loss = ssl_loss(projected_a, projected_b, args)
    gradients = torch.autograd.grad(loss, tuple(parameters.values()), create_graph=create_graph)
    # Match the clipped SGD, momentum, and weight decay used by the SSL evaluator.
    norm = torch.sqrt(sum(gradient.square().sum() for gradient in gradients) + 1e-12)
    clip = (5.0 / (norm + 1e-6)).clamp(max=1.0)
    next_parameters, next_momentum = {}, {}
    for (name, parameter), gradient in zip(parameters.items(), gradients):
        velocity = 0.9 * momentum[name] + clip * gradient + 0.0005 * parameter
        value = parameter - args.lr_net * velocity
        next_parameters[name] = value if create_graph else value.detach().requires_grad_(True)
        next_momentum[name] = velocity if create_graph else velocity.detach()
    return next_parameters, next_momentum


def backbone_kernel(learner, parameters, anchors_a, anchors_b):
    features = functional_call(learner, parameters, (torch.cat([anchors_a, anchors_b], dim=0), False), strict=True).double()
    features_a, features_b = features.split([len(anchors_a), len(anchors_b)], dim=0)
    # A fixed division by feature dimension retains scale and avoids width-dependent Gram magnitudes.
    return features_a @ features_b.T / features.shape[1]


def continuation_loss(images, dataset, channel, im_size, num_classes, args, rng, create_graph=True, position=None):
    position = int(rng.integers(args.horizon)) if position is None else int(position)
    network_seed = int(rng.integers(0, 2**31 - 1))
    network = get_network(args.model, channel, num_classes, im_size, seed=network_seed, device=args.device)
    learner = SSLModel(network, images[:1].detach(), args.projection_dim).train()
    parameters = dict(learner.named_parameters())
    momentum = {name: torch.zeros_like(parameter) for name, parameter in parameters.items()}
    augmenter = make_diff_augmenter(args.ssl_aug_strategy, args.dsa_param)
    anchors_a = sample_real_images(dataset, args.batch_anchor, rng, args.device)
    anchors_b = sample_real_images(dataset, args.batch_anchor, rng, args.device)

    # The whole prefix is differentiated. Both branches inherit its parameters AND its momentum.
    for _ in range(position):
        batch = sample_synthetic_images(images, args.batch_ssl, rng)
        seeds = rng.integers(0, 2**31 - 1, size=2)
        parameters, momentum = functional_ssl_step(learner, parameters, momentum, batch, seeds, augmenter, args, create_graph)

    seeds = rng.integers(0, 2**31 - 1, size=2)
    batch_syn = sample_synthetic_images(images, args.batch_ssl, rng)
    batch_real = sample_real_images(dataset, args.batch_ssl, rng, args.device)
    parameters_syn, momentum_syn = functional_ssl_step(learner, parameters, momentum, batch_syn, seeds, augmenter, args, create_graph)
    parameters_real, momentum_real = functional_ssl_step(learner, parameters, momentum, batch_real, seeds, augmenter, args, create_graph)

    # The real continuation is identical in the two branches, including each pair of augmentation seeds.
    for _ in range(args.horizon - position - 1):
        batch_real = sample_real_images(dataset, args.batch_ssl, rng, args.device)
        seeds = rng.integers(0, 2**31 - 1, size=2)
        parameters_syn, momentum_syn = functional_ssl_step(learner, parameters_syn, momentum_syn, batch_real, seeds, augmenter, args, create_graph)
        parameters_real, momentum_real = functional_ssl_step(learner, parameters_real, momentum_real, batch_real, seeds, augmenter, args, create_graph)

    with torch.set_grad_enabled(create_graph):
        kernel_syn = backbone_kernel(learner, parameters_syn, anchors_a, anchors_b)
        kernel_real = backbone_kernel(learner, parameters_real, anchors_a, anchors_b)
        # Do not detach the real branch: it also depends on the synthetic prefix.
        loss = (kernel_syn - kernel_real).square().mean()
    return loss, position


def continuation_diagnostic(images, dataset, channel, im_size, num_classes, args):
    # Separate fixed evaluation randomness, reused across checkpoints and never used for image updates.
    rng = np.random.default_rng(args.seed + 40000)
    losses = []
    for position in range(args.horizon):
        loss, _ = continuation_loss(images.detach(), dataset, channel, im_size, num_classes, args, rng, create_graph=False, position=position)
        losses.append(float(loss))
    return losses


@torch.no_grad()
def step_synthetic_images(optimizer, images, pixel_min, pixel_max, std, loss_scale):
    if images.grad is None or not torch.isfinite(images.grad).all():
        raise FloatingPointError('The image gradient is missing or non-finite.')
    grad_rms = images.grad.double().square().mean().sqrt().item() / loss_scale
    # Only cap large gradients. No normalization that forces small gradients to keep moving the images.
    torch.nn.utils.clip_grad_norm_([images], 1.0, error_if_nonfinite=True)
    previous = images.detach().clone()
    optimizer.step()
    update = ((images - previous) * std).clamp(-0.5 / 255, 0.5 / 255)
    images.copy_(previous + update / std)
    images.copy_(torch.maximum(torch.minimum(images, pixel_max), pixel_min))
    step_rms = ((images - previous) * std * 255).square().mean().sqrt().item()
    return grad_rms, step_rms
