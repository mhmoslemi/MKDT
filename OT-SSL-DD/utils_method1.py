import time
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from scipy.ndimage import rotate as scipyrotate
from networks_method1 import MLP, ConvNet, LeNet, AlexNet, AlexNetBN, VGG11, VGG11BN, ResNet18, ResNet18BN_AP, ResNet18BN

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
    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)

    return net



def get_time():
    return str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))



def distance_wb(gwr, gws):
    shape = gwr.shape
    if len(shape) == 4: # conv, out*in*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2] * shape[3])
        gws = gws.reshape(shape[0], shape[1] * shape[2] * shape[3])
    elif len(shape) == 3:  # layernorm, C*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2])
        gws = gws.reshape(shape[0], shape[1] * shape[2])
    elif len(shape) == 2: # linear, out*in
        tmp = 'do nothing'
    elif len(shape) == 1: # batchnorm/instancenorm, C; groupnorm x, bias
        gwr = gwr.reshape(1, shape[0])
        gws = gws.reshape(1, shape[0])
        return torch.tensor(0, dtype=torch.float, device=gwr.device)

    dis_weight = torch.sum(1 - torch.sum(gwr * gws, dim=-1) / (torch.norm(gwr, dim=-1) * torch.norm(gws, dim=-1) + 0.000001))
    dis = dis_weight
    return dis



def match_loss(gw_syn, gw_real, args):
    dis = torch.tensor(0.0).to(args.device)

    if args.dis_metric == 'ours':
        for ig in range(len(gw_real)):
            gwr = gw_real[ig]
            gws = gw_syn[ig]
            dis += distance_wb(gwr, gws)

    elif args.dis_metric == 'mse':
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = torch.sum((gw_syn_vec - gw_real_vec)**2)

    elif args.dis_metric == 'cos':
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = 1 - torch.sum(gw_real_vec * gw_syn_vec, dim=-1) / (torch.norm(gw_real_vec, dim=-1) * torch.norm(gw_syn_vec, dim=-1) + 0.000001)

    else:
        exit('unknown distance function: %s'%args.dis_metric)

    return dis



def get_loops(ipc):
    # Get the two hyper-parameters of outer-loop and inner-loop.
    # The following values are empirically good.
    if ipc == 1:
        outer_loop, inner_loop = 1, 1
    elif ipc == 10:
        outer_loop, inner_loop = 10, 50
    elif ipc == 20:
        outer_loop, inner_loop = 20, 25
    elif ipc == 30:
        outer_loop, inner_loop = 30, 20
    elif ipc == 40:
        outer_loop, inner_loop = 40, 15
    elif ipc == 50:
        outer_loop, inner_loop = 50, 10
    else:
        outer_loop, inner_loop = 0, 0
        exit('loop hyper-parameters are not defined for %d ipc'%ipc)
    return outer_loop, inner_loop



def epoch(mode, dataloader, net, optimizer, criterion, args, aug):
    loss_avg, acc_avg, num_exp = 0, 0, 0
    net = net.to(args.device)
    criterion = criterion.to(args.device)

    if mode == 'train':
        net.train()
    else:
        net.eval()

    for i_batch, datum in enumerate(dataloader):
        img = datum[0].float().to(args.device)
        if aug:
            if args.dsa:
                img = DiffAugment(img, args.dsa_strategy, param=args.dsa_param)
            else:
                img = augment(img, args.dc_aug_param, device=args.device)
        lab = datum[1].long().to(args.device)
        n_b = lab.shape[0]

        output = net(img)
        loss = criterion(output, lab)
        acc = np.sum(np.equal(np.argmax(output.cpu().data.numpy(), axis=-1), lab.cpu().data.numpy()))

        loss_avg += loss.item()*n_b
        acc_avg += acc
        num_exp += n_b

        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    loss_avg /= num_exp
    acc_avg /= num_exp

    return loss_avg, acc_avg



def evaluate_synset(it_eval, net, images_train, labels_train, testloader, args):
    net = net.to(args.device)
    images_train = images_train.to(args.device)
    labels_train = labels_train.to(args.device)
    lr = float(args.lr_net)
    Epoch = int(args.epoch_eval_train)
    lr_schedule = [Epoch//2+1]
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    criterion = nn.CrossEntropyLoss().to(args.device)

    dst_train = TensorDataset(images_train, labels_train)
    trainloader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_train, shuffle=True, num_workers=0)

    start = time.time()
    for ep in range(Epoch+1):
        loss_train, acc_train = epoch('train', trainloader, net, optimizer, criterion, args, aug = True)
        if ep in lr_schedule:
            lr *= 0.1
            optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)

    time_train = time.time() - start
    loss_test, acc_test = epoch('test', testloader, net, optimizer, criterion, args, aug = False)
    print('%s Evaluate_%02d: epoch = %04d train time = %d s train loss = %.6f train acc = %.4f, test acc = %.4f' % (get_time(), it_eval, Epoch, int(time_train), loss_train, acc_train, acc_test))

    return net, acc_train, acc_test



def init_ssl(net, images_train, args):
    embed = net.module.embed if isinstance(net, nn.DataParallel) else net.embed
    was_training = net.training
    net.eval()
    with torch.no_grad():
        feature_dim = embed(images_train[:1].to(args.device)).shape[1]
    net.train(was_training)
    projector = nn.Sequential(nn.Linear(feature_dim, args.projection_dim), nn.ReLU(inplace=True), nn.Linear(args.projection_dim, args.projection_dim)).to(args.device)
    optimizer = torch.optim.SGD(list(net.parameters()) + list(projector.parameters()), lr=args.lr_net, momentum=0.9, weight_decay=0.0005)
    return projector, optimizer


def ssl_loss(z_1, z_2, args):
    if args.ssl_method.lower() == 'simclr':
        z = torch.cat([F.normalize(z_1, dim=1), F.normalize(z_2, dim=1)], dim=0)
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


def get_transport_plan(output_real, output_syn, args):
    with torch.no_grad():
        output_real = F.normalize(output_real, dim=1)
        output_syn = F.normalize(output_syn, dim=1)
        cost = 1 - torch.mm(output_real, output_syn.t())

        num_real = output_real.shape[0]
        num_syn = output_syn.shape[0]
        log_a = torch.full((num_real,), -np.log(num_real), device=cost.device, dtype=cost.dtype)
        log_b = torch.full((num_syn,), -np.log(num_syn), device=cost.device, dtype=cost.dtype)
        log_kernel = -cost / args.ot_lambda
        log_u = torch.zeros_like(log_a)
        log_v = torch.zeros_like(log_b)

        for t in range(args.sinkhorn_iterations):
            log_u = log_a - torch.logsumexp(log_kernel + log_v.unsqueeze(0), dim=1)
            log_v = log_b - torch.logsumexp(log_kernel + log_u.unsqueeze(1), dim=0)

        transport_plan = torch.exp(log_u.unsqueeze(1) + log_kernel + log_v.unsqueeze(0))

    return transport_plan







# def transport_contrastive_loss(output_real, output_syn, transport_plan, args):
#     output_real = F.normalize(output_real, dim=1)
#     output_syn = F.normalize(output_syn, dim=1)
#     logits = torch.mm(output_real, output_syn.t()) / args.temperature
#     log_probability = F.log_softmax(logits, dim=1)
#     loss = -torch.sum(transport_plan * log_probability)

#     return loss


def transport_barycentric_loss(output_real, output_syn, transport_plan, args):
    # OT assignments are treated as fixed targets during the synthetic update
    transport_plan = transport_plan.detach()

    # Real representations should not receive gradients
    output_real = F.normalize(output_real.detach(), dim=1)
    output_syn = F.normalize(output_syn, dim=1)

    # ---------------------------------------------------------
    # 1. OT-weighted real barycenter for each synthetic sample
    # ---------------------------------------------------------
    weights = transport_plan / (
        transport_plan.sum(dim=0, keepdim=True).clamp_min(1e-12)
    )

    # [num_syn, dim]
    barycenter_real = torch.mm(weights.t(), output_real)
    barycenter_real = F.normalize(barycenter_real, dim=1)

    # ---------------------------------------------------------
    # 2. Each synthetic representation matches ITS barycenter
    # ---------------------------------------------------------
    bary_loss = (
        1.0 - (output_syn * barycenter_real).sum(dim=1)
    ).mean()

    # ---------------------------------------------------------
    # 3. Preserve relations between barycenters
    #
    # If two real barycenters are similar, their corresponding
    # synthetic samples should also be similar.
    # If they are far apart, synthetic samples should be too.
    # ---------------------------------------------------------
    sim_syn = torch.mm(output_syn, output_syn.t())
    sim_real = torch.mm(barycenter_real, barycenter_real.t())

    # Ignore diagonal since self-similarity is always ~1
    M = output_syn.shape[0]
    mask = ~torch.eye(M, dtype=torch.bool, device=output_syn.device)

    geom_loss = ((sim_syn - sim_real) ** 2)[mask].mean()

    # ---------------------------------------------------------
    # Final distillation loss
    # ---------------------------------------------------------
    loss = bary_loss + args.geometry_weight * geom_loss

    return loss


def transport_contrastive_loss(output_real, output_syn, transport_plan, args):

    barycenter_real = torch.mm(transport_plan.t(), output_real)
    barycenter_real = barycenter_real / transport_plan.sum(dim=0).unsqueeze(1).clamp_min(1e-12)

    output_syn = F.normalize(output_syn, dim=1)
    barycenter_real = F.normalize(barycenter_real, dim=1)

    logits = torch.mm(output_syn, barycenter_real.t()) / args.temperature
    targets = torch.arange(output_syn.shape[0], device=output_syn.device)
    loss = F.cross_entropy(logits, targets)

    return loss


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


def augment(images, dc_aug_param, device):
    # This can be sped up in the future.

    if dc_aug_param != None and dc_aug_param['strategy'] != 'none':
        scale = dc_aug_param['scale']
        crop = dc_aug_param['crop']
        rotate = dc_aug_param['rotate']
        noise = dc_aug_param['noise']
        strategy = dc_aug_param['strategy']

        shape = images.shape
        mean = []
        for c in range(shape[1]):
            mean.append(float(torch.mean(images[:,c])))

        def cropfun(i):
            im_ = torch.zeros(shape[1],shape[2]+crop*2,shape[3]+crop*2, dtype=torch.float, device=device)
            for c in range(shape[1]):
                im_[c] = mean[c]
            im_[:, crop:crop+shape[2], crop:crop+shape[3]] = images[i]
            r, c = np.random.permutation(crop*2)[0], np.random.permutation(crop*2)[0]
            images[i] = im_[:, r:r+shape[2], c:c+shape[3]]

        def scalefun(i):
            h = int((np.random.uniform(1 - scale, 1 + scale)) * shape[2])
            w = int((np.random.uniform(1 - scale, 1 + scale)) * shape[2])
            tmp = F.interpolate(images[i:i + 1], [h, w], )[0]
            mhw = max(h, w, shape[2], shape[3])
            im_ = torch.zeros(shape[1], mhw, mhw, dtype=torch.float, device=device)
            r = int((mhw - h) / 2)
            c = int((mhw - w) / 2)
            im_[:, r:r + h, c:c + w] = tmp
            r = int((mhw - shape[2]) / 2)
            c = int((mhw - shape[3]) / 2)
            images[i] = im_[:, r:r + shape[2], c:c + shape[3]]

        def rotatefun(i):
            im_ = scipyrotate(images[i].cpu().data.numpy(), angle=np.random.randint(-rotate, rotate), axes=(-2, -1), cval=np.mean(mean))
            r = int((im_.shape[-2] - shape[-2]) / 2)
            c = int((im_.shape[-1] - shape[-1]) / 2)
            images[i] = torch.tensor(im_[:, r:r + shape[-2], c:c + shape[-1]], dtype=torch.float, device=device)

        def noisefun(i):
            images[i] = images[i] + noise * torch.randn(shape[1:], dtype=torch.float, device=device)


        augs = strategy.split('_')

        for i in range(shape[0]):
            choice = np.random.permutation(augs)[0] # randomly implement one augmentation
            if choice == 'crop':
                cropfun(i)
            elif choice == 'scale':
                scalefun(i)
            elif choice == 'rotate':
                rotatefun(i)
            elif choice == 'noise':
                noisefun(i)

    return images



def get_daparam(dataset, model, model_eval, ipc):
    # We find that augmentation doesn't always benefit the performance.
    # So we do augmentation for some of the settings.

    dc_aug_param = dict()
    dc_aug_param['crop'] = 4
    dc_aug_param['scale'] = 0.2
    dc_aug_param['rotate'] = 45
    dc_aug_param['noise'] = 0.001
    dc_aug_param['strategy'] = 'none'

    if dataset == 'MNIST':
        dc_aug_param['strategy'] = 'crop_scale_rotate'

    if model_eval in ['ConvNetBN']: # Data augmentation makes model training with Batch Norm layer easier.
        dc_aug_param['strategy'] = 'crop_noise'

    return dc_aug_param


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
        if 'BN' in model:
            print('Attention: Here I will replace BN with IN in evaluation, as the synthetic set is too small to measure BN hyper-parameters.')
        model_eval_pool = [model[:model.index('BN')]] if 'BN' in model else [model]
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
def rand_scale(x, param):
    # x>1, max scale
    # sx, sy: (0, +oo), 1: orignial size, 0.5: enlarge 2 times
    ratio = param.ratio_scale
    set_seed_DiffAug(param)
    sx = torch.rand(x.shape[0]) * (ratio - 1.0/ratio) + 1.0/ratio
    set_seed_DiffAug(param)
    sy = torch.rand(x.shape[0]) * (ratio - 1.0/ratio) + 1.0/ratio
    theta = [[[sx[i], 0,  0],
            [0,  sy[i], 0],] for i in range(x.shape[0])]
    theta = torch.tensor(theta, dtype=torch.float)
    if param.Siamese: # Siamese augmentation:
        theta[:] = theta[0]
    grid = F.affine_grid(theta, x.shape, align_corners=False).to(x.device)
    x = F.grid_sample(x, grid, align_corners=False)
    return x


def rand_rotate(x, param): # [-180, 180], 90: anticlockwise 90 degree
    ratio = param.ratio_rotate
    set_seed_DiffAug(param)
    theta = (torch.rand(x.shape[0]) - 0.5) * 2 * ratio / 180 * float(np.pi)
    theta = [[[torch.cos(theta[i]), torch.sin(-theta[i]), 0],
        [torch.sin(theta[i]), torch.cos(theta[i]),  0],]  for i in range(x.shape[0])]
    theta = torch.tensor(theta, dtype=torch.float)
    if param.Siamese: # Siamese augmentation:
        theta[:] = theta[0]
    grid = F.affine_grid(theta, x.shape, align_corners=False).to(x.device)
    x = F.grid_sample(x, grid, align_corners=False)
    return x


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




def transport_soft_assignment_loss(
    output_real,
    output_syn,
    transport_plan,
    args
):
    # OT is a fixed target during this image update
    q = transport_plan.detach()

    # Normalize features
    output_real = F.normalize(output_real.detach(), dim=1)
    output_syn = F.normalize(output_syn, dim=1)

    # ---------------------------------------------------------
    # q_ij = OT assignment distribution over synthetic images
    # for each real image i
    # ---------------------------------------------------------
    q = q / q.sum(dim=1, keepdim=True).clamp_min(1e-12)

    # ---------------------------------------------------------
    # p_ij = similarity-induced distribution
    # ---------------------------------------------------------
    logits = torch.mm(output_real, output_syn.t()) / args.temperature

    log_p = F.log_softmax(logits, dim=1)

    # Match p(s_j | x_i) to OT assignment q(s_j | x_i)
    loss = -(q * log_p).sum(dim=1).mean()

    return loss



def make_diff_augmenter(strategy, param):
    devices = list(range(torch.cuda.device_count()))

    def augment_images(images, seed):
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(int(seed))
            return DiffAugment(images, strategy, seed=-1, param=param)

    return augment_images


@torch.no_grad()
def estimate_real_statistic(feature_map, loader, augmenter, num_aug_pairs, rng, device):
    total, count = None, 0
    for batch in loader:
        images = batch[0].to(device=device, dtype=torch.float32)
        for _ in range(num_aug_pairs):
            features_a = feature_map(augmenter(images, rng.integers(0, 2**31 - 1)))
            features_b = feature_map(augmenter(images, rng.integers(0, 2**31 - 1)))
            statistic = paired_statistic(features_a, features_b)
            if total is None:
                total = torch.zeros_like(statistic)
            total.add_(statistic)
            count += len(images)
    return total / count


@torch.no_grad()
def estimate_synthetic_statistic(feature_map, images, augmenter, batch_size, seeds):
    total = None
    for batch_index, start in enumerate(range(0, len(images), batch_size)):
        image_batch = images[start:start + batch_size]
        for seed_a, seed_b in seeds[batch_index]:
            features_a = feature_map(augmenter(image_batch, seed_a))
            features_b = feature_map(augmenter(image_batch, seed_b))
            statistic = paired_statistic(features_a, features_b)
            if total is None:
                total = torch.zeros_like(statistic)
            total.add_(statistic)
    return total / (len(images) * seeds.shape[1])


def backward_synthetic_statistic(feature_map, images, augmenter, batch_size, seeds, statistic_gradient):
    # Replay exactly the same views. This is the chain rule for the complete dataset statistic, with one batch graph at a time.
    denominator = len(images) * seeds.shape[1]
    for batch_index, start in enumerate(range(0, len(images), batch_size)):
        for seed_a, seed_b in seeds[batch_index]:
            image_batch = images[start:start + batch_size]
            features_a = feature_map(augmenter(image_batch, seed_a))
            features_b = feature_map(augmenter(image_batch, seed_b))
            contribution = paired_statistic(features_a, features_b)
            ((contribution * statistic_gradient).sum() / denominator).backward()


@torch.no_grad()
def step_synthetic_images(optimizer, images, pixel_min, pixel_max, std):
    gradient = images.grad
    if gradient is None or not torch.isfinite(gradient).all():
        raise FloatingPointError('The synthetic image gradient is missing or non-finite.')
    grad_rms = gradient.double().square().mean().sqrt()
    # Bounded global rescaling preserves the gradient direction and keeps Adam above its numerical floor.
    gradient.mul_((1e-3 / grad_rms.clamp_min(1e-12)).clamp(max=1e4).to(gradient.dtype))
    gradient.clamp_(-0.01, 0.01)
    previous = images.detach().clone()
    optimizer.step()
    # Limit a single update to two levels in the original 0--255 pixel scale.
    update = ((images - previous) * std).clamp(-2.0 / 255, 2.0 / 255)
    images.copy_(previous + update / std)
    images.copy_(torch.maximum(torch.minimum(images, pixel_max), pixel_min))
    step_rms = ((images - previous) * std * 255).square().mean().sqrt().item()
    return grad_rms.item(), step_rms



@torch.no_grad()
def fit_feature_map(feature_map, loader, augmenter, num_aug_pairs, rng, device):
    total = torch.zeros_like(feature_map.center, dtype=torch.float64)
    squared = torch.zeros_like(total)
    count = 0
    for batch in loader:
        images = batch[0].to(device=device, dtype=torch.float32)
        for _ in range(2 * num_aug_pairs):
            features = feature_map.raw_features(augmenter(images, rng.integers(0, 2**31 - 1))).double()
            total.add_(features.sum(dim=0))
            squared.add_(features.square().sum(dim=0))
            count += len(images)
    mean = total / count
    std = (squared / count - mean.square()).clamp_min(0).sqrt()
    floor = (std.mean() * 0.05).clamp_min(1e-6)
    feature_map.center.copy_(mean.float())
    feature_map.scale.copy_(std.clamp_min(floor).float())


def paired_statistic(features_a, features_b):
    # Independent views avoid the self-view bias from squaring a finite augmentation mean.
    features_a, features_b = features_a.double(), features_b.double()
    return (features_a.T @ features_b + features_b.T @ features_a) * 0.5


def distillation_loss(synthetic_statistic, real_statistic):
    # Relative squared MMD: no feature_dim**2 averaging that would shrink the image gradient.
    scale = real_statistic.square().sum().clamp_min(1e-12)
    return (synthetic_statistic - real_statistic).square().sum() / scale
