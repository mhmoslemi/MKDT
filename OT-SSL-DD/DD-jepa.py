import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

def get_time():
    return str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))

# ============================================================================
# 1. Dataset Loader (VICO-UoE Style)
# ============================================================================
def get_dataset(dataset, data_path):
    if dataset == 'CIFAR10':
        channel = 3
        im_size = (32, 32)
        num_classes = 10
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = datasets.CIFAR10(data_path, train=True, download=True, transform=transform)
        dst_test = datasets.CIFAR10(data_path, train=False, download=True, transform=transform)
        class_names = dst_train.classes
    elif dataset == 'CIFAR100':
        channel = 3
        im_size = (32, 32)
        num_classes = 100
        mean = [0.5071, 0.4866, 0.4409]
        std = [0.2673, 0.2564, 0.2762]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = datasets.CIFAR100(data_path, train=True, download=True, transform=transform)
        dst_test = datasets.CIFAR100(data_path, train=False, download=True, transform=transform)
        class_names = dst_train.classes
    else:
        exit('unknown dataset: %s'%dataset)
    
    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test

# ============================================================================
# 2. Network Architecture (VICO-UoE ConvNet)
# ============================================================================
class ConvNet(nn.Module):
    def __init__(self, channel, num_classes=10, net_width=128, net_depth=3, net_act='relu', net_norm='instancenorm', net_pooling='avgpooling', im_size=(32,32)):
        super(ConvNet, self).__init__()
        self.features, shape_feat = self._make_layers(channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size)
        self.out_dim = shape_feat[0] * shape_feat[1] * shape_feat[2]

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        # Using L2 normalized features makes linear regression much more stable
        return F.normalize(out, dim=-1)

    def _make_layers(self, channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size):
        layers = []
        in_channels = channel
        shape_feat = [in_channels, im_size[0], im_size[1]]
        for d in range(net_depth):
            layers.append(nn.Conv2d(in_channels, net_width, kernel_size=3, padding=1))
            if net_norm == 'batchnorm':
                layers.append(nn.BatchNorm2d(net_width))
            elif net_norm == 'instancenorm':
                layers.append(nn.InstanceNorm2d(net_width))
            
            if net_act == 'relu':
                layers.append(nn.ReLU(inplace=True))
            elif net_act == 'leakyrelu':
                layers.append(nn.LeakyReLU(negative_slope=0.01, inplace=True))

            if net_pooling == 'maxpooling':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            elif net_pooling == 'avgpooling':
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2))

            in_channels = net_width
            shape_feat[0] = net_width
            shape_feat[1] //= 2
            shape_feat[2] //= 2

        return nn.Sequential(*layers), shape_feat

# ============================================================================
# 3. Differentiable Action Augmentations (JEPA Predictive Mechanics)
# ============================================================================
def sample_actions(batch_size, scale=(0.25, 0.8), ratio=(3./4., 4./3.), device='cuda'):
    """Samples bounding boxes with random scale and aspect ratio."""
    s = torch.rand(batch_size, device=device) * (scale[1] - scale[0]) + scale[0]
    r = torch.rand(batch_size, device=device) * (ratio[1] - ratio[0]) + ratio[0]
    
    h = torch.clamp(torch.sqrt(s * r), max=1.0)
    w = torch.clamp(torch.sqrt(s / r), max=1.0)
    
    y = torch.rand(batch_size, device=device) * (1 - h)
    x = torch.rand(batch_size, device=device) * (1 - w)
    return y, x, h, w

def get_action_vec(y_u, x_u, h_u, w_u, y_v, x_v, h_v, w_v):
    """Calculates relative 4D spatial shift between crop U and crop V."""
    delta_y = (y_v - y_u) / h_u
    delta_x = (x_v - x_u) / w_u
    log_h = torch.log(h_v / h_u)
    log_w = torch.log(w_v / w_u)
    return torch.stack([delta_y, delta_x, log_h, log_w], dim=1) # [B, 4]

def encode_action(a_vec, L=4):
    """Encodes action via continuous Fourier features."""
    features = []
    for i in range(L):
        features.append(torch.sin(2**i * torch.pi * a_vec))
        features.append(torch.cos(2**i * torch.pi * a_vec))
    return torch.cat(features, dim=1) # [B, 8*L]

def diff_crop(img, y, x, h, w, out_size):
    """Differentiable bounding box cropping using affine grid_sample."""
    B, C, H, W = img.shape
    cy_ndc = (y + h / 2) * 2 - 1
    cx_ndc = (x + w / 2) * 2 - 1
    
    theta = torch.zeros(B, 2, 3, device=img.device)
    theta[:, 0, 0] = w        # scale x
    theta[:, 1, 1] = h        # scale y
    theta[:, 0, 2] = cx_ndc   # translate x
    theta[:, 1, 2] = cy_ndc   # translate y
    
    grid = F.affine_grid(theta, [B, C, out_size, out_size], align_corners=False)
    out = F.grid_sample(img, grid, align_corners=False, mode='bilinear')
    return out

# ============================================================================
# 4. Main Script
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='JEPA Action-Conditional Dataset Distillation')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--num_syn', type=int, default=100, help='number of synthetic images')
    parser.add_argument('--Iteration', type=int, default=5000, help='training iterations')
    parser.add_argument('--eval_freq', type=int, default=200, help='evaluation frequency')
    parser.add_argument('--lr_img', type=float, default=0.1, help='learning rate for updating synthetic images')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--init', type=str, default='real', help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--save_path', type=str, default='result', help='path to save results')
    
    # Custom JEPA-Distillation args
    parser.add_argument('--lambda_reg', type=float, default=1e-3, help='regularization for closed-form predictor')
    parser.add_argument('--fourier_bands', type=int, default=4, help='Number of Fourier frequency bands for action encoding')
    
    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if not os.path.exists(args.data_path):
        os.makedirs(args.data_path)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test = get_dataset(args.dataset, args.data_path)

    # Organize the real dataset
    images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
    images_all = torch.cat(images_all, dim=0)
    
    print('%s build synthetic dataset...' % get_time())
    if args.init == 'real':
        idx = np.random.permutation(len(dst_train))[:args.num_syn]
        image_syn = torch.cat([dst_train[i][0].unsqueeze(0) for i in idx], dim=0).to(args.device)
    else:
        image_syn = torch.randn(size=(args.num_syn, channel, im_size[0], im_size[1]), dtype=torch.float, device=args.device)

    image_syn.requires_grad = True
    optimizer_img = torch.optim.SGD([image_syn], lr=args.lr_img, momentum=0.5)

    print('%s training begins'%get_time())
    for it in range(args.Iteration+1):
        
        # Periodically save synthetic data image grid (like VICO-UoE's evaluate_synset)
        if it % args.eval_freq == 0:
            print('%s Evaluating synthetic dataset at iteration %04d...' % (get_time(), it))
            save_image(image_syn.detach(), os.path.join(args.save_path, f"syn_iter_{it}.png"), nrow=10, normalize=True)

        # Randomly initialize network at each step so synthetic data generalizes to all network states
        net = ConvNet(channel, num_classes, im_size=im_size).to(args.device)
        net.eval()
        for param in net.parameters():
            param.requires_grad = False
            
        # Sample Real Batch
        idx = np.random.randint(0, len(images_all), args.batch_real)
        images_real = images_all[idx].to(args.device)
        
        # ---------------------------------------------------------------------
        # Step 1: Learn Predictor W* Analytically from Synthetic Data S
        # ---------------------------------------------------------------------
        # Sample action (u, v) pairs for Synthetic set
        y_u_S, x_u_S, h_u_S, w_u_S = sample_actions(args.num_syn, device=args.device)
        y_v_S, x_v_S, h_v_S, w_v_S = sample_actions(args.num_syn, device=args.device)
        
        u_S = diff_crop(image_syn, y_u_S, x_u_S, h_u_S, w_u_S, im_size[0])
        v_S = diff_crop(image_syn, y_v_S, x_v_S, h_v_S, w_v_S, im_size[0])
        
        phi_u_S = net(u_S)
        phi_v_S = net(v_S)
        
        a_vec_S = get_action_vec(y_u_S, x_u_S, h_u_S, w_u_S, y_v_S, x_v_S, h_v_S, w_v_S)
        psi_a_S = encode_action(a_vec_S, args.fourier_bands)
        
        # Z_S context representation: concatenate features with Fourier action
        Z_S = torch.cat([phi_u_S, psi_a_S], dim=-1)
        
        # Closed-Form Linear Algebra (Zero Unrolling!)
        dim = Z_S.shape[-1]
        I = torch.eye(dim, device=args.device)
        W_star = torch.linalg.solve(Z_S.t() @ Z_S + args.lambda_reg * I, Z_S.t() @ phi_v_S)
        
        # ---------------------------------------------------------------------
        # Step 2: Meta-Objective (Apply Predictor W* to Real Data X)
        # ---------------------------------------------------------------------
        # Sample action (u, v) pairs for Real set
        y_u_X, x_u_X, h_u_X, w_u_X = sample_actions(args.batch_real, device=args.device)
        y_v_X, x_v_X, h_v_X, w_v_X = sample_actions(args.batch_real, device=args.device)
        
        u_X = diff_crop(images_real, y_u_X, x_u_X, h_u_X, w_u_X, im_size[0])
        v_X = diff_crop(images_real, y_v_X, x_v_X, h_v_X, w_v_X, im_size[0])
        
        phi_u_X = net(u_X)
        phi_v_X = net(v_X)
        
        a_vec_X = get_action_vec(y_u_X, x_u_X, h_u_X, w_u_X, y_v_X, x_v_X, h_v_X, w_v_X)
        psi_a_X = encode_action(a_vec_X, args.fourier_bands)
        
        Z_X = torch.cat([phi_u_X, psi_a_X], dim=-1)
        
        # Evaluate how well the synthetic predictor maps the real context to the real target
        pred_v_X = Z_X @ W_star
        loss = F.mse_loss(pred_v_X, phi_v_X)
        
        # ---------------------------------------------------------------------
        # Step 3: Update Synthetic Pixels
        # ---------------------------------------------------------------------
        optimizer_img.zero_grad()
        loss.backward()
        optimizer_img.step()

        if it % 10 == 0:
            print('%s iter = %05d, meta-loss = %.5f' % (get_time(), it, loss.item()))

if __name__ == '__main__':
    main()