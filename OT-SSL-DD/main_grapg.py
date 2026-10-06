import os
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
from utils import (
    get_dataset, get_network, get_eval_pool, evaluate_synset_SSL,
    get_time, DiffAugment, ParamDiffAug
)

def clear_directory(directory):
    """Remove everything inside directory while keeping directory itself."""
    for root, directories, files in os.walk(directory, topdown=False):
        for filename in files:
            os.remove(os.path.join(root, filename))
        for dirname in directories:
            path = os.path.join(root, dirname)
            if os.path.islink(path):
                os.remove(path)
            else:
                os.rmdir(path)



def compute_covariance(z1, z2):
    """Computes the centered cross-covariance matrix of two feature batches."""
    z1_c = z1 - z1.mean(dim=0)
    z2_c = z2 - z2.mean(dim=0)
    # Compute exact cross-covariance
    cov = (z1_c.T @ z2_c) / (z1.size(0) - 1)
    return cov

def main():
    parser = argparse.ArgumentParser(description='Parameter Processing')

    # -------------------- Data --------------------
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--percentage', type=float, default=1.0, help='percentage of total data size')
    parser.add_argument('--data_path', type=str, default='/home/mmoslem3/scratch/data', help='dataset path')

    # -------------------- Distillation --------------------
    parser.add_argument('--Iteration', type=int, default=1000, help='training iterations')
    parser.add_argument('--lr_img', type=float, default=0.01, help='learning rate for updating synthetic images')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')

    # -------------------- Network --------------------
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')

    # -------------------- Self-Supervised Learning --------------------
    parser.add_argument('--ssl_method', type=str, default='simclr', help='simclr/barlowtwins')
    parser.add_argument('--ssl_aug_strategy', type=str, default='color_crop_cutout_flip_scale_rotate', help='augmentation strategy for SSL training')
    
    # -------------------- Evaluation --------------------
    parser.add_argument('--eval_mode', type=str, default='S', help='eval_mode')
    parser.add_argument('--num_eval', type=int, default=1, help='the number of evaluating randomly initialized models')
    parser.add_argument('--epoch_eval_train', type=int, default=1000, help='epochs to train a model with synthetic data')
    parser.add_argument('--label_percentage', type=float, default=1.0, help='percentage of labeled data for linear probing')
    parser.add_argument('--epoch_linear_train', type=int, default=100, help='epochs to train the linear probe')
    parser.add_argument('--lr_linear', type=float, default=0.1, help='learning rate for the linear probe')
    parser.add_argument('--batch_linear', type=int, default=256, help='batch size for the linear probe')

    # -------------------- Output --------------------
    parser.add_argument('--save_path', type=str, default='result', help='path to save results')

    args = parser.parse_args()
    args.method = 'GraphEigenspace'
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()
    args.dsa = False

    os.makedirs(args.save_path, exist_ok=True)
    clear_directory(args.save_path)
    print('Cleared all previous files from %s' % args.save_path, flush=True)
    os.makedirs(args.data_path, exist_ok=True)

    if not os.path.exists(args.data_path):
        os.makedirs(args.data_path, exist_ok=True)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)

    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset, args.data_path)
    num_syn = int(len(dst_train) * args.percentage / 100)
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

    accs_all_exps = dict()
    for key in model_eval_pool:
        accs_all_exps[key] = []

    print('Hyper-parameters: \n', args.__dict__)

    ''' organize the real dataset '''
    images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
    images_all = torch.cat(images_all, dim=0).to(args.device)

    def get_images(n):
        idx_shuffle = torch.randperm(len(images_all))[:n]
        return images_all[idx_shuffle]

    ''' initialize the synthetic data '''
    image_syn = torch.randn(size=(num_syn, channel, im_size[0], im_size[1]), dtype=torch.float, requires_grad=True, device=args.device)
    image_syn.data = get_images(num_syn).detach().data
    image_syn_init = image_syn.detach().clone()
    
    diag_mean = torch.tensor(mean, device=args.device).view(1, channel, 1, 1)
    diag_std = torch.tensor(std, device=args.device).view(1, channel, 1, 1)
    image_syn_init_uint8 = ((image_syn_init * diag_std + diag_mean) * 255 + 0.5).clamp(0, 255).to(torch.uint8)

    ''' Evaluate synthetic data prior to training '''
    for model_eval in model_eval_pool:
        print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d' % (args.model, model_eval, 0))
        accs = []
        for it_eval in range(4):
            net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device)
            image_syn_eval = copy.deepcopy(image_syn.detach())
            _, acc_train, acc_test = evaluate_synset_SSL(it_eval, net_eval, image_syn_eval, dst_train, testloader, args)
            accs.append(acc_test)
        print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------' % (len(accs), model_eval, np.mean(accs), np.std(accs)))

    ''' training '''
    print('%s training begins' % get_time())

    optimizer_img = torch.optim.Adam([image_syn], lr=args.lr_img)
    dsa_strategy = args.ssl_aug_strategy.replace('_', ',') if '_' in args.ssl_aug_strategy else 'color,crop,cutout,flip'

    for it in range(args.Iteration + 1):
        optimizer_img.zero_grad()
        loss_avg = 0.0
        
        # We sample K random architectures per step to compute the expectation.
        num_random_nets = 4 

        for _ in range(num_random_nets):
            # 1. Sample a purely random network (NO TRAINING)
            net = get_network(args.model, channel, num_classes, im_size).to(args.device)
            net.eval()
            embed = net.module.embed if hasattr(net, 'module') else net.embed

            # 2. Sample a batch of real data
            real_batch = get_images(args.batch_real)

            # 3. Apply differentiable augmentations
            # REAL VIEWS (No gradients needed)
            with torch.no_grad():
                real_aug1 = DiffAugment(real_batch, strategy=dsa_strategy)
                real_aug2 = DiffAugment(real_batch, strategy=dsa_strategy)
                
                f_real_1 = embed(real_aug1)
                f_real_2 = embed(real_aug2)
                
                C_real_cross = compute_covariance(f_real_1, f_real_2)
                C_real_11 = compute_covariance(f_real_1, f_real_1)
                C_real_22 = compute_covariance(f_real_2, f_real_2)

            # SYNTHETIC VIEWS (Requires gradients)
            syn_aug1 = DiffAugment(image_syn, strategy=dsa_strategy)
            syn_aug2 = DiffAugment(image_syn, strategy=dsa_strategy)

            f_syn_1 = embed(syn_aug1)
            f_syn_2 = embed(syn_aug2)

            C_syn_cross = compute_covariance(f_syn_1, f_syn_2)
            C_syn_11 = compute_covariance(f_syn_1, f_syn_1)
            C_syn_22 = compute_covariance(f_syn_2, f_syn_2)

            # 4. Graph Eigenspace Matching Loss
            loss_cross = F.mse_loss(C_syn_cross, C_real_cross)
            loss_marg1 = F.mse_loss(C_syn_11, C_real_11)
            loss_marg2 = F.mse_loss(C_syn_22, C_real_22)

            loss = (loss_cross + 0.5 * (loss_marg1 + loss_marg2)) / num_random_nets
            loss.backward()
            loss_avg += loss.item() * num_random_nets

        optimizer_img.step()

        if it % 1 == 0:
            with torch.no_grad():
                image_syn_uint8 = ((image_syn * diag_std + diag_mean) * 255 + 0.5).clamp(0, 255).to(torch.uint8)
                print('%s iter = %05d, loss = %.10f' % (get_time(), it, loss_avg), end='\t/\t')
                print('PNG values changed = %.5f%%' % ((image_syn_uint8 != image_syn_init_uint8).float().mean().item() * 100), flush=True)

        if it == args.Iteration:
            data_save = copy.deepcopy(image_syn.detach().cpu())
            torch.save({'data': data_save}, os.path.join(args.save_path, 'res_GraphEigenspace_%s_%s_%dpercent.pt' % (args.dataset, args.model, args.percentage)))

        if it % 50 == 0 and it != 0:
            ''' Evaluate synthetic data '''
            for model_eval in model_eval_pool:
                print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d' % (args.model, model_eval, it))

                accs = []
                for it_eval in range(4):
                    net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device) 
                    image_syn_eval = copy.deepcopy(image_syn.detach()) 
                    _, acc_train, acc_test = evaluate_synset_SSL(it_eval, net_eval, image_syn_eval, dst_train, testloader, args)
                    accs.append(acc_test)
                print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------' % (len(accs), model_eval, np.mean(accs), np.std(accs)))

                if it == args.Iteration: 
                    accs_all_exps[model_eval] += accs

        if it % 10 == 0 and it != 0:
            ''' visualize and save '''
            save_name = os.path.join(args.save_path, 'vis_GraphEigenspace_%s_%s_%dpercent_iter%d.png' % (args.dataset, args.model, args.percentage, it))
            image_syn_vis = copy.deepcopy(image_syn.detach().cpu())
            for ch in range(channel):
                image_syn_vis[:, ch] = image_syn_vis[:, ch] * std[ch] + mean[ch]
            image_syn_vis = torch.clamp(image_syn_vis, 0.0, 1.0)
            save_image(image_syn_vis, save_name, nrow=int(np.ceil(np.sqrt(num_syn))))

    print('\n==================== Final Results ====================\n')
    for key in model_eval_pool:
        accs = accs_all_exps[key]
        if len(accs) > 0:
            print('Train on %s, evaluate %d random %s, mean  = %.2f%%  std = %.2f%%' % (args.model, len(accs), key, np.mean(accs)*100, np.std(accs)*100))

if __name__ == '__main__':
    main()