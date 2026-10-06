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



def compute_cross_moment(z1, z2):
    """Return the uncentered cross moment E[z1 z2^T] from the paper equation."""
    return (z1.T @ z2) / z1.size(0)

def main():
    parser = argparse.ArgumentParser(description='Parameter Processing')

    # -------------------- Data --------------------
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--percentage', type=float, default=1.0, help='percentage of total data size')
    parser.add_argument('--data_path', type=str, default='/home/mmoslem3/scratch/data', help='dataset path')

    # -------------------- Distillation --------------------
    parser.add_argument('--Iteration', type=int, default=1000, help='training iterations')
    parser.add_argument('--lr_img', type=float, default=0.5, help='learning rate for updating synthetic images')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--num_random_nets', type=int, default=8, help='number of fixed random feature networks')
    parser.add_argument('--num_aug_pairs', type=int, default=2, help='independent augmented-view pairs per network and step')
    parser.add_argument('--seed', type=int, default=0, help='seed for initialization, sampling, and evaluation networks')

    # -------------------- Network --------------------
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')

    # -------------------- Self-Supervised Learning --------------------
    parser.add_argument('--ssl_method', type=str, default='simclr', help='simclr/barlowtwins')
    parser.add_argument('--ssl_aug_strategy', type=str, default='color_crop_cutout_flip_scale_rotate', help='augmentation strategy for SSL training')
    parser.add_argument('--projection_dim', type=int, default=128, help='projection dimension for SSL training')
    parser.add_argument('--temperature', type=float, default=0.5, help='temperature for SimCLR')
    parser.add_argument('--barlow_lambda', type=float, default=0.005, help='off-diagonal weight for Barlow Twins')
    
    # -------------------- Evaluation --------------------
    parser.add_argument('--eval_mode', type=str, default='S', help='eval_mode')
    parser.add_argument('--num_eval', type=int, default=1, help='the number of evaluating randomly initialized models')
    parser.add_argument('--eval_initial', action='store_true', help='also evaluate the unoptimized initialization')
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

    sample_generator = torch.Generator().manual_seed(args.seed + 1)

    def get_images(n):
        idx_shuffle = torch.randperm(len(images_all), generator=sample_generator)[:n].to(args.device)
        return images_all[idx_shuffle]

    ''' initialize the synthetic data '''
    image_syn = torch.randn(size=(num_syn, channel, im_size[0], im_size[1]), dtype=torch.float, requires_grad=True, device=args.device)
    image_syn.data = get_images(num_syn).detach().data
    image_syn_init = image_syn.detach().clone()
    
    diag_mean = torch.tensor(mean, device=args.device).view(1, channel, 1, 1)
    diag_std = torch.tensor(std, device=args.device).view(1, channel, 1, 1)
    image_syn_init_uint8 = ((image_syn_init * diag_std + diag_mean) * 255 + 0.5).clamp(0, 255).to(torch.uint8)
    pixel_min = (0.0 - diag_mean) / diag_std
    pixel_max = (1.0 - diag_mean) / diag_std

    def evaluate_checkpoint(iteration):
        checkpoint_accs = {}
        for model_eval in model_eval_pool:
            print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d' % (args.model, model_eval, iteration))
            accs = []
            for it_eval in range(args.num_eval):
                # Reuse the same initialization at every checkpoint so changes in
                # accuracy measure changes in the images, not luck from a new net.
                eval_seed = args.seed + 20000 + it_eval
                net_eval = get_network(model_eval, channel, num_classes, im_size, seed=eval_seed).to(args.device)
                image_syn_eval = copy.deepcopy(image_syn.detach())
                _, _, acc_test = evaluate_synset_SSL(
                    it_eval, net_eval, image_syn_eval, dst_train, testloader, args
                )
                accs.append(acc_test)
            checkpoint_accs[model_eval] = accs
            print('Evaluate %d fixed-seed %s, mean = %.4f std = %.4f\n-------------------------' % (len(accs), model_eval, np.mean(accs), np.std(accs)))
        return checkpoint_accs

    if args.eval_initial:
        evaluate_checkpoint(0)

    ''' training '''
    print('%s training begins' % get_time())

    optimizer_img = torch.optim.Adam([image_syn], lr=args.lr_img)
    dsa_strategy = args.ssl_aug_strategy
    augmentation_rng = np.random.default_rng()

    # A fixed Monte Carlo approximation of E_{phi ~ P_net}. Resampling phi on
    # every step makes the reported loss a different objective each iteration.
    random_feature_nets = []
    for net_index in range(args.num_random_nets):
        net_seed = args.seed + 10000 + net_index
        net = get_network(args.model, channel, num_classes, im_size, seed=net_seed).to(args.device)
        net.eval()
        for parameter in net.parameters():
            parameter.requires_grad_(False)
        random_feature_nets.append(net)

    for it in range(1, args.Iteration + 1):
        optimizer_img.zero_grad()
        loss_avg = 0.0

        for net in random_feature_nets:
            embed = net.module.embed if hasattr(net, 'module') else net.embed
            real_batch = get_images(args.batch_real)

            for _ in range(args.num_aug_pairs):
                # Draw fresh t1,t2, then replay exactly the same transformations
                # on T and S. A seeded DSA call represents one sampled transform;
                # the seeds themselves intentionally remain nondeterministic.
                view_seed_1 = int(augmentation_rng.integers(0, 2**31 - 1))
                view_seed_2 = int(augmentation_rng.integers(0, 2**31 - 1))

                with torch.no_grad():
                    real_aug1 = DiffAugment(real_batch, strategy=dsa_strategy, seed=view_seed_1, param=args.dsa_param)
                    real_aug2 = DiffAugment(real_batch, strategy=dsa_strategy, seed=view_seed_2, param=args.dsa_param)
                    C_real = compute_cross_moment(embed(real_aug1), embed(real_aug2))

                syn_aug1 = DiffAugment(image_syn, strategy=dsa_strategy, seed=view_seed_1, param=args.dsa_param)
                syn_aug2 = DiffAugment(image_syn, strategy=dsa_strategy, seed=view_seed_2, param=args.dsa_param)
                C_syn = compute_cross_moment(embed(syn_aug1), embed(syn_aug2))

                # This is ||Sigma_T(phi) - Sigma_S(phi)||_F^2 divided by the
                # constant number of matrix entries; the minimizer is unchanged.
                loss_cross = F.mse_loss(C_syn, C_real)
                loss = loss_cross / (args.num_random_nets * args.num_aug_pairs)
                loss.backward()
                loss_avg += loss.item()

        optimizer_img.step()
        with torch.no_grad():
            image_syn.copy_(torch.maximum(torch.minimum(image_syn, pixel_max), pixel_min))

        if it % 1 == 0:
            with torch.no_grad():
                image_syn_uint8 = ((image_syn * diag_std + diag_mean) * 255 + 0.5).clamp(0, 255).to(torch.uint8)
                print('%s iter = %05d, cross-moment loss = %.10f' % (get_time(), it, loss_avg), end='\t/\t')
                print('PNG values changed = %.5f%%' % ((image_syn_uint8 != image_syn_init_uint8).float().mean().item() * 100), flush=True)

        if it == args.Iteration:
            data_save = copy.deepcopy(image_syn.detach().cpu())
            torch.save({'data': data_save}, os.path.join(args.save_path, 'res_GraphEigenspace_%s_%s_%dpercent.pt' % (args.dataset, args.model, args.percentage)))

        if it % 250 == 0 or it == args.Iteration:
            checkpoint_accs = evaluate_checkpoint(it)
            if it == args.Iteration:
                for model_eval, accs in checkpoint_accs.items():
                    accs_all_exps[model_eval] += accs

        if it % 50 == 0 and it != 0:
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
