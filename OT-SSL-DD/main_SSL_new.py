"""Distill images by matching mean and covariance of fixed order-2 scattering.

Additional dependency: python -m pip install kymatio==0.3.0
SSL networks are trained only for downstream evaluation, never for distillation.
"""

import argparse
import math
import os

import numpy as np
import torch
from torchvision.utils import save_image

from scattering_moments import (
    ScatteringFeatures,
    backward_scattering_moments,
    compute_real_moments,
)
from utils import get_dataset, get_network, get_eval_pool, evaluate_synset_SSL, get_time, ParamDiffAug


def main():
    parser = argparse.ArgumentParser(description='Wavelet scattering moment distillation')

    # -------------------- Data --------------------
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--percentage', type=int, default=1, help='percentage of total data size')
    parser.add_argument('--data_path', type=str, default='/home/mmoslem3/scratch/data', help='dataset path')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda' if torch.cuda.is_available() else 'cpu')

    # -------------------- Distillation --------------------
    parser.add_argument('--Iteration', type=int, default=101, help='number of synthetic pixel updates')
    parser.add_argument('--lr_img', type=float, default=0.01, help='SGD step size for synthetic pixels')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for caching full real moments')
    parser.add_argument('--batch_syn', type=int, default=256, help='synthetic scattering batch size; moments use all images')
    parser.add_argument('--scattering_J', type=int, default=3, help='log2 scattering scale (at least 2)')
    parser.add_argument('--scattering_L', type=int, default=5, help='number of wavelet orientations')
    parser.add_argument('--gamma', type=float, default=1.0, help='weight of squared mean distance; covariance weight is 1')
    parser.add_argument('--covariance_block_size', type=int, default=1024, help='rows per exact covariance-loss block')

    # -------------------- SSL evaluation --------------------
    parser.add_argument('--model', type=str, default='ConvNet', help='evaluation model')
    parser.add_argument('--lr_net', type=float, default=0.01, help='SSL evaluation learning rate')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for SSL evaluation')
    parser.add_argument('--ssl_method', type=str, default='simclr', help='simclr/barlowtwins')
    parser.add_argument('--ssl_aug_strategy', type=str, default='color_crop_cutout_flip_scale_rotate', help='augmentation strategy for SSL evaluation')
    parser.add_argument('--projection_dim', type=int, default=128, help='projection dimension for SSL evaluation')
    parser.add_argument('--temperature', type=float, default=0.5, help='temperature for SimCLR')
    parser.add_argument('--barlow_lambda', type=float, default=0.005, help='off-diagonal weight for Barlow Twins')
    parser.add_argument('--eval_mode', type=str, default='S', help='evaluation architecture pool')
    parser.add_argument('--num_eval', type=int, default=1, help='number of random evaluation networks; 0 skips evaluation')
    parser.add_argument('--epoch_eval_train', type=int, default=1000, help='SSL evaluation epochs')
    parser.add_argument('--label_percentage', type=float, default=1.0, help='percentage of labeled data for linear probing')
    parser.add_argument('--epoch_linear_train', type=int, default=100, help='epochs to train the linear probe')
    parser.add_argument('--lr_linear', type=float, default=0.1, help='learning rate for the linear probe')
    parser.add_argument('--batch_linear', type=int, default=256, help='batch size for the linear probe')

    # -------------------- Output --------------------
    parser.add_argument('--save_path', type=str, default='result', help='path to save results')

    args = parser.parse_args()
    if not 0 < args.percentage <= 100:
        parser.error('--percentage must be in (0, 100].')
    if args.Iteration < 0 or args.num_eval < 0:
        parser.error('--Iteration and --num_eval must be nonnegative.')
    if min(args.batch_real, args.batch_syn, args.covariance_block_size) < 1:
        parser.error('Scattering and covariance batch sizes must be positive.')
    if not math.isfinite(args.gamma) or args.gamma < 0:
        parser.error('--gamma must be finite and nonnegative.')
    if not math.isfinite(args.lr_img) or args.lr_img <= 0:
        parser.error('--lr_img must be finite and positive.')
    args.method = 'Scattering'
    args.dsa_param = ParamDiffAug()
    args.dsa = False

    os.makedirs(args.data_path, exist_ok=True)
    os.makedirs(args.save_path, exist_ok=True)

    channel, im_size, num_classes, _, mean, std, dst_train, _, testloader = get_dataset(args.dataset, args.data_path)
    num_syn = int(len(dst_train) * args.percentage / 100)
    if len(dst_train) < 2 or num_syn < 2:
        parser.error('Unbiased covariance requires at least two real and two synthetic images; increase --percentage.')
    scattering = ScatteringFeatures(im_size, J=args.scattering_J, L=args.scattering_L).to(args.device).eval()
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model) if args.num_eval else []
    accs_all_exps = {model: [] for model in model_eval_pool}
    print('Hyper-parameters: \n', args.__dict__)

    # Keep the real dataset on the host; only scattering batches go to the device.
    indices = np.random.permutation(len(dst_train))[:num_syn]
    image_syn = torch.stack([dst_train[int(i)][0] for i in indices]).to(args.device).detach().requires_grad_(True)
    diag_mean = torch.tensor(mean, device=args.device).view(1, channel, 1, 1)
    diag_std = torch.tensor(std, device=args.device).view(1, channel, 1, 1)

    def to_uint8(images):
        return ((images * diag_std + diag_mean) * 255 + 0.5).clamp(0, 255).to(torch.uint8)

    image_syn_init_uint8 = to_uint8(image_syn.detach())

    def evaluate(iteration, record=False):
        for model_eval in model_eval_pool:
            print('-------------------------\nEvaluation\nmodel_eval = %s, iteration = %d' % (model_eval, iteration))
            accs = []
            for it_eval in range(args.num_eval):
                net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device)
                _, _, acc_test = evaluate_synset_SSL(
                    it_eval, net_eval, image_syn.detach().clone(), dst_train, testloader, args
                )
                accs.append(acc_test)
            print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------' % (
                len(accs), model_eval, np.mean(accs), np.std(accs)
            ))
            if record:
                accs_all_exps[model_eval].extend(accs)

    # evaluate(0, record=args.Iteration == 0)

    # Phi is fixed, so the full-dataset real statistics are computed only once.
    # Inputs use the same dataset normalization as the downstream SSL evaluator.
    with torch.no_grad():
        feature_dim = scattering(image_syn[:1]).shape[1]
    covariance_mib = feature_dim ** 2 * image_syn.element_size() / 1024 ** 2
    print('%s caching real scattering moments: %d features, %.1f MiB covariance' % (
        get_time(), feature_dim, covariance_mib
    ), flush=True)
    real_loader = torch.utils.data.DataLoader(
        dst_train, batch_size=args.batch_real, shuffle=False, num_workers=0, drop_last=False
    )
    real_mean, real_covariance = compute_real_moments(scattering, real_loader, args.device)

    print('%s distillation begins' % get_time(), flush=True)
    # Plain SGD implements S_{t+1} = S_t - lr_img * grad_S L exactly.
    # optimizer_img = torch.optim.SGD([image_syn], lr=args.lr_img)
    optimizer_img = torch.optim.Adam([image_syn, ], lr=args.lr_img) # optimizer_img for synthetic data
    # optimizer_img.zero_grad()


    for it in range(1, args.Iteration + 1):
        optimizer_img.zero_grad(set_to_none=True)
        loss, mean_loss, covariance_loss = backward_scattering_moments(
            image_syn, scattering, real_mean, real_covariance,
            gamma=args.gamma, batch_size=args.batch_syn,
            covariance_block_size=args.covariance_block_size,
        )
        optimizer_img.step()
        with torch.no_grad():
            if not torch.isfinite(image_syn).all().item():
                raise FloatingPointError('Synthetic pixel update is non-finite; reduce --lr_img.')
            changed = (to_uint8(image_syn) != image_syn_init_uint8).float().mean().item() * 100
        print('%s iter = %05d, loss = %.10f, covariance = %.10f, mean = %.10f, PNG values changed = %.5f%%' % (
            get_time(), it, loss.item(), covariance_loss.item(), mean_loss.item(), changed
        ), flush=True)

        if it % 10 == 0 and it != args.Iteration:
            evaluate(it)
        if it % 10 == 0 or it == args.Iteration:
            save_name = os.path.join(args.save_path, 'vis_%s_%s_%s_%dpercent_iter%d.png' % (
                args.method, args.dataset, args.model, args.percentage, it
            ))
            image_syn_vis = (image_syn.detach() * diag_std + diag_mean).clamp(0, 1).cpu()
            save_image(image_syn_vis, save_name, nrow=int(np.ceil(np.sqrt(num_syn))))

    # Save before final evaluation so distilled images survive an evaluation failure.
    save_name = os.path.join(args.save_path, 'res_Scattering-SSL_%s_%s_%dpercent.pt' % (
        args.dataset, args.model, args.percentage
    ))
    torch.save({
        'data': image_syn.detach().cpu(),
        'method': args.method,
        'iteration': args.Iteration,
        'scattering': {'J': args.scattering_J, 'L': args.scattering_L, 'max_order': 2},
        'gamma': args.gamma,
    }, save_name)
    print('Saved synthetic data to %s' % save_name)

    # Release the dense target before training the final evaluation networks.
    del real_mean, real_covariance, scattering
    if args.Iteration > 0:
        evaluate(args.Iteration, record=True)
    for model_eval, accs in accs_all_exps.items():
        print('Scattering distillation, evaluate %d random %s, mean = %.2f%% std = %.2f%%' % (
            len(accs), model_eval, np.mean(accs) * 100, np.std(accs) * 100
        ))


if __name__ == '__main__':
    main()
