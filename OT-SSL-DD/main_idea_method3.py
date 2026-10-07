"""Method 3: differentiate a shared synthetic prefix and two SSL branches with a common real continuation."""

import argparse
import math
import os
import random

import numpy as np
import torch
from torchvision.utils import save_image

from utils_method3 import ParamDiffAug, continuation_diagnostic, continuation_loss, evaluate_synset_SSL, get_dataset, get_eval_pool, get_network, get_time, step_synthetic_images


def clear_directory(directory):
    for root, directories, files in os.walk(directory, topdown=False):
        for filename in files:
            os.remove(os.path.join(root, filename))
        for dirname in directories:
            path = os.path.join(root, dirname)
            if os.path.islink(path):
                os.remove(path)
            else:
                os.rmdir(path)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description='Continuation SSL Dataset Distillation')
    parser.add_argument('--dataset', default='CIFAR10')
    parser.add_argument('--percentage', type=float, default=1.0)
    parser.add_argument('--data_path', default='/home/mmoslem3/scratch/data')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--Iteration', type=int, default=1000)
    parser.add_argument('--lr_img', type=float, default=0.05)
    parser.add_argument('--horizon', type=int, default=10, help='total SSL steps in each finite-horizon comparison, including the synthetic prefix')
    parser.add_argument('--batch_ssl', type=int, default=64, help='same batch size for synthetic and real steps during distillation')
    parser.add_argument('--batch_anchor', type=int, default=64)
    parser.add_argument('--num_rollouts', type=int, default=3, help='fresh independent learner initializations per image update')

    parser.add_argument('--model', default='ConvNet')
    parser.add_argument('--lr_net', type=float, default=0.01)
    parser.add_argument('--batch_train', type=int, default=256)
    parser.add_argument('--ssl_method', choices=['simclr', 'barlowtwins', 'barlow_twins'], default='simclr')
    parser.add_argument('--ssl_aug_strategy', default='color_crop_cutout_flip_scale_rotate')
    parser.add_argument('--aug_mode', choices=['S', 'M'], default='S')
    parser.add_argument('--projection_dim', type=int, default=128)
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--barlow_lambda', type=float, default=0.005)
    parser.add_argument('--eval_mode', default='S')
    parser.add_argument('--num_eval', type=int, default=3)
    parser.add_argument('--epoch_eval_train', type=int, default=1000)
    parser.add_argument('--label_percentage', type=float, default=1.0)
    parser.add_argument('--epoch_linear_train', type=int, default=100)
    parser.add_argument('--lr_linear', type=float, default=0.1)
    parser.add_argument('--batch_linear', type=int, default=256)
    parser.add_argument('--save_path', default='result_method3')
    args = parser.parse_args()

    args.method = 'ContinuationSSLDD'
    args.dsa_param = ParamDiffAug()
    args.dsa_param.aug_mode = args.aug_mode
    args.dsa = False
    # One fixed numerical scale for the whole run; it does not change the objective's minimizers.
    args.loss_scale = 10000.0
    if args.Iteration < 1 or args.horizon < 2 or args.num_rollouts < 1 or min(args.batch_ssl, args.batch_anchor, args.batch_train) < 2:
        raise ValueError('Use a positive iteration count and rollout count, horizon >= 2, and batch sizes >= 2.')
    if min(args.lr_img, args.lr_net, args.temperature, args.lr_linear) <= 0 or args.projection_dim < 1:
        raise ValueError('Learning rates, temperature, and projection dimension must be positive.')
    if args.num_eval < 0 or args.epoch_eval_train < 0 or args.epoch_linear_train < 1 or not 0 < args.label_percentage <= 100:
        raise ValueError('Use num_eval >= 0, SSL epochs >= 0, linear epochs >= 1, and a labeled percentage in (0, 100].')
    set_random_seed(args.seed)
    # Small differences between learning branches need full float32 matrix products on CUDA.
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    os.makedirs(args.save_path, exist_ok=True)
    clear_directory(args.save_path)
    print('Cleared all previous files from %s' % args.save_path, flush=True)
    os.makedirs(args.data_path, exist_ok=True)
    channel, im_size, num_classes, _, mean, std, dst_train, _, testloader = get_dataset(args.dataset, args.data_path)
    num_syn = int(len(dst_train) * args.percentage / 100)
    if not 2 <= num_syn <= len(dst_train):
        raise ValueError('--percentage must select between 2 images and the full training set.')
    args.batch_ssl = min(args.batch_ssl, num_syn)
    args.batch_anchor = min(args.batch_anchor, len(dst_train))

    indices = np.random.RandomState(args.seed).permutation(len(dst_train))[:num_syn]
    image_syn = torch.stack([dst_train[int(index)][0] for index in indices]).float().to(args.device).detach().requires_grad_(True)
    image_initial = image_syn.detach().clone()
    data_mean = torch.tensor(mean, device=args.device).view(1, channel, 1, 1)
    data_std = torch.tensor(std, device=args.device).view(1, channel, 1, 1)
    pixel_min, pixel_max = -data_mean / data_std, (1 - data_mean) / data_std

    def to_uint8(images):
        return ((images * data_std + data_mean) * 255 + 0.5).clamp(0, 255).to(torch.uint8)

    initial_uint8 = to_uint8(image_initial)
    objective_rng = np.random.default_rng(args.seed + 10000)
    optimizer_img = torch.optim.Adam([image_syn], lr=args.lr_img, eps=1e-12)
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model) if args.num_eval else []
    loss_history, gradient_history, step_history = [], [], []
    evaluation_history, diagnostic_history = [], []
    position_counts = np.zeros(args.horizon, dtype=np.int64)

    def visualize(iteration):
        visible = (image_syn.detach() * data_std + data_mean).clamp(0, 1).cpu()
        save_image(visible, os.path.join(args.save_path, 'vis_method3_%s_%s_%gpercent_iter%d.png' % (args.dataset, args.model, args.percentage, iteration)), nrow=int(np.ceil(np.sqrt(num_syn))))

    def save(iteration):
        settings = {key: value for key, value in vars(args).items() if key != 'dsa_param'}
        result = {'data': image_syn.detach().cpu(), 'method': args.method, 'iteration': iteration, 'loss_history': torch.tensor(loss_history, dtype=torch.float64), 'gradient_history': torch.tensor(gradient_history, dtype=torch.float64), 'step_history': torch.tensor(step_history, dtype=torch.float64), 'evaluation_history': evaluation_history, 'diagnostic_history': diagnostic_history, 'position_counts': position_counts.tolist(), 'initial_indices': indices.tolist(), 'settings': settings, 'mean': mean, 'std': std}
        torch.save(result, os.path.join(args.save_path, 'res_method3_%s_%s_%gpercent_iter%d.pt' % (args.dataset, args.model, args.percentage, iteration)))

    def evaluate(iteration):
        for model_eval in model_eval_pool:
            print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d' % (args.model, model_eval, iteration), flush=True)
            accuracies = []
            for eval_index in range(args.num_eval):
                eval_seed = args.seed + 20000 + eval_index
                set_random_seed(eval_seed)
                network = get_network(model_eval, channel, num_classes, im_size, seed=eval_seed, device=args.device)
                _, _, accuracy = evaluate_synset_SSL(eval_index, network, image_syn.detach().clone(), dst_train, testloader, args)
                accuracies.append(float(accuracy))
                del network
            evaluation_history.append({'iteration': iteration, 'model': model_eval, 'accuracies': accuracies})
            print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------' % (len(accuracies), model_eval, np.mean(accuracies), np.std(accuracies)), flush=True)

    def diagnose(iteration):
        losses = continuation_diagnostic(image_syn, dst_train, channel, im_size, num_classes, args)
        if not np.isfinite(losses).all():
            raise FloatingPointError('The continuation diagnostic is non-finite.')
        diagnostic_history.append({'iteration': iteration, 'losses_by_position': losses})
        print('%s continuation check: mean = %.6e, by position = %s' % (get_time(), np.mean(losses), ', '.join('%.6e' % loss for loss in losses)), flush=True)

    print('Hyper-parameters: \n', args.__dict__, flush=True)
    print('Objective: coupled continuation error on backbone kernels; gradients pass through the complete synthetic prefix and both branches.', flush=True)
    print('The short distillation horizon is a surrogate for the longer fresh SSL evaluation.', flush=True)
    diagnose(0)
    visualize(0)
    save(0)
    # evaluate(0)
    # save(0)

    for iteration in range(1, args.Iteration + 1):
        optimizer_img.zero_grad(set_to_none=True)
        optimizer_img.param_groups[0]['lr'] = args.lr_img * 0.5 * (1 + math.cos(math.pi * (iteration - 1) / args.Iteration))
        image_gradient = torch.zeros_like(image_syn)
        losses, positions = [], []
        for _ in range(args.num_rollouts):
            loss, position = continuation_loss(image_syn, dst_train, channel, im_size, num_classes, args, objective_rng)
            if not torch.isfinite(loss):
                raise FloatingPointError('The continuation loss is non-finite.')
            gradient = torch.autograd.grad(loss * args.loss_scale / args.num_rollouts, image_syn)[0]
            image_gradient.add_(gradient.detach())
            losses.append(loss.item())
            positions.append(position)
            position_counts[position] += 1
            del gradient, loss

        image_syn.grad = image_gradient
        grad_rms, step_rms = step_synthetic_images(optimizer_img, image_syn, pixel_min, pixel_max, data_std, args.loss_scale)
        with torch.no_grad():
            drift = ((image_syn - image_initial) * data_std * 255).square().mean().sqrt().item()
            changed = (to_uint8(image_syn) != initial_uint8).float().mean().item() * 100
        loss_history.append(float(np.mean(losses)))
        gradient_history.append(grad_rms)
        step_history.append(step_rms)
        print('%s iter = %05d, loss = %.6e, positions = %s, gradient RMS = %.3e, step RMS = %.5f/255, total drift = %.5f/255, PNG values changed = %.3f%%' % (get_time(), iteration, loss_history[-1], positions, grad_rms, step_rms, drift, changed), flush=True)

        if iteration % 10 == 0 or iteration == args.Iteration:
            visualize(iteration)
        if iteration % 50 == 0 or iteration == args.Iteration:
            save(iteration)
        if iteration % 250 == 0 or iteration == args.Iteration:
            diagnose(iteration)
            evaluate(iteration)
            save(iteration)

    print('Saved Method 3 images and results in %s; replacement-position counts = %s' % (args.save_path, position_counts.tolist()), flush=True)


if __name__ == '__main__':
    main()
