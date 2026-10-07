"""Joint positive-view moment matching in a fixed random feature space. Fresh neural SSL evaluation tests the surrogate; the fixed-feature theorem does not guarantee it."""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image

from networks_method2 import OperatorFeatureMap
from utils_method2 import ParamDiffAug, backward_synthetic_statistic, distillation_loss, estimate_real_statistic, estimate_synthetic_statistic, evaluate_synset_SSL, fit_feature_map, get_dataset, get_eval_pool, get_network, get_time, make_diff_augmenter, step_synthetic_images


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
    parser = argparse.ArgumentParser(description='AugmentationOperatorDistillation')
    parser.add_argument('--dataset', default='CIFAR10')
    parser.add_argument('--percentage', type=float, default=1.0)
    parser.add_argument('--data_path', default='/home/mmoslem3/scratch/data')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--Iteration', type=int, default=2000)
    parser.add_argument('--lr_img', type=float, default=0.05)
    parser.add_argument('--batch_real', type=int, default=256)
    parser.add_argument('--batch_syn', type=int, default=64)
    parser.add_argument('--num_random_networks', type=int, default=10, help='number of fixed random feature maps; none of them are trained')
    parser.add_argument('--num_aug_pairs', type=int, default=4)
    parser.add_argument('--feature_dim', type=int, default=128, help='random feature dimension, capped at the synthetic image count minus one')
    parser.add_argument('--random_models', default=None, help='comma-separated architectures; defaults to --model')
    parser.add_argument('--distill_aug_strategy', default='color_crop_cutout_flip_scale_rotate')
    parser.add_argument('--distill_aug_mode', choices=['S', 'M'], default='S')

    parser.add_argument('--model', default='ConvNet')
    parser.add_argument('--lr_net', type=float, default=0.01)
    parser.add_argument('--batch_train', type=int, default=256)
    parser.add_argument('--ssl_method', choices=['simclr', 'barlowtwins', 'barlow_twins'], default='simclr')
    parser.add_argument('--ssl_aug_strategy', default=None)
    parser.add_argument('--projection_dim', type=int, default=128)
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--barlow_lambda', type=float, default=0.005)
    parser.add_argument('--eval_mode', default='S')
    parser.add_argument('--num_eval', type=int, default=6)
    parser.add_argument('--epoch_eval_train', type=int, default=1000)
    parser.add_argument('--label_percentage', type=float, default=1.0)
    parser.add_argument('--epoch_linear_train', type=int, default=100)
    parser.add_argument('--lr_linear', type=float, default=0.1)
    parser.add_argument('--batch_linear', type=int, default=256)
    parser.add_argument('--save_path', default='result_method2')
    args = parser.parse_args()

    args.method = 'AugmentationOperatorDistillation'
    args.dsa_param = ParamDiffAug()
    args.dsa_param.aug_mode = args.distill_aug_mode
    args.dsa = False
    if args.ssl_aug_strategy is None:
        args.ssl_aug_strategy = args.distill_aug_strategy
    random_models = [name.strip() for name in (args.random_models or args.model).split(',') if name.strip()]
    set_random_seed(args.seed)

    os.makedirs(args.save_path, exist_ok=True)
    clear_directory(args.save_path)
    print('Cleared all previous files from %s' % args.save_path, flush=True)
    os.makedirs(args.data_path, exist_ok=True)
    channel, im_size, num_classes, _, mean, std, dst_train, _, testloader = get_dataset(args.dataset, args.data_path)
    num_syn = int(len(dst_train) * args.percentage / 100)
    if not 2 <= num_syn <= len(dst_train):
        raise ValueError('--percentage must select between 2 images and the full training set.')
    if min(args.num_random_networks, args.num_aug_pairs, args.feature_dim, args.batch_real, args.batch_syn) < 1 or not random_models:
        raise ValueError('The feature-map counts, augmentation count, batch sizes, and model list must be nonempty.')

    indices = np.random.RandomState(args.seed).permutation(len(dst_train))[:num_syn]
    image_syn = torch.stack([dst_train[int(index)][0] for index in indices]).float().to(args.device).detach().requires_grad_(True)
    image_initial = image_syn.detach().clone()
    data_mean = torch.tensor(mean, device=args.device).view(1, channel, 1, 1)
    data_std = torch.tensor(std, device=args.device).view(1, channel, 1, 1)
    pixel_min, pixel_max = -data_mean / data_std, (1 - data_mean) / data_std

    def to_uint8(images):
        return ((images * data_std + data_mean) * 255 + 0.5).clamp(0, 255).to(torch.uint8)

    initial_uint8 = to_uint8(image_initial)
    real_loader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_real, shuffle=False, num_workers=0, pin_memory=args.device == 'cuda', drop_last=False)
    augmenter = make_diff_augmenter(args.distill_aug_strategy, args.dsa_param)
    objective_rng = np.random.default_rng(args.seed + 10000)
    optimizer_img = torch.optim.Adam([image_syn], lr=args.lr_img, eps=1e-8)
    loss_history, gradient_history, step_history = [], [], []
    evaluation_history = []
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model) if args.num_eval else []

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

    print('Hyper-parameters: \n', args.__dict__, flush=True)
    print('Objective: relative operator norm of the joint mean, marginal moments, and positive-view moments.', flush=True)
    evaluate(0)
    feature_maps, real_statistics, feature_map_info = [], [], []
    for map_index in range(args.num_random_networks):
        model_name = random_models[map_index % len(random_models)]
        network_seed = int(objective_rng.integers(0, 2**31 - 1))
        network = get_network(model_name, channel, num_classes, im_size, seed=network_seed, device=args.device)
        network.eval()
        module = network.module if isinstance(network, nn.DataParallel) else network
        with torch.no_grad():
            input_dim = module.embed(image_initial[:1]).shape[1]
        feature_dim = min(args.feature_dim, num_syn - 1)
        feature_map = OperatorFeatureMap(network, input_dim, feature_dim).to(args.device).eval()
        print('%s preparing fixed map %d/%d: %s, seed = %d, feature dimension = %d' % (get_time(), map_index + 1, args.num_random_networks, model_name, network_seed, feature_map.feature_dim), flush=True)
        fit_feature_map(feature_map, real_loader, augmenter, args.num_aug_pairs, objective_rng, args.device)
        real_statistic = estimate_real_statistic(feature_map, real_loader, augmenter, args.num_aug_pairs, objective_rng, args.device)
        feature_maps.append(feature_map)
        real_statistics.append(real_statistic)
        feature_map_info.append({'model': model_name, 'seed': network_seed, 'feature_dim': feature_map.feature_dim, 'trained': False})
        del network, module

    print('%s distillation begins' % get_time(), flush=True)
    for iteration in range(1, args.Iteration + 1):
        optimizer_img.zero_grad(set_to_none=True)
        losses = []
        for feature_map, real_statistic in zip(feature_maps, real_statistics):
            seeds = objective_rng.integers(0, 2**31 - 1, size=((num_syn + args.batch_syn - 1) // args.batch_syn, args.num_aug_pairs, 2))
            synthetic_statistic = estimate_synthetic_statistic(feature_map, image_syn, augmenter, args.batch_syn, seeds).detach().requires_grad_(True)
            loss = distillation_loss(synthetic_statistic, real_statistic)
            statistic_gradient = torch.autograd.grad(loss / args.num_random_networks, synthetic_statistic)[0].detach()
            backward_synthetic_statistic(feature_map, image_syn, augmenter, args.batch_syn, seeds, statistic_gradient)
            losses.append(loss.item())
            del synthetic_statistic, statistic_gradient, loss

        grad_rms, step_rms = step_synthetic_images(optimizer_img, image_syn, pixel_min, pixel_max, data_std)
        with torch.no_grad():
            drift = ((image_syn - image_initial) * data_std * 255).square().mean().sqrt().item()
            changed = (to_uint8(image_syn) != initial_uint8).float().mean().item() * 100
        loss_history.append(float(np.mean(losses)))
        gradient_history.append(grad_rms)
        step_history.append(step_rms)
        print('%s iter = %05d, loss = %.8f, gradient RMS = %.3e, step RMS = %.5f/255, total drift = %.5f/255, PNG values changed = %.3f%%' % (get_time(), iteration, loss_history[-1], grad_rms, step_rms, drift, changed), flush=True)

        if iteration % 250 == 0 or iteration == args.Iteration:
            evaluate(iteration)
        if iteration % 10 == 0 or iteration == args.Iteration:
            visible = (image_syn.detach() * data_std + data_mean).clamp(0, 1).cpu()
            save_image(visible, os.path.join(args.save_path, 'vis_method2_%s_%s_%gpercent_iter%d.png' % (args.dataset, args.model, args.percentage, iteration)), nrow=int(np.ceil(np.sqrt(num_syn))))

    result_path = os.path.join(args.save_path, 'res_method2_%s_%s_%gpercent.pt' % (args.dataset, args.model, args.percentage))
    torch.save({'data': image_syn.detach().cpu(), 'method': args.method, 'iteration': args.Iteration, 'loss_history': torch.tensor(loss_history, dtype=torch.float64), 'gradient_history': torch.tensor(gradient_history, dtype=torch.float64), 'step_history': torch.tensor(step_history, dtype=torch.float64), 'evaluation_history': evaluation_history, 'seed': args.seed, 'feature_maps': feature_map_info, 'augmentation': {'strategy': args.distill_aug_strategy, 'mode': args.distill_aug_mode, 'pairs': args.num_aug_pairs}, 'mean': mean, 'std': std}, result_path)
    print('Saved synthetic data to %s' % result_path, flush=True)


if __name__ == '__main__':
    main()
