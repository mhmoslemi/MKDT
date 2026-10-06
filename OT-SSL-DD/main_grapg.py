import os
import math
import copy
import argparse
import numpy as np
import torch
from torchvision.utils import save_image
from utils import (
    get_dataset, get_network, get_eval_pool, evaluate_synset_SSL,
    get_time, ParamDiffAug
)
from ssl_gradient_matching import (
    SSLReference, draw_view_seeds, mean_parameter_gradients,
    gradient_match_and_backward, projected_backtracking_step,
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



def main():
    parser = argparse.ArgumentParser(description='Dataset distillation by matching SSL training gradients')

    # -------------------- Data --------------------
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--percentage', type=float, default=1.0, help='percentage of total data size')
    parser.add_argument('--data_path', type=str, default='/home/mmoslem3/scratch/data', help='dataset path')

    # -------------------- Distillation --------------------
    parser.add_argument('--Iteration', type=int, default=1000, help='training iterations')
    parser.add_argument('--lr_img', type=float, default=0.5, help='maximum pixel-step RMS in 0-255 units on EVERY iteration; backtracking may reduce it')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--num_reference_nets', '--num_random_nets', dest='num_reference_nets', type=int, default=3, help='number of evolving SSL references (old flag retained as an alias)')
    parser.add_argument('--num_aug_pairs', type=int, default=2, help='view pairs averaged before matching SSL parameter gradients')
    parser.add_argument('--reference_warmup_steps', type=int, default=200, help='requested initial training-age spacing; bounded by max_steps / num_reference_nets and used only at startup')
    parser.add_argument('--reference_max_steps', type=int, default=2000, help='real SSL steps before a reference restarts from a new seeded initialization')
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
    if args.num_reference_nets < 1 or args.num_aug_pairs < 1 or args.batch_real < 2 or args.batch_train < 2:
        parser.error('reference and augmentation counts must be positive; SSL batch sizes must be at least two')
    if not math.isfinite(args.lr_img) or args.lr_img <= 0:
        parser.error('lr_img must be finite and positive')
    if args.reference_warmup_steps < 0 or args.reference_max_steps < 1:
        parser.error('reference_warmup_steps must be nonnegative and reference_max_steps positive')
    if args.ssl_method.lower() not in ('simclr', 'barlowtwins', 'barlow_twins'):
        parser.error('ssl_method must be simclr or barlowtwins')
    if not math.isfinite(args.temperature) or args.temperature <= 0 or args.projection_dim < 1:
        parser.error('temperature and projection_dim must be positive')
    if args.Iteration < 1 or not 0 < args.percentage <= 100:
        parser.error('Iteration must be positive and percentage must lie in (0, 100]')
    # Leave each initial reference a meaningful lifetime. Clamping every warmup
    # separately to max_steps - 1 made mature references restart after ONE step.
    initial_age_spacing = min(
        args.reference_warmup_steps, args.reference_max_steps // args.num_reference_nets
    )
    args.reference_initial_steps = [index * initial_age_spacing for index in range(args.num_reference_nets)]
    args.method = 'SSLGradientMatching'
    args.image_optimizer = 'projected_gradient_armijo'
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()
    args.dsa = False

    os.makedirs(args.save_path, exist_ok=True)
    clear_directory(args.save_path)
    print('Cleared all previous files from %s' % args.save_path, flush=True)
    os.makedirs(args.data_path, exist_ok=True)

    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset, args.data_path)
    num_syn = int(len(dst_train) * args.percentage / 100)
    if num_syn < 2:
        parser.error('the synthetic set needs at least two images for SSL')
    args.match_batch_size = min(args.batch_real, args.batch_train, num_syn)
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

    accs_all_exps = dict()
    for key in model_eval_pool:
        accs_all_exps[key] = []

    print('Hyper-parameters: \n', args.__dict__)

    ''' organize the real dataset '''
    images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
    images_all = torch.cat(images_all, dim=0)

    sample_generator = torch.Generator().manual_seed(args.seed + 1)
    synthetic_generator = torch.Generator().manual_seed(args.seed + 2)

    def get_images(n):
        idx_shuffle = torch.randperm(len(images_all), generator=sample_generator)[:n]
        return images_all[idx_shuffle].to(args.device)

    ''' initialize the synthetic data '''
    image_syn = get_images(num_syn).detach().clone().requires_grad_(True)
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
    trial_step = None
    augmentation_rng = np.random.default_rng()
    reference_generations = [0] * args.num_reference_nets

    def build_reference(index, warmup=0):
        seed = args.seed + 100000 + reference_generations[index] * args.num_reference_nets + index
        reference = SSLReference(args, channel, num_classes, im_size, seed)
        print(f'{get_time()} reference {index}: seed = {seed}, real SSL warmup = {warmup} steps', flush=True)
        for step in range(warmup):
            gradients, real_loss = mean_parameter_gradients(
                reference, get_images(args.match_batch_size), draw_view_seeds(augmentation_rng, 1), args
            )
            reference.advance(gradients, args)
            if (step + 1) % 100 == 0 or step + 1 == warmup:
                print(f'{get_time()} reference {index}: warmup = {step + 1}/{warmup}, SSL loss = {real_loss.item():.6f}', flush=True)
        return reference

    references = [
        build_reference(index, warmup) for index, warmup in enumerate(args.reference_initial_steps)
    ]
    print(f'{get_time()} distillation begins: matching encoder AND projector gradients', flush=True)
    print('Objective: mean_reference ||mean_view grad_SSL(syn) - mean_view grad_SSL(real)||^2 / ||mean_view grad_SSL(real)||^2', flush=True)
    print(f'Matching batches: real = synthetic = {args.match_batch_size}; reference weights stay fixed during each image update', flush=True)
    print(f'Reference initial ages: {args.reference_initial_steps}; subsequent restarts begin at age 0', flush=True)
    print(f'Pixel-step RMS limit: {args.lr_img:g}/255 on every iteration', flush=True)
    print('The before/after loss uses the same sampled targets; losses across iterations use new targets.', flush=True)

    for it in range(1, args.Iteration + 1):
        image_syn.grad = None
        averages = dict(loss=0.0, cosine=0.0, norm_ratio=0.0, real_ssl=0.0, synthetic_ssl=0.0)
        sampled_targets = []
        reference_ages = []
        for index, reference in enumerate(references):
            if reference.steps >= args.reference_max_steps:
                reference_generations[index] += 1
                # Initial staggering is a one-time cost. A recycled reference
                # starts at zero and traverses the whole training-age range.
                reference = references[index] = build_reference(index)
                trial_step = None
            reference_ages.append(reference.steps)
            indices = torch.randperm(num_syn, generator=synthetic_generator)[:args.match_batch_size].to(args.device)
            target, metrics = gradient_match_and_backward(
                reference, get_images(len(indices)), image_syn, indices,
                draw_view_seeds(augmentation_rng, args.num_aug_pairs), args,
            )
            sampled_targets.append(target)
            for key, value in metrics.items():
                averages[key] += value / args.num_reference_nets

        if image_syn.grad is None:
            image_syn.grad = torch.zeros_like(image_syn)
        if not math.isfinite(averages['loss']) or not torch.isfinite(image_syn.grad).all().item():
            raise FloatingPointError('Non-finite SSL gradient-matching loss or image gradient; update aborted')

        image_before_step = image_syn.detach().clone()
        gradient_rms = image_syn.grad.square().mean().sqrt().item()
        loss_before = averages['loss']
        loss_after, accepted_step, trials, accepted = projected_backtracking_step(
            image_syn, sampled_targets, args, pixel_min, pixel_max, diag_std, loss_before, trial_step,
        )
        # Advancing a reference sooner would invalidate the cached gradients and
        # the loss comparisons used to accept/reject the image step.
        for target in sampled_targets:
            target.reference.advance(target.gradients, args)
        sampled_targets.clear()
        if accepted:
            trial_step = min(accepted_step * (2.0 if trials == 1 else 1.0), math.sqrt(torch.finfo(image_syn.dtype).max))
        else:
            trial_step = None

        if it == 1 or it % 2 == 0 or not accepted:
            with torch.no_grad():
                pixel_step = (image_syn - image_before_step) * diag_std * 255
                pixel_drift = (image_syn - image_syn_init) * diag_std * 255
                step_rms_255 = pixel_step.square().mean().sqrt().item()
                drift_rms_255 = pixel_drift.square().mean().sqrt().item()
                image_syn_uint8 = ((image_syn * diag_std + diag_mean) * 255 + 0.5).clamp(0, 255).to(torch.uint8)
                changed_pct = (image_syn_uint8 != image_syn_init_uint8).float().mean().item() * 100
                print(
                    f'{get_time()} iter = {it:05d}, sampled_grad_match = {loss_before:.6f} -> {loss_after:.6f}, '
                    f'grad_cos_before = {averages["cosine"]:.4f}, norm_ratio_before = {averages["norm_ratio"]:.4f}, '
                    f'SSL_real/syn_before = {averages["real_ssl"]:.4f}/{averages["synthetic_ssl"]:.4f}, '
                    f'image_grad_RMS = {gradient_rms:.3e}, reference_steps = {reference_ages}, '
                    f'step_RMS_255 = {step_rms_255:.4f}, drift_RMS_255 = {drift_rms_255:.4f}, '
                    f'step_size = {accepted_step:.3e}, trials = {trials}, accepted = {accepted}, '
                    f'PNG values changed = {changed_pct:.5f}%',
                    flush=True,
                )
        del image_before_step

        if it % 50 == 0 or it == args.Iteration:
            data_save = copy.deepcopy(image_syn.detach().cpu())
            save_path = os.path.join(args.save_path, 'res_%s_%s_%s_%gpercent.pt' % (args.method, args.dataset, args.model, args.percentage))
            torch.save({'data': data_save, 'method': args.method, 'iteration': it, 'seed': args.seed}, save_path)
            print(f'{get_time()} saved synthetic data to {save_path}', flush=True)

        if it % 50 == 0 or it == args.Iteration:
            checkpoint_accs = evaluate_checkpoint(it)
            if it == args.Iteration:
                for model_eval, accs in checkpoint_accs.items():
                    accs_all_exps[model_eval] += accs

        if it % 10 == 0 and it != 0:
            ''' visualize and save '''
            save_name = os.path.join(args.save_path, 'vis_%s_%s_%s_%gpercent_iter%d.png' % (args.method, args.dataset, args.model, args.percentage, it))
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
