import os
import math
import copy
import argparse
import numpy as np
import torch
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
    """Return the uncentered cross moment E[z1 z2^T] in the supplied equation."""
    return (z1.T @ z2) / z1.size(0)


def augmented_cross_moment(embed, images, view_seeds, strategy, aug_param):
    seed_1, seed_2 = view_seeds
    features_1 = embed(DiffAugment(images, strategy=strategy, seed=seed_1, param=aug_param))
    features_2 = embed(DiffAugment(images, strategy=strategy, seed=seed_2, param=aug_param))
    return compute_cross_moment(features_1, features_2)


def cross_moment_loss_and_backward(
    embed, real_images, synthetic_images, view_pairs, strategy, aug_param, num_networks
):
    """Accumulate the image gradient of ||mean_a(C_syn,a - C_real,a)||_F^2 / K.

    For D_a = C_syn,a - C_real,a, the old mean_a ||D_a||_F^2 equals
    ||mean_a D_a||_F^2 + mean_a ||D_a - mean_a D_a||_F^2. Averaging the
    moments before squaring removes that extra empirical variance term.
    Finite view sampling remains a Monte Carlo approximation of the expectation.

    If D = mean_a D_a, then dL/dC_syn,a = 2 D / (K A). Compute D without
    retaining graphs, then replay each view pair and apply this exact upstream
    gradient. This is the chain rule for the sampled objective, with graph memory
    independent of A. The caller must freeze the network and put it in eval mode.
    No division by the feature matrix's d^2 entries is applied to loss or gradient.
    """
    num_pairs = len(view_pairs)
    real_mean = None
    synthetic_mean = None
    with torch.no_grad():
        for seeds in view_pairs:
            real_moment = augmented_cross_moment(embed, real_images, seeds, strategy, aug_param)
            synthetic_moment = augmented_cross_moment(embed, synthetic_images, seeds, strategy, aug_param)
            if real_mean is None:
                real_mean = real_moment
                synthetic_mean = synthetic_moment
            else:
                real_mean.add_(real_moment)
                synthetic_mean.add_(synthetic_moment)

        real_mean.div_(num_pairs)
        synthetic_mean.div_(num_pairs)
        residual = synthetic_mean - real_mean
        loss_f2 = residual.square().sum()
        target_f2 = real_mean.square().sum()
        moment_gradient = residual * (2.0 / (num_networks * num_pairs))

    # Seeds are fresh each step, but must be identical between the two passes.
    # Do not square a separate loss for each replayed pair here.
    for seeds in view_pairs:
        synthetic_moment = augmented_cross_moment(embed, synthetic_images, seeds, strategy, aug_param)
        synthetic_moment.backward(moment_gradient)

    return loss_f2, target_f2, real_mean


@torch.no_grad()
def sampled_cross_moment_loss(images, sampled_targets, strategy, aug_param, reject_above):
    """Evaluate a trial on the SAME targets and views used for its gradient.

    Targets are cached on the CPU to avoid keeping K dense matrices on the GPU.
    All loss contributions are nonnegative, so a partial sum above the acceptance
    threshold proves rejection without evaluating the remaining networks.
    """
    if reject_above < 0:
        return math.inf
    total = torch.zeros((), device=images.device)
    for embed, real_mean_cpu, view_pairs in sampled_targets:
        synthetic_mean = None
        for seeds in view_pairs:
            moment = augmented_cross_moment(embed, images, seeds, strategy, aug_param)
            if synthetic_mean is None:
                synthetic_mean = moment
            else:
                synthetic_mean.add_(moment)
        synthetic_mean.div_(len(view_pairs))
        residual = synthetic_mean - real_mean_cpu.to(images.device)
        total.add_(residual.square().sum() / len(sampled_targets))
        value = total.item()
        if not math.isfinite(value) or value > reject_above:
            return math.inf
    return total.item()


@torch.no_grad()
def projected_backtracking_step(
    images, sampled_targets, strategy, aug_param, pixel_min, pixel_max,
    loss_before, initial_step, max_trials=20,
):
    """Commit only a projected gradient step satisfying Armijo decrease.

    Set y = projection_box(x - alpha*g), d = y-x. For feasible x, projection
    gives <g,d> <= -||d||^2/alpha. Accept only if
        f(y) <= f(x) + 1e-4 * <g,d> < f(x).
    Otherwise halve alpha and retry. Trials never modify x; if no candidate
    passes, leave x unchanged. Reuse all real targets and augmentation seeds:
    resampling during the search would invalidate this comparison.

    This enforces decrease of the current sampled surrogate, not a guarantee
    about a fresh sample, the population objective, or downstream SSL accuracy.
    """
    gradient = images.grad.detach()
    original = images.detach()
    step_size = initial_step
    for trial in range(1, max_trials + 1):
        candidate = original - step_size * gradient
        candidate = torch.maximum(torch.minimum(candidate, pixel_max), pixel_min)
        displacement = candidate - original
        slope = (gradient * displacement).sum(dtype=torch.float64).item()
        if not math.isfinite(slope):
            step_size *= 0.5
            continue
        if slope >= 0:
            # No representable feasible descent step at this point/step size.
            return loss_before, 0.0, trial, False
        acceptance_bound = loss_before + 1e-4 * slope
        loss_after = sampled_cross_moment_loss(
            candidate, sampled_targets, strategy, aug_param, acceptance_bound
        )
        if math.isfinite(loss_after) and loss_after <= acceptance_bound and loss_after < loss_before:
            images.copy_(candidate)
            return loss_after, step_size, trial, True
        step_size *= 0.5
    return loss_before, 0.0, max_trials, False


def main():
    parser = argparse.ArgumentParser(description='Parameter Processing')

    # -------------------- Data --------------------
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--percentage', type=float, default=1.0, help='percentage of total data size')
    parser.add_argument('--data_path', type=str, default='/home/mmoslem3/scratch/data', help='dataset path')

    # -------------------- Distillation --------------------
    parser.add_argument('--Iteration', type=int, default=1000, help='training iterations')
    parser.add_argument('--lr_img', type=float, default=0.5, help='initial trial step for projected gradient with backtracking')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--num_random_nets', type=int, default=20, help='number of fixed random feature networks')
    parser.add_argument('--num_aug_pairs', type=int, default=10, help='view pairs averaged inside the cross-moment distance')
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
    if args.num_random_nets < 1 or args.num_aug_pairs < 1 or args.batch_real < 1:
        parser.error('num_random_nets, num_aug_pairs, and batch_real must be positive')
    if not math.isfinite(args.lr_img) or args.lr_img <= 0:
        parser.error('lr_img must be finite and positive')
    args.method = 'GraphEigenspace'
    args.image_optimizer = 'projected_gradient_armijo'
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
    print('%s training begins' % get_time())

    trial_step = args.lr_img
    dsa_strategy = args.ssl_aug_strategy
    augmentation_rng = np.random.default_rng()

    # A finite fixed-bank approximation of E_{phi ~ P_net}. Resampling networks
    # would also be a valid stochastic estimator; the fixed bank is retained here.
    random_feature_nets = []
    for net_index in range(args.num_random_nets):
        net_seed = args.seed + 10000 + net_index
        net = get_network(args.model, channel, num_classes, im_size, seed=net_seed).to(args.device)
        net.eval()
        for parameter in net.parameters():
            parameter.requires_grad_(False)
        random_feature_nets.append(net)

    print('Objective: mean_network ||mean_view C_syn - mean_view C_real||_F^2 (SUM over matrix entries)', flush=True)
    print('Image updates: projected gradient with Armijo backtracking; loss_before/after use identical samples and views', flush=True)

    for it in range(1, args.Iteration + 1):
        image_syn.grad = None
        loss_avg = torch.zeros((), device=args.device)
        target_f2_avg = torch.zeros((), device=args.device)
        sampled_targets = []

        for net in random_feature_nets:
            embed = net.module.embed if hasattr(net, 'module') else net.embed
            real_batch = get_images(args.batch_real)
            # Nondeterministic new augmentations; replay seeds only tie together
            # real/synthetic views and the moment/gradient computation passes.
            view_pairs = augmentation_rng.integers(
                0, 2**31 - 1, size=(args.num_aug_pairs, 2)
            ).tolist()
            loss_f2, target_f2, real_mean = cross_moment_loss_and_backward(
                embed, real_batch, image_syn, view_pairs, dsa_strategy,
                args.dsa_param, args.num_random_nets,
            )
            loss_avg.add_(loss_f2 / args.num_random_nets)
            target_f2_avg.add_(target_f2 / args.num_random_nets)
            sampled_targets.append((embed, real_mean.cpu(), view_pairs))

        if not torch.isfinite(loss_avg).item() or not torch.isfinite(image_syn.grad).all().item():
            raise FloatingPointError('Non-finite cross-moment loss or image gradient; image update aborted')

        image_before_step = image_syn.detach().clone()
        gradient_rms = image_syn.grad.square().mean().sqrt().item()
        loss_before = loss_avg.item()
        loss_after, accepted_step, trials, accepted = projected_backtracking_step(
            image_syn, sampled_targets, dsa_strategy, args.dsa_param,
            pixel_min, pixel_max, loss_before, trial_step,
        )
        sampled_targets.clear()
        if accepted:
            # Try growing the last successful step; every trial must still pass
            # the same sufficient-decrease check, so growth cannot bypass it.
            trial_step = min(2.0 * accepted_step, math.sqrt(torch.finfo(image_syn.dtype).max))

        if it == 1 or it % 2 == 0 or not accepted:
            with torch.no_grad():
                # Diagnostic normalization only; the optimized loss is the SUM.
                target_norm_squared = max(target_f2_avg.item(), torch.finfo(loss_avg.dtype).tiny)
                relative_before = math.sqrt(loss_before / target_norm_squared)
                relative_after = math.sqrt(loss_after / target_norm_squared)
                pixel_step = (image_syn - image_before_step) * diag_std * 255
                pixel_drift = (image_syn - image_syn_init) * diag_std * 255
                step_rms_255 = pixel_step.square().mean().sqrt().item()
                drift_rms_255 = pixel_drift.square().mean().sqrt().item()
                image_syn_uint8 = ((image_syn * diag_std + diag_mean) * 255 + 0.5).clamp(0, 255).to(torch.uint8)
                changed_pct = (image_syn_uint8 != image_syn_init_uint8).float().mean().item() * 100
                print(
                    f'{get_time()} iter = {it:05d}, loss_F2_before = {loss_before:.6f}, loss_F2_after = {loss_after:.6f}, '
                    f'rel_F = {relative_before:.6f} -> {relative_after:.6f}, grad_RMS = {gradient_rms:.3e}, '
                    f'step_RMS_255 = {step_rms_255:.4f}, drift_RMS_255 = {drift_rms_255:.4f}, '
                    f'step_size = {accepted_step:.3e}, trials = {trials}, accepted = {accepted}, '
                    f'PNG values changed = {changed_pct:.5f}%',
                    flush=True,
                )
        del image_before_step

        if it == args.Iteration:
            data_save = copy.deepcopy(image_syn.detach().cpu())
            torch.save({'data': data_save}, os.path.join(args.save_path, 'res_GraphEigenspace_%s_%s_%dpercent.pt' % (args.dataset, args.model, args.percentage)))

        if it % 50 == 0 or it == args.Iteration:
            checkpoint_accs = evaluate_checkpoint(it)
            if it == args.Iteration:
                for model_eval, accs in checkpoint_accs.items():
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
