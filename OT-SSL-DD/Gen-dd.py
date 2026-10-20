import os
import argparse
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
from eval import ROOT, reference_modules
from utils import get_dataset, get_time, DiffAugment, ParamDiffAug


class VAE(nn.Module):

    def __init__(self, channel, im_size, latent_dim):
        super().__init__()
        size = im_size[0] // 8
        self.encoder = nn.Sequential(
            nn.Conv2d(channel, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),
            nn.Flatten())
        self.fc_mu = nn.Linear(128*size*size, latent_dim)
        self.fc_logvar = nn.Linear(128*size*size, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128*size*size), nn.ReLU(),
            nn.Unflatten(1, (128, size, size)),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, 3, 1, 1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, 3, 1, 1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32, channel, 3, 1, 1), nn.Sigmoid())

    def forward(self, images):
        features = self.encoder(images)
        mu, logvar = self.fc_mu(features), self.fc_logvar(features)
        z = mu + torch.randn_like(mu) * torch.exp(0.5*logvar)
        return self.decoder(z), mu, logvar


class ResidualBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(8, channels), nn.SiLU(),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.GroupNorm(8, channels), nn.SiLU(),
            nn.Conv2d(channels, channels, 3, 1, 1))

    def forward(self, images):
        return images + self.block(images)


class ResidualVAE(nn.Module):

    def __init__(self, channel, im_size, latent_dim):
        super().__init__()
        size = im_size[0] // 8
        channels = [64, 128, 256, 512]
        encoder = [nn.Conv2d(channel, channels[0], 3, 1, 1)]
        for index, width in enumerate(channels):
            if index > 0:
                encoder.append(nn.Conv2d(channels[index-1], width, 4, 2, 1))
            encoder.extend([ResidualBlock(width), ResidualBlock(width)])
        encoder.extend([nn.GroupNorm(8, 512), nn.SiLU(), nn.Flatten()])
        self.encoder = nn.Sequential(*encoder)
        self.fc_mu = nn.Linear(512*size*size, latent_dim)
        self.fc_logvar = nn.Linear(512*size*size, latent_dim)

        decoder = [nn.Linear(latent_dim, 512*size*size),
                   nn.Unflatten(1, (512, size, size))]
        for index, width in enumerate(reversed(channels)):
            if index > 0:
                decoder.extend([
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                    nn.Conv2d(width*2, width, 3, 1, 1)])
            decoder.extend([ResidualBlock(width), ResidualBlock(width)])
        decoder.extend([nn.GroupNorm(8, 64), nn.SiLU(),
                        nn.Conv2d(64, channel, 3, 1, 1), nn.Sigmoid()])
        self.decoder = nn.Sequential(*decoder)

    def forward(self, images):
        features = self.encoder(images)
        mu, logvar = self.fc_mu(features), self.fc_logvar(features)
        z = mu + torch.randn_like(mu) * torch.exp(0.5*logvar)
        return self.decoder(z), mu, logvar


def feature_moments(features):
    mean = features.mean(dim=0)
    centered = features - mean
    covariance = centered.t() @ centered / len(features)
    return mean, covariance


def mmd_kernel(features_a, features_b, bandwidth):
    distances = (features_a.square().sum(1, keepdim=True)
                 + features_b.square().sum(1)[None] - 2 * features_a @ features_b.t()).clamp_min(0)
    kernel = 0
    for scale in (0.25, 0.5, 1.0, 2.0, 4.0):
        kernel = kernel + torch.exp(-distances / (2 * bandwidth * scale))
    return kernel / 5


def mmd_loss(features_syn, features_real, bandwidth, real_kernel):
    features_syn = F.normalize(features_syn, dim=1)
    return (mmd_kernel(features_syn, features_syn, bandwidth).mean()
            - 2 * mmd_kernel(features_syn, features_real, bandwidth).mean() + real_kernel)


def mmd_features(latents, generator, net, normalization, batch_size):
    mean, std = normalization
    return torch.cat([net((generator(batch) - mean) / std) for batch in latents.split(batch_size)])


def mmd_update(z, generator, teachers, normalizations, real_distributions, bandwidths, args):
    loss_avg = 0
    for net, normalization, distribution, bandwidth in zip(teachers, normalizations, real_distributions, bandwidths):
        order = torch.randperm(len(distribution))[:args.mmd_batch_real]
        real = distribution[order].to(args.device)
        with torch.no_grad():
            real_kernel = mmd_kernel(real, real, bandwidth).mean()
        ''' compute the joint feature gradient, then backpropagate in image batches '''
        with torch.no_grad():
            features = mmd_features(z, generator, net, normalization, args.batch_syn)
        features.requires_grad_(True)
        loss = mmd_loss(features, real, bandwidth, real_kernel) / len(teachers)
        gradient = torch.autograd.grad(loss, features)[0]
        teacher_mean, teacher_std = normalization
        for start in range(0, len(z), args.batch_syn):
            end = min(start + args.batch_syn, len(z))
            images = generator(z[start:end])
            features_syn = net((images - teacher_mean) / teacher_std)
            features_syn.backward(gradient[start:end])
        loss_avg += loss.item()
    return loss_avg


def resume_distillation(path, z, optimizer, expected, common):
    if not os.path.exists(path):
        return 0, []
    saved = torch.load(path, map_location='cpu', weights_only=True)
    for key, value in expected.items():
        if saved.get(key) != value:
            raise ValueError('Cannot resume: checkpoint %s differs from this run'%key)
    latents = saved['latents']
    iteration = saved['iteration']
    history = saved['loss_history'].tolist()
    if (latents.shape != z.shape or not torch.isfinite(latents).all()
            or not isinstance(iteration, int) or iteration < 0 or len(history) != iteration):
        raise ValueError('Cannot resume: invalid latents, iteration or loss history')
    with torch.no_grad():
        z.copy_(latents)
    if 'optimizer_z' in saved:
        optimizer.load_state_dict(saved['optimizer_z'])
    else:
        # shortcut: old checkpoints lack Adam state; only future saves resume it exactly.
        print('WARNING: old checkpoint has no Adam state; continuing saved latents with fresh Adam.')
    if 'rng_state' in saved:
        common.restore_rng(saved['rng_state'])
    else:
        print('WARNING: old checkpoint has no RNG state; augmentation sequence will restart.')
    print('Resumed distillation at iteration %d: %s'%(iteration, path))
    return iteration, history


def main():

    parser = argparse.ArgumentParser(description='Generative Latent Feature Matching')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--model', type=str, default='ResNet18BN', help='saved teacher architecture')
    parser.add_argument('--percentage', type=float, default=0.2, help='synthetic images as a percentage of the real training set')
    parser.add_argument('--teacher_ssl', type=str, default='simclr', help='saved teacher SSL method')
    parser.add_argument('--teacher_seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--teacher_path', type=str, default=str(ROOT / 'Evaluate_benchamrk/full_ssl_checkpoints_1000ep_3seeds'))
    parser.add_argument('--num_exp', type=int, default=1, help='the number of distillation experiments')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--Iteration', type=int, default=20000, help='latent optimization iterations')
    parser.add_argument('--lr_z', type=float, default=0.05, help='learning rate for latent vectors')
    parser.add_argument('--batch_real', type=int, default=512, help='batch size for VAE training and real feature extraction')
    parser.add_argument('--batch_syn', type=int, default=512, help='synthetic images per forward/backward pass')
    parser.add_argument('--latent_dim', type=int, default=None)
    parser.add_argument('--vae_epochs', type=int, default=None)
    parser.add_argument('--vae_lr', type=float, default=None)
    parser.add_argument('--vae_beta', type=float, default=None, help='KL weight during VAE training')
    parser.add_argument('--vae_path', type=str, default=None)
    parser.add_argument('--residual_vae', action='store_true', default=True)
    parser.add_argument('--vae_warmup', type=int, default=50, help='residual VAE KL warmup epochs')
    parser.add_argument('--vae_ema', type=float, default=0.999, help='residual VAE EMA decay')
    parser.add_argument('--train_vae', action='store_true', help='train the VAE instead of reusing vae_path')
    parser.add_argument('--mmd', action='store_true', help='per-teacher kernel MMD using the saved 350-epoch residual VAE')
    parser.add_argument('--mmd_batch_real', type=int, default=256, help='cached real features sampled per MMD update')
    parser.add_argument('--aug_strategy', type=str, default='color_crop_cutout_flip_scale_rotate')
    parser.add_argument('--cov_weight', type=float, default=2.0)
    parser.add_argument('--save_every', type=int, default=2000)
    parser.add_argument('--resume', action='store_true', help='resume saved experiments and skip completed ones')
    parser.add_argument('--data_path', type=str, default=str(ROOT.parent / 'data'))
    parser.add_argument('--cache_path', type=str, default=str(ROOT / 'OT-SSL-DD/gen_dd_cache'))
    parser.add_argument('--save_path', type=str, default=str(ROOT / 'OT-SSL-DD/result_GenerativeFeatureMatching'))

    args = parser.parse_args()
    if args.mmd:
        args.residual_vae = True
    if args.latent_dim is None:
        args.latent_dim = 256 if args.residual_vae else 128
    if args.vae_epochs is None:
        args.vae_epochs = 350 if args.residual_vae else 100
    if args.vae_lr is None:
        args.vae_lr = 0.0002 if args.residual_vae else 0.001
    if args.vae_beta is None:
        args.vae_beta = 0.01 if args.residual_vae else 0.1
    if args.vae_path is None:
        if args.residual_vae:
            args.vae_path = '/scratch/mmoslem3/MKDT/OT-SSL-DD/gen_dd_cache/residual_vae_CIFAR10_z256_e400_beta0.01_seed0.pt'
        else:
            args.vae_path = '/scratch/mmoslem3/MKDT/OT-SSL-DD/gen_dd_cache/vae_CIFAR10_z128_e100_beta0.1_seed0.pt'
    args.train_vae = args.train_vae or (args.residual_vae and not os.path.exists(args.vae_path))
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    common = reference_modules()['common']
    common.seed_all(args.seed)

    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.cache_path, exist_ok=True)
    print('Hyper-parameters: \n', args.__dict__)

    ''' organize the real dataset '''
    channel, im_size, num_classes, _, mean, std, dst_train, _, _ = get_dataset(args.dataset, args.data_path)
    data_mean = torch.tensor(mean, device=args.device).view(1, channel, 1, 1)
    data_std = torch.tensor(std, device=args.device).view(1, channel, 1, 1)
    num_syn = int(len(dst_train) * args.percentage / 100)

    ''' load the three frozen full-data SSL teachers '''
    teachers, normalizations, teacher_checkpoints, teacher_hashes = [], [], [], []
    for seed in args.teacher_seeds:
        checkpoint = os.path.join(args.teacher_path, args.dataset, args.model, 'full', 'pt100', args.teacher_ssl, 'seed_%03d'%seed, 'encoder.pt')
        net, _, saved = common.load_encoder(checkpoint, device=args.device)
        teachers.append(net)
        teacher_mean, teacher_std = (torch.tensor(value, device=args.device).view(1, channel, 1, 1) for value in saved['normalization'])
        normalizations.append((teacher_mean, teacher_std))
        teacher_checkpoints.append(checkpoint)
        teacher_hashes.append(saved['encoder_sha256'])
        print('Loaded teacher seed %d: %s'%(seed, checkpoint))

    ''' cache separate real feature distributions for MMD '''
    feature_paths, real_distributions, bandwidths = [], [], []
    if args.mmd:
        for net, normalization, teacher_hash in zip(teachers, normalizations, teacher_hashes):
            feature_key = common.digest_json({
                'dataset': args.dataset, 'data_path': os.path.abspath(args.data_path),
                'teacher': teacher_hash, 'normalization': [mean, std], 'feature_normalization': 'unit_l2'})
            feature_path = os.path.join(args.cache_path, 'mmd_real_%s_%s.pt'%(args.dataset, feature_key))
            if os.path.exists(feature_path):
                features = torch.load(feature_path, map_location='cpu', weights_only=True)['features']
                print('Loaded real MMD features: %s'%feature_path)
            else:
                trainloader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_real, shuffle=False, num_workers=0)
                features = []
                teacher_mean, teacher_std = normalization
                with torch.no_grad():
                    for images, _ in trainloader:
                        pixels = images.to(args.device) * data_std + data_mean
                        features.append(F.normalize(net((pixels - teacher_mean) / teacher_std), dim=1).cpu())
                features = torch.cat(features)
                torch.save({'features': features, 'teacher_sha256': teacher_hash}, feature_path)
                print('Saved real MMD features: %s'%feature_path)
            feature_paths.append(feature_path)
            real_distributions.append(features)
            order = torch.randperm(len(features))[:min(2048, len(features))]
            real = features[order].to(args.device)
            with torch.no_grad():
                distances = torch.pdist(real[:min(512, len(real))]).square()
                bandwidth = distances.median().clamp_min(1e-6)
            bandwidths.append(bandwidth)

    ''' extract and cache real two-view means and covariances for every teacher '''
    moment_param = ParamDiffAug()
    moment_param.aug_mode = 'M'
    target_key = common.digest_json({
        'version': 1, 'dataset': args.dataset, 'data_path': os.path.abspath(args.data_path),
        'teachers': teacher_hashes, 'seed': args.seed, 'augmentation': args.aug_strategy})
    target_path = os.path.join(args.cache_path, 'augmented_moments_%s_%s.pt'%(args.dataset, target_key))
    if args.mmd:
        target_path = None
    elif os.path.exists(target_path):
        saved_moments = torch.load(target_path, map_location=args.device, weights_only=True)
        real_means = saved_moments['means']
        real_covariances = saved_moments['covariances']
        print('Loaded real augmented moments: %s'%target_path)
    else:
        common.seed_all(args.seed + 10000)
        trainloader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_real, shuffle=False, num_workers=0)
        feature_sums = [None for _ in teachers]
        feature_products = [None for _ in teachers]
        feature_count = 0
        with torch.no_grad():
            for images, _ in trainloader:
                pixels = images.to(args.device) * data_std + data_mean
                first = DiffAugment(pixels.clone(), args.aug_strategy, param=moment_param)
                second = DiffAugment(pixels.clone(), args.aug_strategy, param=moment_param)
                feature_count += len(first) + len(second)
                for teacher_index, (net, (teacher_mean, teacher_std)) in enumerate(zip(teachers, normalizations)):
                    features = torch.cat([
                        net((first - teacher_mean) / teacher_std),
                        net((second - teacher_mean) / teacher_std)])
                    feature_sum = features.sum(dim=0)
                    feature_product = features.t() @ features
                    if feature_sums[teacher_index] is None:
                        feature_sums[teacher_index] = feature_sum
                        feature_products[teacher_index] = feature_product
                    else:
                        feature_sums[teacher_index] += feature_sum
                        feature_products[teacher_index] += feature_product

            real_means = [feature_sum / feature_count for feature_sum in feature_sums]
            real_covariances = [feature_product / feature_count - mean[:, None] @ mean[None, :]
                                for feature_product, mean in zip(feature_products, real_means)]

        torch.save({
            'means': [mean.cpu() for mean in real_means],
            'covariances': [covariance.cpu() for covariance in real_covariances],
            'teacher_checkpoints': teacher_checkpoints,
            'augmentation': args.aug_strategy,
            'feature_count': feature_count}, target_path)
        print('Saved real augmented moments: %s'%target_path)

    ''' reuse the saved VAE by default, or train it when requested '''
    common.seed_all(args.seed)
    os.makedirs(os.path.dirname(os.path.abspath(args.vae_path)), exist_ok=True)

    saved_vae = None
    if not args.train_vae:
        saved_vae = torch.load(args.vae_path, map_location='cpu', weights_only=True)
        args.latent_dim = saved_vae['latent_dim']
        if args.residual_vae and not args.mmd and saved_vae['epoch'] < args.vae_epochs:
            args.train_vae = True
    vae_class = ResidualVAE if args.residual_vae else VAE
    vae = vae_class(channel, im_size, args.latent_dim).to(args.device)
    if not args.train_vae:
        vae.load_state_dict(saved_vae['vae_ema'] if args.residual_vae else saved_vae['vae'])
        vae_epoch = saved_vae['epoch']
        print('Loaded VAE at epoch %d: %s'%(saved_vae['epoch'], args.vae_path))
        del saved_vae
    else:
        optimizer_vae = torch.optim.Adam(vae.parameters(), lr=args.vae_lr)
        if args.residual_vae:
            vae_ema = copy.deepcopy(vae).eval().requires_grad_(False)
        start_epoch = 0
        vae_loss_history = []
        if saved_vae is not None:
            vae.load_state_dict(saved_vae['vae'])
            vae_ema.load_state_dict(saved_vae['vae_ema'])
            optimizer_vae.load_state_dict(saved_vae['optimizer'])
            start_epoch = saved_vae['epoch']
            vae_loss_history = saved_vae['loss_history']
            print('Resuming VAE at epoch %d: %s'%(start_epoch, args.vae_path))
            del saved_vae
        trainloader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_real, shuffle=True, num_workers=0)
        vae.train()
        for epoch in range(start_epoch, args.vae_epochs):
            loss_avg = 0
            beta = args.vae_beta
            if args.residual_vae and args.vae_warmup > 0:
                beta *= min(epoch / args.vae_warmup, 1.0)
            for images, _ in trainloader:
                images = images.to(args.device) * data_std + data_mean
                reconstruction, mu, logvar = vae(images)
                if args.residual_vae:
                    loss_reconstruction = (reconstruction - images).abs().flatten(1).sum(1).mean()
                else:
                    loss_reconstruction = (reconstruction - images).square().flatten(1).sum(1).mean()
                loss_kl = -0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(1).mean()
                loss = loss_reconstruction + beta * loss_kl
                optimizer_vae.zero_grad()
                loss.backward()
                optimizer_vae.step()
                if args.residual_vae:
                    with torch.no_grad():
                        for averaged, current in zip(vae_ema.parameters(), vae.parameters()):
                            averaged.lerp_(current, 1 - args.vae_ema)
                loss_avg += loss.item() * len(images)

            loss_avg /= len(dst_train)
            vae_loss_history.append(loss_avg)
            print('%s VAE epoch = %04d, loss = %.4f'%(get_time(), epoch+1, loss_avg))
            saved_vae = {
                'vae': vae.state_dict(), 'optimizer': optimizer_vae.state_dict(),
                'epoch': epoch+1, 'latent_dim': args.latent_dim,
                'dataset': args.dataset, 'normalization': [mean, std],
                'loss_history': vae_loss_history,
                'architecture': 'residual' if args.residual_vae else 'basic',
                'vae_lr': args.vae_lr, 'vae_beta': args.vae_beta}
            if args.residual_vae:
                saved_vae.update(vae_ema=vae_ema.state_dict(), vae_warmup=args.vae_warmup, vae_ema_decay=args.vae_ema)
            torch.save(saved_vae, args.vae_path)
            del saved_vae
        if args.residual_vae:
            vae.load_state_dict(vae_ema.state_dict())
            del vae_ema
        vae_epoch = args.vae_epochs
        del optimizer_vae

    ''' freeze the generator; only Z is optimized below '''
    vae.eval().requires_grad_(False)
    generator = vae.decoder
    print('Frozen generator: %s'%args.vae_path)

    for exp in range(args.num_exp):
        print('\n================== Exp %d ==================\n '%exp)
        common.seed_all(args.seed + exp)

        ''' Z is the only optimized variable in the distillation loop '''
        z = torch.randn(num_syn, args.latent_dim, device=args.device, requires_grad=True)
        optimizer_z = torch.optim.Adam([z], lr=args.lr_z)
        result_path = os.path.join(args.save_path, 'res_GFM_%s_%s_%gpercent_exp%d.pt'%(args.dataset, args.model, args.percentage, exp))
        start_iteration = 0
        loss_history = []
        if args.resume:
            start_iteration, loss_history = resume_distillation(
                result_path, z, optimizer_z, {
                    'dataset': args.dataset, 'percentage': args.percentage, 'seed': args.seed + exp,
                    'method': 'GenerativeMMD' if args.mmd else 'GenerativeAugmentedMomentMatching',
                    'teacher_ssl': args.teacher_ssl, 'teacher_checkpoints': teacher_checkpoints,
                    'vae_path': args.vae_path, 'vae_epoch': vae_epoch,
                    'augmentation': args.aug_strategy, 'covariance_weight': args.cov_weight,
                    'target_path': target_path}, common)
        if start_iteration >= args.Iteration:
            print('Already complete: %s (%d iterations)'%(result_path, start_iteration))
            continue
        print('%s latent distillation begins, synthetic images = %d'%(get_time(), num_syn))

        for it in range(start_iteration+1, args.Iteration+1):

            ''' generate two views and match their teacher-feature means and covariances '''
            optimizer_z.zero_grad()
            loss_avg = 0
            if args.mmd:
                loss_avg = mmd_update(z, generator, teachers, normalizations, real_distributions, bandwidths, args)
            else:
                image_syn = torch.cat([generator(batch) for batch in z.split(args.batch_syn)])
                first = DiffAugment(image_syn.clone(), args.aug_strategy, param=moment_param)
                second = DiffAugment(image_syn.clone(), args.aug_strategy, param=moment_param)
                loss = 0
                for teacher_index, (net, (teacher_mean, teacher_std)) in enumerate(zip(teachers, normalizations)):
                    features = torch.cat([
                        net((first - teacher_mean) / teacher_std),
                        net((second - teacher_mean) / teacher_std)])
                    synthetic_mean, synthetic_covariance = feature_moments(features)
                    mean_loss = (synthetic_mean - real_means[teacher_index]).square().sum()
                    covariance_loss = (synthetic_covariance - real_covariances[teacher_index]).square().sum()
                    loss = loss + (mean_loss + args.cov_weight * covariance_loss) / len(teachers)
                loss.backward()
                loss_avg = loss.item()
            optimizer_z.step()
            loss_history.append(loss_avg)

            if it%100 == 0:
                print('%s iter = %04d, loss = %.6f'%(get_time(), it, loss_avg))

            ''' visualize and save '''
            if it%args.save_every == 0 or it == args.Iteration:
                with torch.no_grad():
                    selected_z = z.detach().cpu()
                    pixels = torch.cat([generator(batch).cpu() for batch in z.split(args.batch_syn)])
                    image_syn = (pixels - data_mean.cpu()) / data_std.cpu()

                save_name = os.path.join(args.save_path, 'vis_GFM_%s_%s_%gpercent_exp%d_iter%d.png'%(args.dataset, args.model, args.percentage, exp, it))
                save_image(pixels, save_name, nrow=int(np.ceil(np.sqrt(num_syn))))
                common.atomic_save(result_path, {
                    'data': image_syn, 'latents': selected_z,
                    'optimizer_z': optimizer_z.state_dict(), 'rng_state': common.rng_state(),
                    'dataset': args.dataset, 'percentage': args.percentage,
                    'normalization': [mean, std],
                    'method': 'GenerativeMMD' if args.mmd else 'GenerativeAugmentedMomentMatching',
                    'iteration': it, 'seed': args.seed + exp,
                    'loss_history': torch.tensor(loss_history),
                    'teacher_checkpoints': teacher_checkpoints,
                    'teacher_ssl': args.teacher_ssl,
                    'target_path': target_path,
                    'vae_path': args.vae_path, 'vae_epoch': vae_epoch,
                    'dis_metric': 'mmd' if args.mmd else 'augmented_moments',
                    'augmentation': args.aug_strategy,
                    'covariance_weight': args.cov_weight,
                    'mmd_feature_paths': feature_paths,
                    'mmd_bandwidths': [value.item() for value in bandwidths],
                    'mmd_kernel_scales': [0.25, 0.5, 1.0, 2.0, 4.0],
                    'mmd_batch_real': args.mmd_batch_real,
                    'vae_architecture': 'residual' if args.residual_vae else 'basic'})
                print('Saved synthetic data to %s'%result_path)


if __name__ == '__main__':
    main()
