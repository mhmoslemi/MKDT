import os
import argparse
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from torchvision.utils import save_image
from dm_dsa import DMDSA
from eval import ROOT, evaluate_synset_SSL, reference_modules
from utils import get_dataset, get_transport_plan, get_time, transport_soft_assignment_loss


# -------------------------
# Evaluation
# model_train = ConvNet, model_eval = ConvNet, iteration = 0
# [2026-10-02 09:13:39] Evaluate_SSL_00: method = simclr ssl epoch = 1000 linear epoch = 0100 labeled = 1.00% train time = 42 s ssl loss = 4.292197 train loss = 0.008754 train acc = 1.0000, test acc = 0.3800
# [2026-10-02 09:14:22] Evaluate_SSL_01: method = simclr ssl epoch = 1000 linear epoch = 0100 labeled = 1.00% train time = 42 s ssl loss = 4.299274 train loss = 0.013706 train acc = 1.0000, test acc = 0.3882
# [2026-10-02 09:15:06] Evaluate_SSL_02: method = simclr ssl epoch = 1000 linear epoch = 0100 labeled = 1.00% train time = 42 s ssl loss = 4.295786 train loss = 0.008535 train acc = 1.0000, test acc = 0.3625
# [2026-10-02 09:15:49] Evaluate_SSL_03: method = simclr ssl epoch = 1000 linear epoch = 0100 labeled = 1.00% train time = 42 s ssl loss = 4.315185 train loss = 0.013006 train acc = 1.0000, test acc = 0.3753
# Evaluate 4 random ConvNet, mean = 0.3765 std = 0.0093


# Evaluation
# model_train = ConvNet, model_eval = ConvNet, iteration = 0
# [2026-10-02 09:23:21] Evaluate_SSL_00: method = simclr ssl epoch = 1000 linear epoch = 0100 labeled = 1.00% train time = 42 s ssl loss = 4.296797 train loss = 0.014125 train acc = 1.0000, test acc = 0.3767
# [2026-10-02 09:24:04] Evaluate_SSL_01: method = simclr ssl epoch = 1000 linear epoch = 0100 labeled = 1.00% train time = 42 s ssl loss = 4.289597 train loss = 0.007593 train acc = 1.0000, test acc = 0.3829
# [2026-10-02 09:24:47] Evaluate_SSL_02: method = simclr ssl epoch = 1000 linear epoch = 0100 labeled = 1.00% train time = 42 s ssl loss = 4.289647 train loss = 0.012315 train acc = 1.0000, test acc = 0.3696
# [2026-10-02 09:25:31] Evaluate_SSL_03: method = simclr ssl epoch = 1000 linear epoch = 0100 labeled = 1.00% train time = 42 s ssl loss = 4.303205 train loss = 0.007448 train acc = 1.0000, test acc = 0.3706
# Evaluate 4 random ConvNet, mean = 0.3750 std = 0.0053

def main(loss_function=transport_soft_assignment_loss, method='OTSoftAssignment'):
    parser = argparse.ArgumentParser(description='%s distillation with a frozen SSL teacher' % method)

    # -------------------- Data --------------------
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--percentage', type=float, default=0.2, help='percentage of total data size')
    parser.add_argument('--data_path', type=str, default=str(ROOT.parent / 'data'), help='dataset path')

    # -------------------- Distillation --------------------
    parser.add_argument('--Iteration', type=int, default=8000, help='synthetic image updates')
    parser.add_argument('--lr_img', type=float, default=0.001, help='Adam learning rate for synthetic images')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--save_every', type=int, default=500)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--dsa', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--dsa_strategy', '--dsa-strategy', default='color_crop_cutout_flip_scale_rotate')

    # -------------------- Network --------------------
    parser.add_argument('--model', choices=['ConvNet', 'ConvNetBN', 'ResNet18', 'ResNet18BN', 'VGG11', 'VGG11BN'], default='ResNet18BN')
    parser.add_argument('--teacher_ssl', choices=['simclr', 'barlowtwins'], default='simclr')
    parser.add_argument('--teacher_seed', type=int, default=0)
    parser.add_argument('--teacher_path', default=str(ROOT / 'Evaluate_benchamrk/full_ssl_checkpoints_1000ep_3seeds'))
    parser.add_argument('--temperature', type=float, default=0.5, help='temperature for SimCLR')

    # -------------------- Optimal Transport --------------------
    parser.add_argument('--ot_lambda', type=float, default=0.01, help='entropy regularization for the transport plan')
    parser.add_argument('--sinkhorn_iterations', type=int, default=100, help='number of Sinkhorn iterations')

    # -------------------- Evaluation --------------------
    parser.add_argument('--ssl_method', '--ssl-method', choices=['simclr', 'barlowtwins'], default='simclr')
    parser.add_argument('--num_eval', '--runs', type=int, default=1)
    parser.add_argument('--epoch_eval_train', '--ssl-epochs', type=int, default=None)
    parser.add_argument('--ssl-log-every', type=int, default=100)
    parser.add_argument('--label_percentage', '--label-percentage', type=int, choices=[1, 5], default=1)
    parser.add_argument('--epoch_linear_train', '--probe-epochs', type=int, default=100)
    parser.add_argument('--lr_linear', '--lr-linear', type=float, default=0.1)
    parser.add_argument('--batch_linear', '--batch-linear', type=int, default=256)

    # -------------------- Output --------------------
    parser.add_argument('--save_path', type=str, default=str(ROOT / 'OT-SSL-DD/result_OT_SSL'), help='path to save results')

    args = parser.parse_args()
    args.geometry_weight = 0.0
    if (not 0 < args.percentage <= 100 or args.Iteration < 1 or args.lr_img <= 0
            or min(args.batch_real, args.save_every, args.sinkhorn_iterations) < 1
            or args.ot_lambda <= 0 or args.temperature <= 0 or args.num_eval < 0):
        parser.error('Invalid data size, iterations, optimizer, batch or OT setting')
    common = reference_modules()['common']
    common.seed_all(args.seed)
    os.makedirs(args.save_path, exist_ok=True)
    print('Hyper-parameters: \n', args.__dict__)

    channel, im_size, _, _, mean, std, dst_train, _, _ = get_dataset(args.dataset, args.data_path)
    num_syn = int(len(dst_train) * args.percentage / 100)
    if num_syn < 2:
        parser.error('Need at least two synthetic images')

    ''' organize the real dataset '''
    images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
    images_all = torch.cat(images_all, dim=0)

    def get_images(n):
        idx_shuffle = np.random.permutation(len(images_all))[:n]
        return images_all[idx_shuffle].to(args.device)

    
    ''' initialize the synthetic data '''
    image_syn = get_images(num_syn).detach().clone().requires_grad_(True)
    image_syn_init = image_syn.detach().clone()
    diag_mean = torch.tensor(mean, device=args.device).view(1, channel, 1, 1)
    diag_std = torch.tensor(std, device=args.device).view(1, channel, 1, 1)
    augmentation = DMDSA(args.dsa_strategy, mean, std).to(args.device) if args.dsa else None

    ''' load the saved full-data teacher '''
    checkpoint = Path(args.teacher_path) / args.dataset / args.model.removesuffix('BN') / 'full' / 'pt100' / args.teacher_ssl / ('seed_%03d' % args.teacher_seed) / 'encoder.pt'
    teacher, _, teacher_saved = common.load_encoder(checkpoint, device=args.device)
    if teacher_saved['model'] != args.model.removesuffix('BN') or teacher_saved['input_size'] != im_size[0]:
        raise ValueError('Teacher architecture or input size does not match this run')
    teacher_mean, teacher_std = (torch.tensor(value, device=args.device).view(1, channel, 1, 1)
                                 for value in teacher_saved['normalization'])
    print('Loaded teacher seed %d: %s' % (args.teacher_seed, checkpoint), flush=True)

    ''' training '''
    optimizer_img = torch.optim.Adam([image_syn], lr=args.lr_img)
    result_path = Path(args.save_path) / ('res_OT-SSL_%s_%s_%gpercent.pt' % (args.dataset, args.model, args.percentage))
    print('%s training begins' % get_time(), flush=True)
    for it in range(1, args.Iteration + 1):
        synthetic_view = image_syn
        seed = DMDSA.seeds(1)[0] if args.dsa else None
        if args.dsa:
            synthetic_view = augmentation.normalized(synthetic_view, seed)
        real_features = []
        with torch.no_grad():
            for real_batch in images_all.split(args.batch_real):
                real_view = real_batch.to(args.device)
                if args.dsa:
                    real_view = augmentation.normalized(real_view, seed)
                real_features.append(teacher((real_view * diag_std + diag_mean - teacher_mean) / teacher_std))
        output_real = torch.cat(real_features)
        output_syn = teacher((synthetic_view * diag_std + diag_mean - teacher_mean) / teacher_std)
        transport_plan = get_transport_plan(output_real, output_syn.detach(), args)
        loss = loss_function(output_real, output_syn, transport_plan, args)
        optimizer_img.zero_grad()
        loss.backward()
        optimizer_img.step()
        with torch.no_grad():
            image_syn.copy_(torch.maximum(torch.minimum(image_syn, (1 - diag_mean) / diag_std), -diag_mean / diag_std))

        if it % 100 == 0 or it == 1:
            with torch.no_grad():
                pixel_change = ((image_syn - image_syn_init) * diag_std * 255).square().mean().sqrt().item()
            print('%s iter = %05d, loss = %.6f, RMS = %.2f/255' % (get_time(), it, loss.item(), pixel_change), flush=True)
        if it % args.save_every == 0 or it == args.Iteration:
            pixels = (image_syn.detach() * diag_std + diag_mean).clamp(0, 1).cpu()
            save_image(pixels, str(result_path.with_name(result_path.stem.replace('res_', 'vis_', 1) + '_iter%d.png' % it)), nrow=int(np.ceil(np.sqrt(num_syn))))
            common.atomic_save(result_path, {
                'data': image_syn.detach().cpu(), 'normalization': [mean, std],
                'dataset': args.dataset, 'model': args.model, 'percentage': args.percentage,
                'method': method, 'iteration': it, 'seed': args.seed,
                'teacher_checkpoint': str(checkpoint), 'teacher_sha256': teacher_saved['encoder_sha256'],
                'dsa': args.dsa, 'augmentation': augmentation.configuration() if args.dsa else None,
                'optimizer': 'adam', 'lr_img': args.lr_img,
            })
            print('Saved synthetic data to %s' % result_path, flush=True)

    ''' evaluate the exact saved synthetic data '''
    del teacher, output_syn, output_real, loss, optimizer_img
    saved = torch.load(result_path, map_location='cpu', weights_only=True)
    output = Path(args.save_path) / 'evaluation' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    accuracies = []
    for run in range(args.num_eval):
        _, _, accuracy = evaluate_synset_SSL(run, args.model, saved['data'], args,
                                             output / ('seed_%03d' % (args.seed + run)), saved['normalization'])
        accuracies.append(accuracy * 100)
        common.atomic_json(output / 'summary.json', {
            'config': dict(vars(args), synthetic_checkpoint=str(result_path.resolve())),
            'complete': len(accuracies) == args.num_eval, 'accuracies_percent': accuracies,
            'mean_percent': float(np.mean(accuracies)),
        })
        print('RESULT seed=%d accuracy=%.2f%%' % (args.seed + run, accuracy * 100), flush=True)



if __name__ == '__main__':
    main()
