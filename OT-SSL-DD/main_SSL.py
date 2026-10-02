import os
import copy
import argparse
import numpy as np

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from utils import (
    get_dataset,
    get_network,
    get_eval_pool,
    evaluate_synset_SSL,
    get_time,
    init_ssl,
    epoch_ssl,
    get_transport_plan,
    ParamDiffAug,
)


# random subset baseline:
# 2%: test acc ~ 0.3993
# 1%: test acc ~ 0.3895


def transport_soft_assignment_loss(
    output_real,
    output_syn,
    transport_plan,
    args
):
    """
    Match the OT-induced soft assignment distribution q(s_j | x_i)
    with the similarity-induced distribution p(s_j | x_i).

    OT is treated as a fixed target during each synthetic-image update.
    """

    # ---------------------------------------------------------
    # Fixed OT target
    # ---------------------------------------------------------
    q = transport_plan.detach()

    # Each real image gets a probability distribution
    # over ALL synthetic images.
    #
    # q_ij = P(s_j | x_i)
    q = q / q.sum(dim=1, keepdim=True).clamp_min(1e-12)

    # ---------------------------------------------------------
    # Representations
    # ---------------------------------------------------------
    # Real features should never receive gradients.
    output_real = F.normalize(
        output_real.detach(),
        dim=1
    )

    # Synthetic features DO receive gradients.
    output_syn = F.normalize(
        output_syn,
        dim=1
    )

    # ---------------------------------------------------------
    # Similarity-induced distribution
    #
    # p_ij = softmax(sim(x_i, s_j) / tau)
    # ---------------------------------------------------------
    logits = torch.mm(
        output_real,
        output_syn.t()
    )

    logits = logits / args.temperature

    log_p = F.log_softmax(
        logits,
        dim=1
    )

    # ---------------------------------------------------------
    # Cross entropy between OT assignments q
    # and similarity assignments p.
    #
    # This does NOT force one synthetic image to be
    # the only positive for a real image.
    #
    # A real image can distribute its mass over several
    # synthetic images.
    # ---------------------------------------------------------
    loss_per_real = -(q * log_p).sum(dim=1)

    loss = loss_per_real.mean()

    return loss


def main():

    parser = argparse.ArgumentParser(
        description='OT-based SSL Dataset Distillation'
    )

    # =========================================================
    # Data
    # =========================================================
    parser.add_argument(
        '--dataset',
        type=str,
        default='CIFAR10'
    )

    parser.add_argument(
        '--percentage',
        type=int,
        default=1,
        help='percentage of total dataset used as synthetic set'
    )

    parser.add_argument(
        '--data_path',
        type=str,
        default='/home/mmoslem3/scratch/data'
    )

    # =========================================================
    # Distillation
    # =========================================================
    parser.add_argument(
        '--Iteration',
        type=int,
        default=100,
        help='number of fresh temporary encoders'
    )

    parser.add_argument(
        '--lr_img',
        type=float,
        default=0.01
    )

    parser.add_argument(
        '--syn_steps_per_encoder',
        type=int,
        default=1,
        help='number of synthetic updates for each temporary encoder'
    )

    # =========================================================
    # Network
    # =========================================================
    parser.add_argument(
        '--model',
        type=str,
        default='ConvNet'
    )

    parser.add_argument(
        '--lr_net',
        type=float,
        default=0.01
    )

    parser.add_argument(
        '--batch_train',
        type=int,
        default=256
    )

    # =========================================================
    # SSL
    # =========================================================
    parser.add_argument(
        '--ssl_method',
        type=str,
        default='simclr',
        help='simclr/barlowtwins'
    )

    parser.add_argument(
        '--epoch_ssl_train',
        type=int,
        default=1,
        help='SSL epochs on real data for each temporary encoder'
    )

    parser.add_argument(
        '--ssl_aug_strategy',
        type=str,
        default='color_crop_cutout_flip_scale_rotate'
    )

    parser.add_argument(
        '--projection_dim',
        type=int,
        default=128
    )

    parser.add_argument(
        '--temperature',
        type=float,
        default=0.5
    )

    parser.add_argument(
        '--barlow_lambda',
        type=float,
        default=0.005
    )

    # =========================================================
    # Optimal Transport
    # =========================================================
    parser.add_argument(
        '--ot_lambda',
        type=float,
        default=0.01
    )

    parser.add_argument(
        '--sinkhorn_iterations',
        type=int,
        default=100
    )

    # =========================================================
    # Evaluation
    # =========================================================
    parser.add_argument(
        '--eval_mode',
        type=str,
        default='S'
    )

    parser.add_argument(
        '--num_eval',
        type=int,
        default=1
    )

    parser.add_argument(
        '--epoch_eval_train',
        type=int,
        default=1000
    )

    parser.add_argument(
        '--label_percentage',
        type=float,
        default=1.0
    )

    parser.add_argument(
        '--epoch_linear_train',
        type=int,
        default=100
    )

    parser.add_argument(
        '--lr_linear',
        type=float,
        default=0.1
    )

    parser.add_argument(
        '--batch_linear',
        type=int,
        default=256
    )

    parser.add_argument(
        '--eval_interval',
        type=int,
        default=5
    )

    # =========================================================
    # Output
    # =========================================================
    parser.add_argument(
        '--save_path',
        type=str,
        default='result'
    )

    args = parser.parse_args()

    args.method = 'OT_Soft'
    args.device = 'cuda'
    args.dsa_param = ParamDiffAug()
    args.dsa = False

    os.makedirs(
        args.data_path,
        exist_ok=True
    )

    os.makedirs(
        args.save_path,
        exist_ok=True
    )

    # =========================================================
    # Dataset
    # =========================================================
    (
        channel,
        im_size,
        num_classes,
        class_names,
        mean,
        std,
        dst_train,
        dst_test,
        testloader
    ) = get_dataset(
        args.dataset,
        args.data_path
    )

    num_syn = int(
        len(dst_train)
        * args.percentage
        / 100
    )

    model_eval_pool = get_eval_pool(
        args.eval_mode,
        args.model,
        args.model
    )

    accs_all_exps = {
        key: []
        for key in model_eval_pool
    }

    print(
        'Hyper-parameters:\n',
        args.__dict__
    )

    # =========================================================
    # Load real dataset into memory
    # =========================================================
    images_all = [
        torch.unsqueeze(
            dst_train[i][0],
            dim=0
        )
        for i in range(len(dst_train))
    ]

    images_all = torch.cat(
        images_all,
        dim=0
    ).to(args.device)

    def get_images(n):

        idx_shuffle = np.random.permutation(
            len(images_all)
        )[:n]

        return images_all[
            idx_shuffle
        ]

    # =========================================================
    # Initialize synthetic images from real images
    # =========================================================
    image_syn = torch.randn(
        size=(
            num_syn,
            channel,
            im_size[0],
            im_size[1]
        ),
        dtype=torch.float,
        requires_grad=True,
        device=args.device
    )

    image_syn.data = get_images(
        num_syn
    ).detach().data

    image_syn_init = (
        image_syn
        .detach()
        .clone()
    )

    diag_mean = torch.tensor(
        mean,
        device=args.device
    ).view(
        1,
        channel,
        1,
        1
    )

    diag_std = torch.tensor(
        std,
        device=args.device
    ).view(
        1,
        channel,
        1,
        1
    )

    image_syn_init_uint8 = (
        (
            (
                image_syn_init
                * diag_std
                + diag_mean
            )
            * 255
            + 0.5
        )
        .clamp(0, 255)
        .to(torch.uint8)
    )

    # =========================================================
    # Synthetic-image optimizer
    # =========================================================
    optimizer_img = torch.optim.Adam(
        [image_syn],
        lr=args.lr_img
    )

    # =========================================================
    # Evaluation helper
    # =========================================================
    def evaluate_current_synthetic(iteration, record_final=False):

        print(
            '-------------------------\n'
            'Evaluation\n'
            'model_train = %s, iteration = %d'
            % (
                args.model,
                iteration
            )
        )

        for model_eval in model_eval_pool:

            print(
                'model_eval = %s'
                % model_eval
            )

            accs = []

            for it_eval in range(args.num_eval):

                net_eval = get_network(
                    model_eval,
                    channel,
                    num_classes,
                    im_size
                ).to(args.device)

                image_syn_eval = (
                    image_syn
                    .detach()
                    .clone()
                )

                _, acc_train, acc_test = evaluate_synset_SSL(
                    it_eval,
                    net_eval,
                    image_syn_eval,
                    dst_train,
                    testloader,
                    args
                )

                accs.append(
                    acc_test
                )

            print(
                'Evaluate %d random %s, '
                'mean = %.4f std = %.4f\n'
                '-------------------------'
                % (
                    len(accs),
                    model_eval,
                    np.mean(accs),
                    np.std(accs)
                )
            )

            if record_final:
                accs_all_exps[
                    model_eval
                ] += accs

    # =========================================================
    # IMPORTANT:
    # Evaluate untouched initialization first.
    # =========================================================
    print(
        '\n========================================'
    )

    print(
        'INITIAL RANDOM-SUBSET EVALUATION'
    )

    print(
        '========================================\n'
    )

    evaluate_current_synthetic(
        iteration=0,
        record_final=False
    )

    # =========================================================
    # Distillation
    # =========================================================
    print(
        '%s training begins'
        % get_time()
    )

    for it in range(
        1,
        args.Iteration + 1
    ):

        # =====================================================
        # Fresh temporary encoder
        # =====================================================
        net = get_network(
            args.model,
            channel,
            num_classes,
            im_size
        ).to(args.device)

        net.train()

        # =====================================================
        # Train temporary encoder briefly on REAL DATA
        # =====================================================
        dst_ssl_train = (
            torch.utils.data.TensorDataset(
                images_all.detach()
            )
        )

        trainloader_ssl = (
            torch.utils.data.DataLoader(
                dst_ssl_train,
                batch_size=args.batch_train,
                shuffle=True,
                num_workers=0
            )
        )

        projector, optimizer_net = init_ssl(
            net,
            images_all.detach(),
            args
        )

        ssl_losses = []

        for _ in range(
            args.epoch_ssl_train
        ):

            ssl_loss = epoch_ssl(
                trainloader_ssl,
                net,
                projector,
                optimizer_net,
                args
            )

            ssl_losses.append(
                ssl_loss
            )

        # =====================================================
        # Freeze this representation space
        # =====================================================
        for param in (
            list(net.parameters())
            +
            list(projector.parameters())
        ):
            param.requires_grad = False

        net.eval()
        projector.eval()

        embed = (
            net.module.embed
            if torch.cuda.device_count() > 1
            else net.embed
        )

        # =====================================================
        # Encode all REAL data once
        # =====================================================
        real_features = []

        with torch.no_grad():

            for start in range(
                0,
                len(images_all),
                args.batch_train
            ):

                end = min(
                    start + args.batch_train,
                    len(images_all)
                )

                real_features.append(
                    embed(
                        images_all[start:end]
                    )
                )

        output_real = torch.cat(
            real_features,
            dim=0
        )

        # =====================================================
        # Synthetic update(s)
        # =====================================================
        loss_avg = 0.0

        for syn_step in range(
            args.syn_steps_per_encoder
        ):

            # ---------------------------------------------
            # Synthetic representations with gradients
            # ---------------------------------------------
            output_syn = embed(
                image_syn
            )

            # ---------------------------------------------
            # OT target.
            #
            # output_syn is detached here because the OT
            # plan is treated as a fixed pseudo-target.
            # ---------------------------------------------
            transport_plan = get_transport_plan(
                output_real,
                output_syn.detach(),
                args
            )

            # ---------------------------------------------
            # New soft-assignment loss
            # ---------------------------------------------
            loss = transport_soft_assignment_loss(
                output_real,
                output_syn,
                transport_plan,
                args
            )

            optimizer_img.zero_grad()

            loss.backward()

            optimizer_img.step()

            loss_avg += loss.item()

        loss_avg /= (
            args.syn_steps_per_encoder
        )

        # =====================================================
        # Diagnostics
        # =====================================================
        with torch.no_grad():

            image_syn_uint8 = (
                (
                    (
                        image_syn
                        * diag_std
                        + diag_mean
                    )
                    * 255
                    + 0.5
                )
                .clamp(0, 255)
                .to(torch.uint8)
            )

            pixel_changed = (
                (
                    image_syn_uint8
                    != image_syn_init_uint8
                )
                .float()
                .mean()
                .item()
                * 100
            )

        print(
            '%s iter = %05d, '
            'ssl loss = %.6f, '
            'distill loss = %.6f, '
            'PNG values changed = %.5f%%'
            % (
                get_time(),
                it,
                np.mean(ssl_losses),
                loss_avg,
                pixel_changed
            ),
            flush=True
        )

        # =====================================================
        # Free temporary encoder before evaluation
        # =====================================================
        del output_real
        del output_syn
        del transport_plan
        del real_features
        del embed
        del projector
        del optimizer_net
        del trainloader_ssl
        del dst_ssl_train
        del net

        torch.cuda.empty_cache()

        # =====================================================
        # Evaluation
        # =====================================================
        if (
            it % args.eval_interval == 0
            or it == args.Iteration
        ):

            evaluate_current_synthetic(
                iteration=it,
                record_final=(
                    it == args.Iteration
                )
            )

        # =====================================================
        # Visualization
        # =====================================================
        if (
            it % args.eval_interval == 0
            or it == args.Iteration
        ):

            save_name = os.path.join(
                args.save_path,
                'vis_%s_%s_%s_%dpercent_iter%d.png'
                % (
                    args.method,
                    args.dataset,
                    args.model,
                    args.percentage,
                    it
                )
            )

            image_syn_vis = (
                image_syn
                .detach()
                .cpu()
                .clone()
            )

            for ch in range(channel):

                image_syn_vis[:, ch] = (
                    image_syn_vis[:, ch]
                    * std[ch]
                    + mean[ch]
                )

            image_syn_vis[
                image_syn_vis < 0
            ] = 0.0

            image_syn_vis[
                image_syn_vis > 1
            ] = 1.0

            save_image(
                image_syn_vis,
                save_name,
                nrow=int(
                    np.ceil(
                        np.sqrt(num_syn)
                    )
                )
            )

        # =====================================================
        # Final save
        # =====================================================
        if it == args.Iteration:

            data_save = (
                image_syn
                .detach()
                .cpu()
                .clone()
            )

            torch.save(
                {
                    'data': data_save
                },
                os.path.join(
                    args.save_path,
                    'res_OT-SSL_%s_%s_%dpercent.pt'
                    % (
                        args.dataset,
                        args.model,
                        args.percentage
                    )
                )
            )

    # =========================================================
    # Final results
    # =========================================================
    print(
        '\n==================== '
        'Final Results '
        '====================\n'
    )

    for key in model_eval_pool:

        accs = accs_all_exps[
            key
        ]

        print(
            'Train on %s, evaluate %d random %s, '
            'mean = %.2f%% std = %.2f%%'
            % (
                args.model,
                len(accs),
                key,
                np.mean(accs) * 100,
                np.std(accs) * 100
            )
        )


if __name__ == '__main__':
    main()