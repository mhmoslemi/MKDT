import os
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
from utils import get_loops, get_dataset, get_network, get_eval_pool, evaluate_synset, evaluate_synset_SSL, get_daparam, match_loss, get_time, TensorDataset, epoch, init_ssl, epoch_ssl, get_transport_plan, transport_contrastive_loss, DiffAugment, ParamDiffAug
from utils import transport_soft_assignment_loss


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

def main():

    parser = argparse.ArgumentParser(description='Parameter Processing')

    # -------------------- Data --------------------
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--percentage', type=int, default=1, help='percentage of total data size')
    parser.add_argument('--data_path', type=str, default='/home/mmoslem3/scratch/data', help='dataset path')

    # -------------------- Distillation --------------------
    parser.add_argument('--Iteration', type=int, default=100, help='training iterations')
    parser.add_argument('--lr_img', type=float, default=0.01, help='learning rate for updating synthetic images')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')

    # -------------------- Network --------------------
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')

    # -------------------- Self-Supervised Learning --------------------
    parser.add_argument('--ssl_method', type=str, default='simclr', help='simclr/barlowtwins')
    parser.add_argument('--ssl_train_data', type=str, default='synthetic', help='real/synthetic')
    parser.add_argument('--epoch_ssl_train', type=int, default=15, help='epochs to train the temporary SSL network')
    parser.add_argument('--ssl_aug_strategy', type=str, default='color_crop_cutout_flip_scale_rotate', help='augmentation strategy for SSL training')
    parser.add_argument('--projection_dim', type=int, default=128, help='projection dimension for SSL training')
    parser.add_argument('--temperature', type=float, default=0.5, help='temperature for SimCLR')
    parser.add_argument('--barlow_lambda', type=float, default=0.005, help='off-diagonal weight for Barlow Twins')

    # -------------------- Optimal Transport --------------------
    parser.add_argument('--ot_lambda', type=float, default=0.01, help='entropy regularization for the transport plan')
    parser.add_argument('--sinkhorn_iterations', type=int, default=100, help='number of Sinkhorn iterations')

    # -------------------- Evaluation --------------------
    parser.add_argument('--eval_mode', type=str, default='S', help='eval_mode') # S: the same to training model, M: multi architectures,  W: net width, D: net depth, A: activation function, P: pooling layer, N: normalization layer,
    parser.add_argument('--num_eval', type=int, default=1, help='the number of evaluating randomly initialized models')
    parser.add_argument('--epoch_eval_train', type=int, default=1000, help='epochs to train a model with synthetic data') # it can be small for speeding up with little performance drop
    parser.add_argument('--label_percentage', type=float, default=1.0, help='percentage of labeled data for linear probing')
    parser.add_argument('--epoch_linear_train', type=int, default=100, help='epochs to train the linear probe')
    parser.add_argument('--lr_linear', type=float, default=0.1, help='learning rate for the linear probe')
    parser.add_argument('--batch_linear', type=int, default=256, help='batch size for the linear probe')

    # -------------------- Output --------------------
    parser.add_argument('--save_path', type=str, default='result', help='path to save results')

    args = parser.parse_args()
    args.method = 'DM'
    args.device = 'cuda'
    args.dsa_param = ParamDiffAug()
    args.dsa = False

    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset, args.data_path)
    num_syn = int(len(dst_train) * args.percentage / 100)
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)


    accs_all_exps = dict() # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []



    print('Hyper-parameters: \n', args.__dict__)

    ''' organize the real dataset '''
    images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
    images_all = torch.cat(images_all, dim=0).to(args.device)

    def get_images(n):
        idx_shuffle = np.random.permutation(len(images_all))[:n]
        return images_all[idx_shuffle]

    
    ''' initialize the synthetic data '''
    image_syn = torch.randn(size=(num_syn, channel, im_size[0], im_size[1]), dtype=torch.float, requires_grad=True, device=args.device)
    image_syn.data = get_images(num_syn).detach().data
    image_syn_init = image_syn.detach().clone()
    diag_mean = torch.tensor(mean, device=args.device).view(1, channel, 1, 1)
    diag_std = torch.tensor(std, device=args.device).view(1, channel, 1, 1)
    image_syn_init_uint8 = ((image_syn_init * diag_std + diag_mean) * 255 + 0.5).clamp(0, 255).to(torch.uint8)




    
    
    ''' Evaluate synthetic data '''
    for model_eval in model_eval_pool:
        print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d'%(args.model, model_eval, 0))

        accs = []
        for it_eval in range(4):
            net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device) # get a random model
            image_syn_eval = copy.deepcopy(image_syn.detach()) # avoid any unaware modification
            _, acc_train, acc_test = evaluate_synset_SSL(it_eval, net_eval, image_syn_eval, dst_train, testloader, args)
            accs.append(acc_test)
        print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------'%(len(accs), model_eval, np.mean(accs), np.std(accs)))


    ''' training '''
    print('%s training begins'%get_time())

    


    optimizer_img = torch.optim.Adam([image_syn, ], lr=args.lr_img) # optimizer_img for synthetic data
    optimizer_img.zero_grad()
    
    for it in range(args.Iteration+1):




        # ---------------------------------------------------------
        # Fresh encoder for this outer iteration
        # ---------------------------------------------------------
        net = get_network(args.model, channel, num_classes, im_size).to(args.device)
        net.train()

        # IMPORTANT: train temporary encoder on REAL data
        images_ssl_train = copy.deepcopy(images_all.detach())

        dst_ssl_train = torch.utils.data.TensorDataset(images_ssl_train)
        trainloader_ssl = torch.utils.data.DataLoader(
            dst_ssl_train,
            batch_size=args.batch_train,
            shuffle=True,
            num_workers=0
        )

        projector, optimizer_net = init_ssl(net, images_ssl_train, args)

        for il in range(args.epoch_ssl_train):
            epoch_ssl(trainloader_ssl, net, projector, optimizer_net, args)

        # freeze temporary encoder
        for param in list(net.parameters()) + list(projector.parameters()):
            param.requires_grad = False

        net.eval()
        projector.eval()

        embed = net.module.embed if torch.cuda.device_count() > 1 else net.embed


        # ---------------------------------------------------------
        # Real features stay fixed for this temporary encoder
        # ---------------------------------------------------------
        real_features = []

        with torch.no_grad():
            for start in range(0, len(images_all), args.batch_train):
                end = min(start + args.batch_train, len(images_all))
                real_features.append(embed(images_all[start:end]))

        output_real = torch.cat(real_features, dim=0)


        # ---------------------------------------------------------
        # Optimize synthetic images IN THE SAME representation space
        # ---------------------------------------------------------
        loass_avg = 0 
        args.geometry_weight = 0.0

        for it2 in range(1):

            output_syn = embed(image_syn)

            transport_plan = get_transport_plan(
                output_real,
                output_syn.detach(),
                args
            )

            # loss = transport_barycentric_loss(
            #     output_real,
            #     output_syn,
            #     transport_plan,
            #     args
            # )

            loss = transport_soft_assignment_loss(
                    output_real,
                    output_syn,
                    transport_plan,
                    args
                )

            optimizer_img.zero_grad()
            loss.backward()
            optimizer_img.step()
            loass_avg += loss.item()

            # if it2 %10  == 0 and it2!=0:
            #     print()

            # print(round(loss.item(), 4), end=" / ")


        if it%1 == 0:
            with torch.no_grad():
                image_syn_uint8 = ((image_syn * diag_std + diag_mean) * 255 + 0.5).clamp(0, 255).to(torch.uint8)
                print(' %s iter = %05d, loss = %.10f' % (get_time(), it, loass_avg),end = '\t/\t')
                print('PNG values changed = %.5f%%' % ( (image_syn_uint8 != image_syn_init_uint8).float().mean().item() * 100), flush=True)

        if it == args.Iteration: # only record the final results
            data_save = copy.deepcopy(image_syn.detach().cpu())
            torch.save({'data': data_save, }, os.path.join(args.save_path, 'res_OT-SSL_%s_%s_%dpercent.pt'%(args.dataset, args.model, args.percentage)))


        
        if it%50==0 and it!=0:
            ''' Evaluate synthetic data '''
            for model_eval in model_eval_pool:
                print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d'%(args.model, model_eval, it))

                accs = []
                # for it_eval in range(args.num_eval):
                for it_eval in range(4):
                    net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device) # get a random model
                    image_syn_eval = copy.deepcopy(image_syn.detach()) # avoid any unaware modification
                    _, acc_train, acc_test = evaluate_synset_SSL(it_eval, net_eval, image_syn_eval, dst_train, testloader, args)
                    accs.append(acc_test)
                print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------'%(len(accs), model_eval, np.mean(accs), np.std(accs)))

                if it == args.Iteration: # record the final results
                    accs_all_exps[model_eval] += accs

        if it%20==0 and it!=0:
            ''' visualize and save '''
            save_name = os.path.join(args.save_path, 'vis_%s_%s_%s_%dpercent_iter%d.png'%(args.method, args.dataset, args.model, args.percentage, it))
            image_syn_vis = copy.deepcopy(image_syn.detach().cpu())
            for ch in range(channel):
                image_syn_vis[:, ch] = image_syn_vis[:, ch]  * std[ch] + mean[ch]
            image_syn_vis[image_syn_vis<0] = 0.0
            image_syn_vis[image_syn_vis>1] = 1.0
            save_image(image_syn_vis, save_name, nrow=int(np.ceil(np.sqrt(num_syn))))


    print('\n==================== Final Results ====================\n')
    for key in model_eval_pool:
        accs = accs_all_exps[key]
        print('Train on %s, evaluate %d random %s, mean  = %.2f%%  std = %.2f%%'%(args.model, len(accs), key, np.mean(accs)*100, np.std(accs)*100))



if __name__ == '__main__':
    main()
