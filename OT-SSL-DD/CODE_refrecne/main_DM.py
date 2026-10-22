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


# random , test acc = 0.3993



def main():

    parser = argparse.ArgumentParser(description='Parameter Processing')

    # -------------------- Data --------------------
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--percentage', type=int, default=2, help='percentage of total data size')
    parser.add_argument('--data_path', type=str, default='/home/mmoslem3/scratch/data', help='dataset path')

    # -------------------- Distillation --------------------
    parser.add_argument('--Iteration', type=int, default=10, help='training iterations')
    parser.add_argument('--lr_img', type=float, default=0.01, help='learning rate for updating synthetic images')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')

    # -------------------- Network --------------------
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--lr_net', type=float, default=0.01, help='lewarning rate for updating network parameters')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')

    # -------------------- Self-Supervised Learning --------------------
    parser.add_argument('--ssl_method', type=str, default='simclr', help='simclr/barlowtwins')
    parser.add_argument('--ssl_train_data', type=str, default='synthetic', help='real/synthetic')
    parser.add_argument('--epoch_ssl_train', type=int, default=20, help='epochs to train the temporary SSL network')
    parser.add_argument('--ssl_aug_strategy', type=str, default='color_crop_cutout_flip_scale_rotate', help='augmentation strategy for SSL training')
    parser.add_argument('--projection_dim', type=int, default=64, help='projection dimension for SSL training')
    parser.add_argument('--temperature', type=float, default=0.5, help='temperature for SimCLR')
    parser.add_argument('--barlow_lambda', type=float, default=0.005, help='off-diagonal weight for Barlow Twins')

    # -------------------- Optimal Transport --------------------
    parser.add_argument('--ot_lambda', type=float, default=0.1, help='entropy regularization for the transport plan')
    parser.add_argument('--sinkhorn_iterations', type=int, default=25, help='number of Sinkhorn iterations')

    # -------------------- Evaluation --------------------
    parser.add_argument('--eval_mode', type=str, default='S', help='eval_mode') # S: the same to training model, M: multi architectures,  W: net width, D: net depth, A: activation function, P: pooling layer, N: normalization layer,
    parser.add_argument('--num_eval', type=int, default=1, help='the number of evaluating randomly initialized models')
    parser.add_argument('--epoch_eval_train', type=int, default=1000, help='epochs to train a model with synthetic data') # it can be small for speeding up with little performance drop
    parser.add_argument('--label_percentage', type=float, default=1.0, help='percentage of labeled data for linear probing')
    parser.add_argument('--epoch_linear_train', type=int, default=100, help='epochs to train the linear probe')
    parser.add_argument('--lr_linear', type=float, default=0.1, help='learning rate for the linear probe')
    parser.add_argument('--batch_linear', type=int, default=256, help='batch size for the linear probe')

    # -------------------- Output --------------------
    parser.add_argument('--save_path', type=str, default='result2', help='path to save results')

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


    ''' training '''
    optimizer_img = torch.optim.Adam([image_syn, ], lr=args.lr_img) # optimizer_img for synthetic data
    optimizer_img.zero_grad()
    print('%s training begins'%get_time())
    print('synthetic images = %d, uniform contrastive loss = %.8f, image leaf = %s, requires_grad = %s, optimizer owns image = %s' % (num_syn, np.log(num_syn), image_syn.is_leaf, image_syn.requires_grad, optimizer_img.param_groups[0]['params'][0] is image_syn))

    
    
    # ''' Evaluate synthetic data '''
    # for model_eval in model_eval_pool:
    #     print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d'%(args.model, model_eval, 0))

    #     accs = []
    #     for it_eval in range(1):
    #         net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device) # get a random model
    #         image_syn_eval = copy.deepcopy(image_syn.detach()) # avoid any unaware modification
    #         _, acc_train, acc_test = evaluate_synset_SSL(it_eval, net_eval, image_syn_eval, dst_train, testloader, args)
    #         accs.append(acc_test)
    #     print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------'%(len(accs), model_eval, np.mean(accs), np.std(accs)))


    
    for it in range(args.Iteration+1):

        ''' Train synthetic data '''
        net = get_network(args.model, channel, num_classes, im_size).to(args.device) # get a random model
        net.train()

        if args.ssl_train_data == 'real':
            images_ssl_train = copy.deepcopy(images_all.detach())
        elif args.ssl_train_data == 'synthetic':
            images_ssl_train = copy.deepcopy(image_syn.detach())
        else:
            exit('unknown SSL training data: %s'%args.ssl_train_data)

        dst_ssl_train = torch.utils.data.TensorDataset(images_ssl_train)
        trainloader_ssl = torch.utils.data.DataLoader(dst_ssl_train, batch_size=args.batch_train, shuffle=True, num_workers=0)
        projector, optimizer_net = init_ssl(net, images_ssl_train, args)
        losses_ssl = []
        for il in range(args.epoch_ssl_train):
            losses_ssl.append(epoch_ssl(trainloader_ssl, net, projector, optimizer_net, args))

        for param in list(net.parameters()):
            param.requires_grad = False
        net.eval()

        embed = net.module.embed if torch.cuda.device_count() > 1 else net.embed # for GPU parallel

        loss_avg = 0

        ''' update synthetic data '''
        img_real = get_images(args.batch_real)
        img_syn = image_syn.reshape((num_syn, channel, im_size[0], im_size[1]))


        output_real = embed(img_real).detach()
        output_syn = embed(img_syn)
        if output_syn.requires_grad:
            output_syn.retain_grad()
        transport_plan = get_transport_plan(output_real, output_syn.detach(), args)
        loss = transport_contrastive_loss(output_real, output_syn, transport_plan, args)



        image_syn_before = image_syn.detach().clone()
        optimizer_img.zero_grad()
        loss.backward()
        optimizer_img.step()
        loss_avg += loss.item()


        if it%1 == 0:
            with torch.no_grad():
                output_syn_after = embed(image_syn)
                loss_after = transport_contrastive_loss(output_real, output_syn_after, transport_plan, args).item()
                real_normalized = F.normalize(output_real, dim=1)
                syn_normalized = F.normalize(output_syn.detach(), dim=1)
                similarity = torch.mm(real_normalized, syn_normalized.t())
                log_probability = F.log_softmax(similarity / args.temperature, dim=1)
                probability = log_probability.exp()
                row_mass = transport_plan.sum(dim=1, keepdim=True)
                col_mass = transport_plan.sum(dim=0)
                target_probability = transport_plan / row_mass.clamp_min(1e-30)
                log_target = target_probability.clamp_min(1e-30).log()
                target_entropy = -(target_probability * log_target).sum(dim=1).mean()
                prediction_entropy = -(probability * log_probability).sum(dim=1).mean()
                kl_target_prediction = (target_probability * (log_target - log_probability)).sum(dim=1).mean()
                logit_gradient = row_mass * probability - transport_plan
                step_delta = (image_syn - image_syn_before).abs()
                total_delta = (image_syn - image_syn_init).abs()
                image_syn_uint8 = ((image_syn * diag_std + diag_mean) * 255 + 0.5).clamp(0, 255).to(torch.uint8)

                print('%s iter = %05d, loss = %.10f -> %.10f, drop = %+.3e (same encoder, batch, OT plan)' % (get_time(), it, loss_avg, loss_after, loss_avg - loss_after))
                if losses_ssl:
                    print('  SSL epoch loss: first = %.6f, last = %.6f' % (losses_ssl[0], losses_ssl[-1]))
                print('  features: dim = %d, mean norm real/syn = %.3e/%.3e, normalized std real/syn = %.3e/%.3e' % (output_syn.shape[1], output_real.norm(dim=1).mean().item(), output_syn.norm(dim=1).mean().item(), real_normalized.std(dim=0, unbiased=False).mean().item(), syn_normalized.std(dim=0, unbiased=False).mean().item()))
                print('  cosine: mean = %.6f, std = %.3e, min = %.6f, max = %.6f' % (similarity.mean().item(), similarity.std(unbiased=False).item(), similarity.min().item(), similarity.max().item()))
                print('  OT: shape = %d x %d, mass = %.8f, max relative row/col error = %.3e/%.3e' % (transport_plan.shape[0], transport_plan.shape[1], transport_plan.sum().item(), (row_mass * output_real.shape[0] - 1).abs().max().item(), (col_mass * num_syn - 1).abs().max().item()))
                print('  assignments: entropy target/pred = %.6f/%.6f, mean peak target/pred = %.3e/%.3e, KL(target||pred) = %.3e' % (target_entropy.item(), prediction_entropy.item(), target_probability.max(dim=1).values.mean().item(), probability.max(dim=1).values.mean().item(), kl_target_prediction.item()))
                print('  logit grad: mean abs = %.3e, max abs = %.3e' % (logit_gradient.abs().mean().item(), logit_gradient.abs().max().item()))
                if output_syn.grad is None:
                    print('  feature grad: None')
                else:
                    print('  feature grad: mean abs = %.3e, max abs = %.3e' % (output_syn.grad.abs().mean().item(), output_syn.grad.abs().max().item()))
                if image_syn.grad is None:
                    print('  pixel grad: None')
                else:
                    print('  pixel grad: mean abs = %.3e, max abs = %.3e, L2 = %.3e, zero-grad images = %d/%d, finite = %s' % (image_syn.grad.abs().mean().item(), image_syn.grad.abs().max().item(), image_syn.grad.norm().item(), (image_syn.grad.flatten(1).abs().max(dim=1).values == 0).sum().item(), num_syn, torch.isfinite(image_syn.grad).all().item()))
                print('  pixel step: mean abs = %.3e, max abs = %.3e, changed = %.4f%%, lr = %.3e' % (step_delta.mean().item(), step_delta.max().item(), (step_delta > 0).float().mean().item() * 100, optimizer_img.param_groups[0]['lr']))
                print('  pixels from init: mean abs = %.3e, max abs = %.3e, mean abs in 0-255 units = %.3e, PNG values changed = %.4f%%' % (total_delta.mean().item(), total_delta.max().item(), (total_delta * diag_std * 255).mean().item(), (image_syn_uint8 != image_syn_init_uint8).float().mean().item() * 100), flush=True)

        if it == args.Iteration: # only record the final results
            data_save = copy.deepcopy(image_syn.detach().cpu())
            torch.save({'data': data_save, }, os.path.join(args.save_path, 'res_OT-SSL_%s_%s_%dpercent.pt'%(args.dataset, args.model, args.percentage)))


    ''' Evaluate synthetic data '''
    for model_eval in model_eval_pool:
        print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d'%(args.model, model_eval, it))

        accs = []
        for it_eval in range(args.num_eval):
            net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device) # get a random model
            image_syn_eval = copy.deepcopy(image_syn.detach()) # avoid any unaware modification
            _, acc_train, acc_test = evaluate_synset_SSL(it_eval, net_eval, image_syn_eval, dst_train, testloader, args)
            accs.append(acc_test)
        print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------'%(len(accs), model_eval, np.mean(accs), np.std(accs)))

        if it == args.Iteration: # record the final results
            accs_all_exps[model_eval] += accs


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
