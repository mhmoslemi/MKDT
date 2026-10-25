import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from utils import get_dataset, get_network, epoch, build_trainset

def main(args):

    args.device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu' 
    print('Hyper-parameters: \n', args.__dict__)

    progress_step = args.progress_percent_step
    if progress_step < 0 or progress_step > 100:
        raise ValueError('--progress_percent_step must be between 0 and 100')
    total_progress_units = args.num_experts * args.train_epochs
    completed_progress_units = 0
    next_progress = progress_step
    if progress_step:
        print('PROGRESS: 0%', flush=True)

    # Get the dataset   
    channel, im_size, _, dst_train, _ = get_dataset(dataset=args.dataset, data_path=args.data_path)

    # Location to save the buffers
    save_dir = os.path.join(args.buffer_path, args.dataset)
    save_dir = os.path.join(save_dir, args.model)
    # Expert shards are launched concurrently and share this directory.
    # exist_ok avoids the harmless check/create race between those jobs.
    os.makedirs(save_dir, exist_ok=True)

    trainloader, labels_all = build_trainset(dataset=args.dataset,
                                 dst_train=dst_train,
                                 train_labels_path=args.train_labels_path, 
                                 channel=channel, 
                                 batch_train=args.batch_train,
                                 shuffle=True,
                                 preload=not args.stream_data,
                                 num_workers=args.workers
                                )
    
    # Loss function
    criterion = None
    if args.criterion == "mse":
        criterion = nn.MSELoss().to(args.device)
    elif args.criterion == "ce":
        criterion = nn.CrossEntropyLoss().to(args.device)
    
    # Get the expert trajectories
    trajectories = []
    for it in range(0, args.num_experts):

        if args.start_index is not None:
            planned_path = os.path.join(
                save_dir, "replay_buffer_{}.pt".format(args.start_index + it))
            if os.path.exists(planned_path):
                print("Skipping existing {}".format(planned_path))
                completed_progress_units += args.train_epochs
                while (progress_step and next_progress <= 100 and
                       completed_progress_units * 100 >= next_progress * total_progress_units):
                    print(f'PROGRESS: {next_progress}%', flush=True)
                    next_progress += progress_step
                continue

        # Sample model 
        if args.seed is not None:
            seed = args.seed + (args.start_index or 0) + it
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        teacher_net = get_network(args.model, channel, labels_all.shape[1], im_size,
                                  fix_net=args.seed is not None).to(args.device)

        if it == 0:
            print(teacher_net)
        
        teacher_optim = torch.optim.SGD(teacher_net.parameters(), lr=args.lr_teacher, momentum=args.mom, weight_decay=args.l2)
        teacher_optim.zero_grad()

        timestamps = []
        timestamps.append([p.detach().cpu() for p in teacher_net.parameters()]) # Random network

        # Train on real data to get trajectory
        for e in range(args.train_epochs):

            train_loss = epoch("train", dataloader=trainloader, net=teacher_net, optimizer=teacher_optim,
                                        criterion=criterion, device=args.device)

            print(f"Itr: {it} \tEpoch: {e} \tTrain Loss: {train_loss}")

            completed_progress_units += 1
            while (progress_step and next_progress <= 100 and
                   completed_progress_units * 100 >= next_progress * total_progress_units):
                print(f'PROGRESS: {next_progress}%', flush=True)
                next_progress += progress_step

            timestamps.append([p.detach().cpu() for p in teacher_net.parameters()])

        trajectories.append(timestamps)
        
        # Save trajectory checkpoint
        print("Saving trajectory checkpoint")
        if args.start_index is None:
            n = 0
            while os.path.exists(os.path.join(save_dir, "replay_buffer_{}.pt".format(n))):
                n += 1
        else:
            n = args.start_index + it
        print("Saving {}".format(os.path.join(save_dir, "replay_buffer_{}.pt".format(n))))
        destination = os.path.join(save_dir, "replay_buffer_{}.pt".format(n))
        temporary = os.path.join(save_dir, f'.replay_buffer_{n}.{os.getpid()}.tmp.pt')
        torch.save(trajectories, temporary)
        os.replace(temporary, destination)
        trajectories = []


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Collect Trajectories Parameter Processing')
    parser.add_argument('--dataset', type=str, default='CIFAR100', help='dataset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--num_experts', type=int, default=100, help='training iterations')
    parser.add_argument('--lr_teacher', type=float, default=0.1, help='learning rate for updating network parameters')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--data_path', type=str, default='/home/data', help='dataset path')
    parser.add_argument('--buffer_path', type=str, default='./buffers_barlow_twins', help='buffer path')
    parser.add_argument('--train_epochs', type=int, default=20)
    parser.add_argument('--mom', type=float, default=0.9, help='momentum')
    parser.add_argument('--l2', type=float, default=1e-4, help='l2 regularization')
    parser.add_argument('--train_labels_path', type=str, required=True)
    parser.add_argument('--criterion', type=str, default="mse")
    parser.add_argument('--device', type=int, default=0, help='gpu number')
    parser.add_argument('--start_index', type=int, default=None, help='first explicit replay-buffer index for collision-free parallel shards')
    parser.add_argument('--seed', type=int, default=None, help='base seed plus expert index')
    parser.add_argument('--stream_data', action='store_true', help='normalize images batchwise instead of preloading all float pixels')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--progress_percent_step', type=int, default=0,
                        help='print aggregate progress at this percentage interval; 0 disables it')
    
    args = parser.parse_args()
    main(args)
