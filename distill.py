import os
import argparse
import numpy as np
import pprint
import torch
import torch.nn as nn
from tqdm import tqdm
from utils import get_dataset, get_network, get_time
import copy
import pickle 
import random
from reparam_module import ReparamModule
from datetime import datetime

def atomic_save(value, path):
    temporary = os.path.join(os.path.dirname(path), f'.{os.path.basename(path)}.{os.getpid()}.tmp.pt')
    torch.save(value, temporary)
    os.replace(temporary, path)

def main(args):
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    # Load the dataset
    channel, im_size, _, dst_train, _ = get_dataset(dataset=args.dataset, data_path=args.data_path)
    args.im_size = im_size
    print(f"Image size: {im_size}")

    args.distributed = torch.cuda.device_count() > 1

    print('Hyper-parameters: \n', args.__dict__)

    # Organize the Real Dataset
    ''' organize the real dataset '''
    
    labels_all = torch.load(args.train_labels_path, map_location="cpu", weights_only=True).detach()

    ''' initialize the synthetic data '''
    syn_lr = torch.tensor(args.lr_teacher).to(args.device)
    with open(args.image_init_idx_path, "rb") as f:
        img_init_idx = pickle.load(f)
    # Only the selected initialization pixels are needed during trajectory matching.
    image_syn = torch.stack([dst_train[int(i)][0] for i in img_init_idx])
    print("Shape of the synthetic images: ", image_syn.shape)
    label_syn = copy.deepcopy(labels_all[img_init_idx])
    print("Shape of the synthetic image labels: ", label_syn.shape)

    if args.batch_syn is None:
        args.batch_syn = len(image_syn)
        
    ''' training '''
    # The image, the labels, and the learning rate are all differentiable
    image_syn = image_syn.detach().to(args.device).requires_grad_(True)
    label_syn = label_syn.detach().to(args.device).requires_grad_(True)
    syn_lr = syn_lr.detach().to(args.device).requires_grad_(True)
    optimizer_img = torch.optim.SGD([image_syn], lr=args.lr_img, momentum=0.5)
    optimizer_labels = torch.optim.SGD([label_syn], lr=args.lr_labels, momentum=0.5)
    optimizer_lr = torch.optim.SGD([syn_lr], lr=args.lr_lr, momentum=0.5)
    optimizer_img.zero_grad()

    criterion = None
    if args.criterion == "mse":
        criterion = nn.MSELoss().to(args.device)
    elif args.criterion == "ce":
        criterion = nn.CrossEntropyLoss().to(args.device)

    # Extract the buffers
    expert_dir = args.expert_dir
    print("Expert Dir: {}".format(expert_dir))

    expert_files = []
    n = 0
    while os.path.exists(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n))):
        expert_files.append(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n)))
        n += 1
    if n == 0:
        raise AssertionError("No buffers detected at {}".format(expert_dir))
    file_idx, expert_idx = 0, 0
    expert_order = []
    loaded_file = None
    random.shuffle(expert_files)

    # A deterministic save_dir lets callers detect a completed/partial run and
    # decide whether to skip or resume; without it, fall back to a timestamped dir.
    if args.save_dir:
        save_dir = args.save_dir
    else:
        current_datetime = datetime.now()
        RUN_ID = current_datetime.strftime("%Y-%m-%d_%H:%M:%S") + str(args.run_id)
        save_dir = os.path.join(".", "logged_files", args.dataset, RUN_ID)
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "hparams.txt"), "w") as f:
        f.write(pprint.pformat(vars(args), indent=4))
        
    checkpoint_path = os.path.join(save_dir, 'training.pt')
    config = {k: v for k, v in vars(args).items()
              if k not in ('resume', 'stop_at', 'device', 'distributed', 'im_size')}
    start_it = 0
    if args.resume:
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f'Cannot resume without full state: {checkpoint_path}')
        saved = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if saved['config'] != config or set(saved['expert_files']) != set(expert_files):
            raise ValueError('Distillation recipe or expert file set changed')
        with torch.no_grad():
            image_syn.copy_(saved['images'])
            label_syn.copy_(saved['labels'])
            syn_lr.copy_(saved['synlr'])
        for optimizer, state in zip((optimizer_img, optimizer_labels, optimizer_lr), saved['optimizers']):
            optimizer.load_state_dict(state)
        start_it = saved['iteration']
        expert_files = saved['expert_files']
        file_idx, expert_idx, expert_order = saved['file_idx'], saved['expert_idx'], saved['expert_order']
        random.setstate(saved['python_rng'])
        np.random.set_state(saved['numpy_rng'])
        torch.set_rng_state(saved['torch_rng'])
        if args.device == 'cuda':
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
        print(f'Resuming complete state after iteration {start_it}', flush=True)
    elif args.checkpoint_every and os.path.exists(checkpoint_path):
        raise FileExistsError(f'Use --resume to continue {checkpoint_path}')

    def save_state(iteration, export=False):
        if export:
            for name, value in (('images', image_syn), ('labels', label_syn), ('synlr', syn_lr)):
                atomic_save(value.detach().cpu(), os.path.join(save_dir, f'{name}_{iteration}.pt'))
        if args.checkpoint_every:
            atomic_save({'config': config, 'iteration': iteration,
                'images': image_syn.detach().cpu(), 'labels': label_syn.detach().cpu(),
                'synlr': syn_lr.detach().cpu(),
                'optimizers': [o.state_dict() for o in (optimizer_img, optimizer_labels, optimizer_lr)],
                'expert_files': expert_files, 'file_idx': file_idx, 'expert_idx': expert_idx,
                'expert_order': expert_order, 'python_rng': random.getstate(),
                'numpy_rng': np.random.get_state(), 'torch_rng': torch.get_rng_state(),
                'cuda_rng': torch.cuda.get_rng_state_all() if args.device == 'cuda' else []}, checkpoint_path)

    stop_at = min(args.iters, args.stop_at or args.iters)
    if start_it > stop_at:
        raise ValueError('Saved iteration is beyond the requested stop point')
    if start_it == 0:
        save_state(0, export=True)
    print('%s training begins'%get_time())
    pbar = tqdm(range(start_it, stop_at), desc="Distilling")
    for it in pbar:
        log_it = it + 1
        
        student_net = get_network(args.model, channel, label_syn.shape[1], im_size, dist=False,
                                  fix_net=args.seed is not None).to(args.device)
        student_net = ReparamModule(student_net)

        if args.distributed:
            student_net = torch.nn.DataParallel(student_net)

        student_net.train()

        num_params = sum([np.prod(p.size()) for p in (student_net.parameters())])
        if loaded_file != expert_files[file_idx]:
            buffer = torch.load(expert_files[file_idx], map_location="cpu", weights_only=True)
            loaded_file = expert_files[file_idx]
        if not expert_order:
            expert_order = list(range(len(buffer)))
            random.shuffle(expert_order)
        expert_trajectory = buffer[expert_order[expert_idx]]
        expert_idx += 1
        if expert_idx == len(buffer):
            expert_idx = 0
            file_idx += 1
            if file_idx == len(expert_files):
                file_idx = 0
                random.shuffle(expert_files)
            expert_order = []

        start_epoch = np.random.randint(args.min_start_epoch, args.max_start_epoch)
        starting_params = expert_trajectory[start_epoch]
        target_params = expert_trajectory[start_epoch+args.expert_epochs]
        target_params = torch.cat([p.data.to(args.device).reshape(-1) for p in target_params], 0)

        student_params = [torch.cat([p.data.to(args.device).reshape(-1) for p in starting_params], 0).requires_grad_(True)]

        starting_params = torch.cat([p.data.to(args.device).reshape(-1) for p in starting_params], 0)
        y_hat = label_syn.to(args.device)

        param_loss_list = []
        param_dist_list = []
        indices_chunks = []

        for step in range(args.syn_steps):

            if not indices_chunks:
                indices = torch.randperm(len(image_syn))
                indices_chunks = list(torch.split(indices, args.batch_syn))

            these_indices = indices_chunks.pop()


            x = image_syn[these_indices]
            this_y = y_hat[these_indices]

            if args.distributed:
                forward_params = student_params[-1].unsqueeze(0).expand(torch.cuda.device_count(), -1)
            else:
                forward_params = student_params[-1]
            x = student_net(x, flat_param=forward_params)
            mse_loss = criterion(x, this_y)

            grad = torch.autograd.grad(mse_loss, student_params[-1], create_graph=True)[0]

            student_params.append(student_params[-1] - syn_lr * grad)

        param_loss = torch.tensor(0.0).to(args.device)
        param_dist = torch.tensor(0.0).to(args.device)

        param_loss += torch.nn.functional.mse_loss(student_params[-1], target_params, reduction="sum")
        param_dist += torch.nn.functional.mse_loss(starting_params, target_params, reduction="sum")

        param_loss_list.append(param_loss)
        param_dist_list.append(param_dist)


        param_loss /= num_params
        param_dist /= num_params

        param_loss /= param_dist

        grand_loss = param_loss
        if not torch.isfinite(grand_loss).item():
            raise FloatingPointError(
                f"Non-finite trajectory-matching loss at iteration {log_it}")
        optimizer_img.zero_grad()
        optimizer_labels.zero_grad()
        optimizer_lr.zero_grad()
        
        grand_loss.backward()

        for name, value in (("images", image_syn), ("labels", label_syn),
                            ("synthetic learning rate", syn_lr)):
            if value.grad is None or not torch.isfinite(value.grad).all().item():
                raise FloatingPointError(
                    f"Non-finite {name} gradient at iteration {log_it}")
        
        avg_grad = torch.mean(torch.abs(image_syn.grad)).item()
        optimizer_img.step()
        optimizer_labels.step()
        optimizer_lr.step()

        for name, value in (("images", image_syn), ("labels", label_syn),
                            ("synthetic learning rate", syn_lr)):
            if not torch.isfinite(value).all().item():
                raise FloatingPointError(
                    f"Non-finite {name} after iteration {log_it}")

        pbar.set_postfix({"Grand_Loss": grand_loss.detach().cpu().item(),
                   "Start_Epoch": start_epoch, 
                   "Avg Grad": avg_grad})
        for _ in student_params:
            del _

        export = log_it % args.eval_it == 0 or log_it == args.iters
        if export or log_it == stop_at or (args.checkpoint_every and log_it % args.checkpoint_every == 0):
            save_state(log_it, export=export)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')

    parser.add_argument('--dataset', type=str, default='CIFAR100', help='dataset')

    parser.add_argument('--model', type=str, default='ConvNet', help='model')

    parser.add_argument('--eval_it', type=int, default=1000, help='how often to evaluate')

    parser.add_argument('--iters', type=int, default=5000, help='how many distillation steps to perform')

    parser.add_argument('--lr_img', type=float, default=1000, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_lr', type=float, default=1e-04, help='learning rate for updating... learning rate')
    parser.add_argument('--lr_teacher', type=float, default=1e-1, help='initialization for synthetic learning rate')

    parser.add_argument('--batch_syn', type=int, default=256, help='should only use this if you run out of VRAM')

    parser.add_argument('--data_path', type=str, default='/data', help='dataset path')

    parser.add_argument('--expert_epochs', type=int, default=5, help='how many expert epochs the target params are') # M but in epochs from paper
    parser.add_argument('--syn_steps', type=int, default=40, help='how many steps to take on synthetic data')
    parser.add_argument('--max_start_epoch', type=int, default=5, help='max epoch we can start at')
    parser.add_argument('--min_start_epoch', type=int, default=0, help='minimum expert epoch we can start at (inclusive)')
    
    parser.add_argument('--lr_labels', type=float, default=1, help='learning rate for updating labels')
    parser.add_argument('--image_init_idx_path', type=str, help='path to the initial images')
    parser.add_argument('--train_labels_path', type=str, required=True, help='path to the target representation')
    parser.add_argument('--run_id', type=str, default=None, help='id for run')
    parser.add_argument('--save_dir', type=str, default=None, help='deterministic output dir (enables detecting a completed/partial run to resume); default is a timestamped dir under ./logged_files/{dataset}')
    parser.add_argument('--resume', action='store_true', help='restore images, labels, learned LR, optimizers, expert cursor and RNG')
    parser.add_argument('--checkpoint_every', type=int, default=0, help='replace training.pt every N completed updates; 0 disables full-state saves')
    parser.add_argument('--stop_at', type=int, default=None, help='stop this job at this update without changing the total recipe')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--expert_dir', type=str, help='dir for expert trajectories')
    parser.add_argument('--criterion', type=str, default="mse") # Should always use MSE

    args = parser.parse_args()

    if args.min_start_epoch < 0 or args.min_start_epoch >= args.max_start_epoch:
        parser.error('min_start_epoch must satisfy 0 <= min_start_epoch < max_start_epoch')

    main(args)
