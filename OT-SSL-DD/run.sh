#!/bin/bash
python Gen-dd.py --model ResNet18 --teacher_ssl simclr --teacher_seeds 0 1 2 --teacher_path /scratch/mmoslem3/MKDT/Evaluate_benchamrk/full_ssl_checkpoints_1000ep_3seeds --vae_path /scratch/mmoslem3/MKDT/OT-SSL-DD/gen_dd_cache/residual_vae_CIFAR10_z256_e400_beta0.01_seed0.pt --vae_epochs 350
