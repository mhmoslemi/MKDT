#!/bin/sh

# COBRA_KEYS = [
#     "CIFAR10_S_90",
#     "Colored_MNIST_foreground",
#     "Colored_MNIST_background",
#     "Colored_FashionMNIST_foreground",
#     "Colored_FashionMNIST_background",
#     "UTKface",
#     "CelebA",
#     "BFFHQ",
# ]

DATASET="Colored_MNIST_foreground"
RESUME=""

while [ $# -gt 0 ]; do
  case "$1" in
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --resume)
      RESUME="--resume"
      shift
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

source /home/mmoslem3/ENV/bin/activate

python train_teacher.py --dataset "$DATASET" \
 --data_path /home/mmoslem3/scratch/data \
 --ckpt_dir /home/mmoslem3/scratch/MKDT/krrst_teacher_ckpt \
 --epochs 300 --batch_size 256 --resume
