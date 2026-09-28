import argparse
import math
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18
from tqdm import tqdm

PAPER_DEFAULTS = {
    "epochs": 1000,
    "batch_size": 256,
    "lr": 0.01,
    "weight_decay": 1e-6,
    "proj_dim": 1024,
    "lambd": 0.0078125,
    "warmup_epochs": 10,
}


def build_backbone():
    """ResNet-18 with the CIFAR stem used by the referenced SSL setup."""
    model = resnet18()
    model.fc = nn.Identity()
    model.conv1 = nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False
    )
    model.maxpool = nn.Identity()
    return model


class Projector(nn.Module):
    """Two-layer projection head from the referenced ResNet-18 setup."""

    def __init__(self, in_dim=512, feature_dim=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, feature_dim, bias=True),
        )

    def forward(self, x):
        return self.net(x)


def off_diagonal(x):
    rows, cols = x.shape
    if rows != cols:
        raise ValueError("cross-correlation matrix must be square")
    return x.flatten()[:-1].view(rows - 1, rows + 1)[:, 1:].flatten()


class BarlowTwins(nn.Module):
    def __init__(self, proj_dim=1024, lambd=0.0078125):
        super().__init__()
        self.backbone = build_backbone()
        self.projector = Projector(512, proj_dim)
        self.lambd = lambd

    def forward(self, view_1, view_2):
        z1 = F.normalize(self.projector(self.backbone(view_1)), dim=-1)
        z2 = F.normalize(self.projector(self.backbone(view_2)), dim=-1)

        z1 = (z1 - z1.mean(0)) / z1.std(0).clamp_min(1e-6)
        z2 = (z2 - z2.mean(0)) / z2.std(0).clamp_min(1e-6)
        cross_correlation = (z1.T @ z2) / z1.shape[0]

        on_diagonal = torch.diagonal(cross_correlation).add(-1).pow(2).sum()
        off_diagonal_loss = off_diagonal(cross_correlation).pow(2).sum()
        return on_diagonal + self.lambd * off_diagonal_loss


class TwoViewTransform:
    """Create the two independently augmented views used by Barlow Twins."""

    def __init__(self, mean, std):
        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(32),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    def __call__(self, image):
        return self.transform(image), self.transform(image)


class ViewsOnlyDataset(torch.utils.data.Dataset):
    """Drop CIFAR class labels because SSL training needs only two views."""

    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        views, _ = self.base[index]
        return views


class TensorTwoCrops(torch.utils.data.Dataset):
    """Compatibility path for non-CIFAR datasets already returning tensors."""

    def __init__(self, base, image_size):
        self.base = base
        self.augmentation = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    image_size, scale=(0.2, 1.0), antialias=True
                ),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.4, 0.4, 0.2, 0.0)], p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
            ]
        )

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        image = self.base[index][0]
        return self.augmentation(image), self.augmentation(image)


def build_ssl_dataset(dataset, data_path):
    """Build paper-aligned CIFAR input data without using class labels."""
    if dataset == "CIFAR10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        return ViewsOnlyDataset(
            datasets.CIFAR10(
                data_path,
                train=True,
                download=True,
                transform=TwoViewTransform(mean, std),
            )
        )

    if dataset == "CIFAR100":
        mean = (0.5071, 0.4865, 0.4409)
        std = (0.2009, 0.1984, 0.2023)
        return ViewsOnlyDataset(
            datasets.CIFAR100(
                data_path,
                train=True,
                download=True,
                transform=TwoViewTransform(mean, std),
            )
        )

    # Preserve support for the repository's other/custom dataset keys. The
    # paper-aligned teacher.sh intentionally launches only CIFAR10/CIFAR100.
    from utils import get_dataset

    _, image_size, _, train_data, _ = get_dataset(dataset, data_path)
    return TensorTwoCrops(train_data, image_size[0])


def learning_rate_at_step(
    step, total_steps, warmup_steps, base_learning_rate, final_ratio=0.001
):
    """Linear warmup followed by cosine decay to 0.1% of the base LR."""
    if warmup_steps > 0 and step <= warmup_steps:
        return base_learning_rate * step / warmup_steps

    decay_steps = max(total_steps - warmup_steps, 1)
    progress = min(max((step - warmup_steps) / decay_steps, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_learning_rate * (final_ratio + (1.0 - final_ratio) * cosine)


def save_atomic(payload, path):
    temporary_path = path + ".tmp"
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)


def checkpoint_names(checkpoint_dir, dataset):
    checkpoint_base = {
        "CIFAR10": "cifar10",
        "CIFAR100": "cifar100",
        "Tiny": "tinyimagenet",
    }.get(dataset, dataset.lower())
    backbone_path = os.path.join(
        checkpoint_dir, f"barlow_twins_resnet18_{checkpoint_base}.pt"
    )
    training_path = os.path.join(
        checkpoint_dir, f"barlow_twins_resnet18_{checkpoint_base}_training.pt"
    )
    return backbone_path, training_path


def main(args):
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.batch_size <= 1:
        raise ValueError("--batch_size must be greater than 1")
    if args.warmup_epochs < 0 or args.warmup_epochs >= args.epochs:
        raise ValueError("--warmup_epochs must be at least 0 and less than --epochs")
    if args.checkpoint_every <= 0:
        raise ValueError("--checkpoint_every must be positive")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        device = torch.device(f"cuda:{args.device}")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")

    ssl_dataset = build_ssl_dataset(args.dataset, args.data_path)
    loader = DataLoader(
        ssl_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    if len(loader) == 0:
        raise ValueError(
            f"dataset has fewer than batch_size={args.batch_size} examples"
        )

    model = BarlowTwins(proj_dim=args.proj_dim, lambd=args.lambd).to(device)
    scaled_learning_rate = args.lr * args.batch_size / 256
    optimizer = torch.optim.Adam(
        model.parameters(), lr=scaled_learning_rate, weight_decay=args.weight_decay
    )

    os.makedirs(args.ckpt_dir, exist_ok=True)
    backbone_path, training_path = checkpoint_names(args.ckpt_dir, args.dataset)

    start_epoch = 0
    if args.resume:
        if not os.path.exists(training_path):
            print(
                f"--resume requested, but no full checkpoint exists at {training_path}; "
                "starting from scratch"
            )
        else:
            checkpoint = torch.load(training_path, map_location=device)
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            start_epoch = int(checkpoint["epoch"])
            print(f"resuming {args.dataset} after epoch {start_epoch}: {training_path}")

    print("Resolved teacher configuration:")
    print(f"  dataset:          {args.dataset}")
    print(f"  examples:         {len(ssl_dataset)}")
    print(f"  device:           {device}")
    print(f"  epochs:           {args.epochs}")
    print(f"  batch size:       {args.batch_size}")
    print(f"  base LR:          {scaled_learning_rate}")
    print(f"  warmup epochs:    {args.warmup_epochs}")
    print("  schedule:         cosine")
    print(f"  weight decay:     {args.weight_decay}")
    print(f"  projection dim:   {args.proj_dim}")
    print("  representation:   512")
    print(f"  Barlow lambda:    {args.lambd}")

    if start_epoch >= args.epochs:
        print(
            f"checkpoint is already at epoch {start_epoch}, "
            f"which satisfies --epochs {args.epochs}"
        )
        print("teacher backbone:", backbone_path)
        return

    total_steps = args.epochs * len(loader)
    warmup_steps = args.warmup_epochs * len(loader)

    model.train()
    for epoch in range(start_epoch, args.epochs):
        running_loss = 0.0
        seen = 0
        progress = tqdm(loader, desc=f"{args.dataset} epoch {epoch + 1}/{args.epochs}")

        for batch_index, (view_1, view_2) in enumerate(progress):
            global_step = epoch * len(loader) + batch_index + 1
            current_lr = learning_rate_at_step(
                global_step,
                total_steps,
                warmup_steps,
                scaled_learning_rate,
            )
            for group in optimizer.param_groups:
                group["lr"] = current_lr

            view_1 = view_1.to(device, non_blocking=True)
            view_2 = view_2.to(device, non_blocking=True)
            loss = model(view_1, view_2)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            batch_examples = view_1.shape[0]
            running_loss += loss.item() * batch_examples
            seen += batch_examples
            progress.set_postfix(
                loss=f"{running_loss / max(seen, 1):.4f}",
                lr=f"{current_lr:.6g}",
            )

        completed_epoch = epoch + 1
        should_save = (
            completed_epoch % args.checkpoint_every == 0
            or completed_epoch == args.epochs
        )
        if should_save:
            save_atomic(model.backbone.state_dict(), backbone_path)
            save_atomic(
                {
                    "epoch": completed_epoch,
                    "dataset": args.dataset,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "args": vars(args),
                },
                training_path,
            )
            print(
                f"[epoch {completed_epoch}/{args.epochs}] saved backbone: "
                f"{backbone_path}"
            )

    print("Training complete.")
    print("Teacher backbone:", backbone_path)
    print(
        "get_target_rep.py will find this checkpoint automatically when "
        f"called with --dataset {args.dataset} --ckpt_dir {args.ckpt_dir}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the Barlow Twins ResNet-18 teacher used by MKDT"
    )
    parser.add_argument("--dataset", default="CIFAR10")
    parser.add_argument("--data_path", default="./data")
    parser.add_argument("--ckpt_dir", default="./krrst_teacher_ckpt")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=PAPER_DEFAULTS["epochs"])
    parser.add_argument(
        "--batch_size", type=int, default=PAPER_DEFAULTS["batch_size"]
    )
    parser.add_argument("--lr", type=float, default=PAPER_DEFAULTS["lr"])
    parser.add_argument(
        "--weight_decay", type=float, default=PAPER_DEFAULTS["weight_decay"]
    )
    parser.add_argument(
        "--proj_dim", type=int, default=PAPER_DEFAULTS["proj_dim"]
    )
    parser.add_argument(
        "--lambd", type=float, default=PAPER_DEFAULTS["lambd"]
    )
    parser.add_argument(
        "--warmup_epochs", type=int, default=PAPER_DEFAULTS["warmup_epochs"]
    )
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint_every", type=int, default=10)
    main(parser.parse_args())
