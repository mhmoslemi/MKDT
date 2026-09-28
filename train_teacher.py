import argparse
import math
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from torchvision.models import resnet18

PAPER_DEFAULTS = {
    "epochs": 1000,
    "batch_size": 256,
    "lr": 0.01,
    "weight_decay": 1e-6,
    "proj_dim": 1024,
    "lambd": 0.0078125,
    "warmup_epochs": 10,
}


def build_backbone(dataset="CIFAR10"):
    """ResNet-18 with the small-image stem used by the SSL setup."""
    model = resnet18()
    model.fc = nn.Identity()
    model.conv1 = nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False
    )
    # The referenced setup removes max-pooling for 32x32 CIFAR images but
    # retains torchvision's max-pool for 64x64 Tiny ImageNet images.
    if dataset != "Tiny":
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
    def __init__(self, dataset="CIFAR10", proj_dim=1024, lambd=0.0078125):
        super().__init__()
        self.backbone = build_backbone(dataset)
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


TINY_IMAGENET_INPUT_MEAN = (0.485, 0.456, 0.406)
TINY_IMAGENET_INPUT_STD = (0.229, 0.224, 0.225)
TINY_IMAGENET_TRAIN_MEAN = (0.480, 0.448, 0.398)
TINY_IMAGENET_TRAIN_STD = (0.277, 0.269, 0.282)


def load_torch_file(path):
    """Load a trusted local dataset file across PyTorch versions."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def image_source_from_payload(payload):
    """Find an image collection or Dataset inside a common .pt payload."""
    if isinstance(payload, dict):
        for key in (
            "images",
            "image",
            "x",
            "data",
            "images_train",
            "train_images",
            "train_data",
            "dataset",
            "train",
        ):
            if key in payload:
                return image_source_from_payload(payload[key])
        if len(payload) == 1:
            return next(iter(payload.values()))
        raise ValueError(
            "could not find images in Tiny ImageNet .pt dictionary; "
            f"available keys: {sorted(map(str, payload.keys()))}"
        )

    # A common saved format is (images, labels). Do not mistake a list of
    # individual (image, label) samples for that container format.
    if isinstance(payload, tuple) and len(payload) >= 1:
        return payload[0]
    if (
        isinstance(payload, list)
        and len(payload) == 2
        and not isinstance(payload[0], (tuple, dict))
    ):
        return payload[0]
    return payload


def image_from_sample(sample):
    """Discard a class label or other sample metadata."""
    if isinstance(sample, dict):
        for key in ("image", "images", "img", "x", "data"):
            if key in sample:
                return sample[key]
        raise ValueError(
            "could not find an image in Tiny ImageNet sample dictionary; "
            f"available keys: {sorted(map(str, sample.keys()))}"
        )
    if isinstance(sample, (tuple, list)):
        if not sample:
            raise ValueError("encountered an empty Tiny ImageNet sample")
        return sample[0]
    return sample


def as_chw_tensor(image):
    """Convert PIL, NumPy, or tensor images to an unscaled CHW tensor."""
    if isinstance(image, torch.Tensor):
        tensor = image.detach().cpu()
    else:
        try:
            tensor = torch.as_tensor(image)
        except (TypeError, RuntimeError):
            tensor = TF.pil_to_tensor(image)

    if tensor.ndim == 4 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3:
        raise ValueError(
            "Tiny ImageNet images must have 2 or 3 dimensions; "
            f"got shape {tuple(tensor.shape)}"
        )
    if tensor.shape[0] not in (1, 3, 4) and tensor.shape[-1] in (1, 3, 4):
        tensor = tensor.permute(2, 0, 1)
    if tensor.shape[0] == 1:
        tensor = tensor.repeat(3, 1, 1)
    elif tensor.shape[0] == 4:
        tensor = tensor[:3]
    if tensor.shape[0] != 3:
        raise ValueError(
            "Tiny ImageNet images must have 1, 3, or 4 channels; "
            f"got shape {tuple(tensor.shape)}"
        )
    return tensor.contiguous()


class TinyImageNetTwoViewDataset(torch.utils.data.Dataset):
    """Two-view Tiny ImageNet loader for ImageFolder or serialized .pt data."""

    INPUT_FORMATS = {"auto", "uint8", "zero_one", "zero_255", "normalized"}

    def __init__(self, source, description, input_format="auto"):
        self.source = source
        self.description = description
        if input_format not in self.INPUT_FORMATS:
            raise ValueError(
                f"Tiny ImageNet input format must be one of "
                f"{sorted(self.INPUT_FORMATS)}; got {input_format}"
            )
        if not hasattr(source, "__len__") or not hasattr(source, "__getitem__"):
            raise TypeError(
                "Tiny ImageNet .pt must contain an image tensor, an "
                "(images, labels) pair, or an indexable Dataset"
            )
        if len(source) == 0:
            raise ValueError("Tiny ImageNet dataset is empty")

        self.input_format = (
            self._infer_input_format() if input_format == "auto" else input_format
        )
        self.augmentation = transforms.Compose(
            [
                transforms.RandomApply(
                    [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8
                ),
                transforms.RandomGrayscale(p=0.1),
                transforms.RandomResizedCrop(
                    64,
                    scale=(0.2, 1.0),
                    ratio=(0.75, 4.0 / 3.0),
                    interpolation=InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Normalize(
                    TINY_IMAGENET_TRAIN_MEAN, TINY_IMAGENET_TRAIN_STD
                ),
            ]
        )

    def __len__(self):
        return len(self.source)

    def _raw_image(self, index):
        return image_from_sample(self.source[index])

    def _infer_input_format(self):
        minimum = float("inf")
        maximum = float("-inf")
        saw_float = False
        for index in range(min(len(self.source), 8)):
            tensor = as_chw_tensor(self._raw_image(index))
            saw_float = saw_float or torch.is_floating_point(tensor)
            values = tensor.float()
            minimum = min(minimum, float(values.min()))
            maximum = max(maximum, float(values.max()))

        if not saw_float:
            return "uint8"
        if minimum >= 0.0 and maximum <= 1.01:
            return "zero_one"
        if minimum >= 0.0 and maximum <= 255.01:
            return "zero_255"
        return "normalized"

    def _to_zero_one(self, image):
        tensor = as_chw_tensor(image).float()
        if self.input_format in ("uint8", "zero_255"):
            tensor = tensor / 255.0
        elif self.input_format == "normalized":
            mean = tensor.new_tensor(TINY_IMAGENET_INPUT_MEAN).view(3, 1, 1)
            std = tensor.new_tensor(TINY_IMAGENET_INPUT_STD).view(3, 1, 1)
            tensor = tensor * std + mean
        return tensor.clamp_(0.0, 1.0)

    def __getitem__(self, index):
        image = self._to_zero_one(self._raw_image(index))
        return self.augmentation(image), self.augmentation(image)


def build_tiny_imagenet_dataset(data_path, data_file, input_format):
    if data_file is not None:
        data_file = os.path.abspath(os.path.expanduser(data_file))
        if not os.path.isfile(data_file):
            raise FileNotFoundError(
                f"Tiny ImageNet .pt file not found: {data_file}\n"
                "Pass the correct path with --data_file."
            )
        payload = load_torch_file(data_file)
        source = image_source_from_payload(payload)
        return TinyImageNetTwoViewDataset(source, data_file, input_format)

    candidate_roots = (
        os.path.join(data_path, "tiny_imagenet", "train"),
        os.path.join(data_path, "tiny-imagenet-200", "train"),
        os.path.join(data_path, "train"),
    )
    for train_root in candidate_roots:
        if os.path.isdir(train_root):
            source = datasets.ImageFolder(train_root)
            return TinyImageNetTwoViewDataset(source, train_root, "uint8")
    raise FileNotFoundError(
        "Tiny ImageNet data not found. Pass --data_file /path/to/tinyimagenet.pt "
        "or --data_path pointing to an extracted Tiny ImageNet directory."
    )


def build_ssl_dataset(dataset, data_path, data_file=None, tiny_input_format="auto"):
    """Build paper-aligned SSL input data without using class labels."""
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

    if dataset == "Tiny":
        return build_tiny_imagenet_dataset(
            data_path, data_file, tiny_input_format
        )

    # Preserve support for the repository's other/custom dataset keys. The
    # paper-aligned launchers intentionally use CIFAR10/CIFAR100/Tiny.
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
    if args.log_every <= 0:
        raise ValueError("--log_every must be positive")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        device = torch.device(f"cuda:{args.device}")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")

    ssl_dataset = build_ssl_dataset(
        args.dataset,
        args.data_path,
        data_file=args.data_file,
        tiny_input_format=args.tiny_input_format,
    )
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

    model = BarlowTwins(
        dataset=args.dataset, proj_dim=args.proj_dim, lambd=args.lambd
    ).to(device)
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
    if args.dataset == "Tiny":
        print(f"  data source:      {ssl_dataset.description}")
        print(f"  input format:     {ssl_dataset.input_format}")
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
    print(f"  log every:        {args.log_every} iterations")

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

        for batch_index, (view_1, view_2) in enumerate(loader):
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
            iteration = batch_index + 1
            if iteration % args.log_every == 0 or iteration == len(loader):
                print(
                    f"{args.dataset} epoch {epoch + 1}/{args.epochs} "
                    f"iteration {iteration}/{len(loader)} "
                    f"loss={running_loss / max(seen, 1):.4f} "
                    f"lr={current_lr:.6g}",
                    flush=True,
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
    parser.add_argument(
        "--data_file",
        default=None,
        help="Tiny ImageNet .pt file; overrides --data_path for dataset Tiny",
    )
    parser.add_argument(
        "--tiny_input_format",
        default="auto",
        choices=sorted(TinyImageNetTwoViewDataset.INPUT_FORMATS),
        help=(
            "encoding of images in a Tiny ImageNet .pt file; normalized means "
            "ImageNet mean/std normalization"
        ),
    )
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
    parser.add_argument("--log_every", type=int, default=50)
    main(parser.parse_args())
