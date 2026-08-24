import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torchvision.models import resnet18
from torchvision import transforms

from utils import get_dataset


# ---------------------------------------------------------------------------
# Backbone: identical architecture to get_target_rep.build_barlow_twins_teacher
# so the saved state_dict loads there with strict=True.
# ---------------------------------------------------------------------------
def build_backbone():
    m = resnet18()
    m.fc = nn.Identity()
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
    m.maxpool = nn.Identity()
    return m


class Projector(nn.Module):
    def __init__(self, in_dim=512, dim=2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, dim), nn.BatchNorm1d(dim), nn.ReLU(inplace=True),
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(inplace=True),
            nn.Linear(dim, dim, bias=False),
        )

    def forward(self, x):
        return self.net(x)


class BarlowTwins(nn.Module):
    def __init__(self, proj_dim=2048, lambd=0.005):
        super().__init__()
        self.backbone = build_backbone()
        self.projector = Projector(512, proj_dim)
        self.lambd = lambd

    def forward(self, y1, y2):
        z1 = self.projector(self.backbone(y1))
        z2 = self.projector(self.backbone(y2))

        # normalize each dimension across the batch
        z1 = (z1 - z1.mean(0)) / (z1.std(0) + 1e-6)
        z2 = (z2 - z2.mean(0)) / (z2.std(0) + 1e-6)

        n = z1.shape[0]
        c = (z1.T @ z2) / n  # cross-correlation matrix, DxD

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off = c.clone()
        off.diagonal().zero_()
        off_diag = off.pow_(2).sum()
        return on_diag + self.lambd * off_diag


# ---------------------------------------------------------------------------
# Two-view augmentation. COBRA datasets already return normalized tensors, so
# these transforms operate on tensors. Hue is disabled because it requires a
# [0,1] range that normalized tensors do not satisfy; the remaining ops are
# scale-invariant and safe on arbitrary floats.
# ---------------------------------------------------------------------------
class TwoCrops(torch.utils.data.Dataset):
    def __init__(self, base, im_size):
        self.base = base
        self.aug = transforms.Compose([
            transforms.RandomResizedCrop(im_size, scale=(0.2, 1.0), antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.0)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(3)], p=0.5),
        ])

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        img = self.base[i][0]
        return self.aug(img), self.aug(img)


def main(args):
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    torch.backends.cudnn.benchmark = True

    _, im_size, _, dst_train, _ = get_dataset(dataset=args.dataset, data_path=args.data_path)
    crop = im_size[0]
    ssl_set = TwoCrops(dst_train, crop)
    loader = torch.utils.data.DataLoader(
        ssl_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True)

    model = BarlowTwins(proj_dim=args.proj_dim, lambd=args.lambd).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    out_path = os.path.join(args.ckpt_dir, f"barlow_twins_resnet18_{args.dataset}.pt")
    status_path = os.path.join(args.ckpt_dir, "status.txt")

    start_epoch = 0
    if args.resume:
        last_epoch = None
        if os.path.exists(status_path):
            prefix = f"dataset={args.dataset} epoch="
            with open(status_path) as f:
                for line in f:
                    if line.startswith(prefix):
                        last_epoch = int(line[len(prefix):].split("/")[0])
        if last_epoch is not None and os.path.exists(out_path):
            model.backbone.load_state_dict(torch.load(out_path, map_location=device))
            start_epoch = last_epoch
            print(f"resuming {args.dataset} from epoch {start_epoch}:", out_path)
        else:
            print(f"--resume set but no checkpoint found for {args.dataset}, starting from scratch")

    model.train()
    for ep in range(start_epoch, args.epochs):
        running, seen = 0.0, 0
        pbar = tqdm(loader, desc=f"epoch {ep + 1}/{args.epochs}")
        for y1, y2 in pbar:
            y1 = y1.to(device, non_blocking=True)
            y2 = y2.to(device, non_blocking=True)
            loss = model(y1, y2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item() * y1.size(0)
            seen += y1.size(0)
            pbar.set_postfix(loss=running / max(seen, 1))

        if (ep + 1) % 10 == 0 or (ep + 1) == args.epochs:
            torch.save(model.backbone.state_dict(), out_path)
            with open(status_path, "a") as f:
                f.write(f"dataset={args.dataset} epoch={ep + 1}/{args.epochs}\n")
            print(f"[epoch {ep + 1}/{args.epochs}] saved teacher backbone:", out_path)

    print("pass this to get_target_rep.py with --teacher_ckpt", out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a Barlow Twins teacher for MKDT')
    parser.add_argument('--dataset', type=str, default='CIFAR10_S_90')
    parser.add_argument('--data_path', type=str, default='/home/mmoslem3/scratch/data')
    parser.add_argument('--ckpt_dir', type=str, default='./krrst_teacher_ckpt')
    parser.add_argument('--resume', action='store_true', help='resume from saved checkpoint/status for this dataset')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--proj_dim', type=int, default=2048)
    parser.add_argument('--lambd', type=float, default=0.005)
    parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args()
    main(args)
