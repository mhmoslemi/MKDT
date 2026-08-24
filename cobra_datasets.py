"""
cobra_datasets.py — make MKDT work on COBRA's datasets.

Setup (two copy steps, no other dependencies):
  1. Copy COBRA's `data_handler/` folder into the MKDT repo root.
  2. Put this file in the MKDT repo root.

Then apply the small patch to MKDT's utils.get_dataset shown at the
bottom of this file, and pass the new dataset keys to get_target_rep.py,
buffer.py, distill.py, eval.py via --dataset. If any script restricts
--dataset with argparse `choices=[...]`, append the keys in COBRA_KEYS
to that list.

Design notes:
  * COBRA datasets yield (image, label, sensitive) triples. MKDT expects
    (image, label) and, during distillation, replaces labels with target
    representations. The wrapper below drops the sensitive attribute for
    the MKDT pipeline but keeps it retrievable (get_sensitive_attrs) so
    fairness metrics can still be computed after linear probing.
  * COBRA loaders already bake ToTensor + Normalize into the datasets,
    so no transform is added here.
  * Model choice follows MKDT's convention: ConvNet (depth 3) for 32x32
    datasets, ConvNetD4 for 64x64 datasets (UTKface, CelebA, BFFHQ).
  * Teacher SSL checkpoints: MKDT ships none for these datasets. Either
    (a) reuse the KRRST/SAS Barlow Twins or SimCLR teacher trained on
    CIFAR (for the 32x32 keys) and on Tiny ImageNet (for the 64x64
    keys), which requires no new code, or (b) train a Barlow Twins
    teacher on each COBRA dataset for a better-matched teacher. In both
    cases get_target_rep.py runs unchanged once the checkpoint path is
    given.
"""

import torch
from torch.utils.data import Dataset

from data_handler.colour_mnist import get_biased_mnist_dataloader
from data_handler.FashionMNIST import get_biased_fashionmnist_dataloader
from data_handler import cifar10 as cobra_cifar10
from data_handler.celeba import CelebA
from data_handler.utkface import UTKFaceDataset
from data_handler.bffhq import BFFHQ


COBRA_KEYS = [
    "CIFAR10_S_90",
    "Colored_MNIST_foreground",
    "Colored_MNIST_background",
    "Colored_FashionMNIST_foreground",
    "Colored_FashionMNIST_background",
    "UTKface",
    "CelebA",
    "BFFHQ",
]

# 64x64 keys should be run with --model ConvNetD4; the rest with ConvNet.
COBRA_64PX_KEYS = {"UTKface", "CelebA", "BFFHQ"}


class DropSensitive(Dataset):
    """Wrap a COBRA dataset yielding (img, label, sensitive[, ...]) into
    an (img, label) dataset, exposing the sensitive attribute separately."""

    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        item = self.base[index]
        return item[0], item[1]

    def get_sensitive(self, index):
        return self.base[index][2]


def get_sensitive_attrs(wrapped_dataset, num_workers=4):
    """Return the full sensitive-attribute vector of a DropSensitive
    dataset, aligned with its indices. Useful for computing Equalized
    Odds after evaluating an encoder trained on the distilled set."""
    loader = torch.utils.data.DataLoader(
        wrapped_dataset.base, batch_size=512, shuffle=False,
        num_workers=num_workers,
    )
    attrs = []
    for batch in loader:
        attrs.append(batch[2].long())
    return torch.cat(attrs)


def get_cobra_dataset(dataset, data_path, skew_ratio=0.9, severity=0):
    """Same contract as MKDT's utils.get_dataset:
    returns (channel, im_size, num_classes, dst_train, dst_test)."""

    channel = 3

    if dataset == "CIFAR10_S_90":
        im_size, num_classes = (32, 32), 10
        dst_train, dst_test, _, _ = cobra_cifar10.CIFAR_10s(
            data_path, skew_ratio=skew_ratio, severity=severity)

    elif dataset in ("Colored_MNIST_foreground", "Colored_MNIST_background"):
        im_size, num_classes = (32, 32), 10
        fg = dataset.endswith("foreground")
        dst_train, _ = get_biased_mnist_dataloader(
            data_path, batch_size=256, data_label_correlation=0.9,
            is_foreground=fg, n_confusing_labels=9, train=True)
        dst_test, _ = get_biased_mnist_dataloader(
            data_path, batch_size=256, data_label_correlation=0.1,
            is_foreground=fg, n_confusing_labels=9, train=False)

    elif dataset in ("Colored_FashionMNIST_foreground",
                     "Colored_FashionMNIST_background"):
        im_size, num_classes = (32, 32), 10
        fg = dataset.endswith("foreground")
        dst_train, _ = get_biased_fashionmnist_dataloader(
            data_path, batch_size=256, data_label_correlation=0.9,
            is_foreground=fg, n_confusing_labels=9, train=True)
        dst_test, _ = get_biased_fashionmnist_dataloader(
            data_path, batch_size=256, data_label_correlation=0.1,
            is_foreground=fg, n_confusing_labels=9, train=False)

    elif dataset == "UTKface":
        im_size, num_classes = (64, 64), 3
        dst_train, dst_test, _, _ = UTKFaceDataset()

    elif dataset == "CelebA":
        im_size, num_classes = (64, 64), 2
        target_label_idx, sensitive_label_idx = 2, 20
        dst_train, dst_test, _, _ = CelebA(target_label_idx,
                                           sensitive_label_idx)

    elif dataset == "BFFHQ":
        im_size, num_classes = (64, 64), 2
        dst_train, dst_test, _, _ = BFFHQ()

    else:
        raise ValueError("unknown COBRA dataset: %s" % dataset)

    return channel, im_size, num_classes, \
        DropSensitive(dst_train), DropSensitive(dst_test)


# ---------------------------------------------------------------------------
# Patch to MKDT/utils.py, inside get_dataset, added as a new branch before
# the subset handling at the end of the function (the existing branches all
# end by defining channel, im_size, num_classes, dst_train, dst_test, so
# this branch does the same and falls through to the shared tail):
#
#     elif dataset in COBRA_KEYS:
#         channel, im_size, num_classes, dst_train, dst_test = \
#             get_cobra_dataset(dataset, data_path)
#
# with this import at the top of utils.py:
#
#     from cobra_datasets import get_cobra_dataset, COBRA_KEYS
#
# Nothing else in MKDT needs to change: build_trainset stacks sample[0],
# distillation uses the loaded target representations as labels, and
# eval.py's linear probe reads datum[1], all of which the wrapper satisfies.
# ---------------------------------------------------------------------------
