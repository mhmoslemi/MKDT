import os
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from utils import get_dataset
from torchvision.models import resnet18


# Maps each dataset key to the base name of the KRRST teacher checkpoint to
# use. COBRA's 32x32 sets reuse the CIFAR-10 teacher; its 64x64 sets reuse the
# Tiny-ImageNet teacher. Override the whole path with --teacher_ckpt if you
# train your own teacher on a COBRA dataset.
KRRST_TEACHER = {
    'CIFAR10': 'cifar10',
    'CIFAR100': 'cifar100',
    'Tiny': 'tinyimagenet',
    'CIFAR10_S_90': 'cifar10',
    'Colored_MNIST_foreground': 'cifar10',
    'Colored_MNIST_background': 'cifar10',
    'Colored_FashionMNIST_foreground': 'cifar10',
    'Colored_FashionMNIST_background': 'cifar10',
    'UTKface': 'tinyimagenet',
    'CelebA': 'tinyimagenet',
    'BFFHQ': 'tinyimagenet',
}


def build_barlow_twins_teacher(device):
    # ResNet18 backbone with the CIFAR-style stem used by KRRST's teachers.
    model = resnet18()
    model.fc = nn.Identity()
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
    model.maxpool = nn.Identity()
    return model.to(device)


def load_teacher_state(model, ckpt_path, device):
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            "teacher checkpoint not found: %s\n"
            "Download the KRRST teacher checkpoints "
            "(https://github.com/db-Lee/selfsup_dd) into --ckpt_dir, or pass "
            "--teacher_ckpt with the full path to a teacher you trained." % ckpt_path
        )
    obj = torch.load(ckpt_path, map_location="cpu")

    # Accept a plain state_dict, a {'state_dict': ...} wrapper, or a full module.
    if isinstance(obj, nn.Module):
        state = obj.state_dict()
    elif isinstance(obj, dict) and "state_dict" in obj:
        state = obj["state_dict"]
    else:
        state = obj

    for key in ("fc.weight", "fc.bias"):
        state.pop(key, None)

    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as e:
        print("strict load failed, retrying non-strict:\n%s" % e)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print("missing keys:", missing)
        if unexpected:
            print("unexpected keys:", unexpected)
    return model


def main(args):
    args.device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    _, _, _, dst_train, _ = get_dataset(dataset=args.dataset, data_path=args.data_path)

    ''' Organize the real dataset '''
    images_all = []
    print("BUILDING TRAINSET")
    for i in tqdm(range(len(dst_train))):
        sample = dst_train[i]
        images_all.append(torch.unsqueeze(sample[0], dim=0))
    images_all = torch.cat(images_all, dim=0).to("cpu")
    print("images", tuple(images_all.shape))

    output_dir = os.path.join(args.result_dir, args.ssl_algorithm)
    os.makedirs(output_dir, exist_ok=True)

    ''' Build the teacher and load its weights '''
    if args.ssl_algorithm == "barlow_twins":
        target_model = build_barlow_twins_teacher(args.device)
        if args.teacher_ckpt is not None:
            ckpt_path = args.teacher_ckpt
        else:
            base = KRRST_TEACHER.get(args.dataset, args.dataset.lower())
            ckpt_path = os.path.join(args.ckpt_dir, f"barlow_twins_resnet18_{base}.pt")
        target_model = load_teacher_state(target_model, ckpt_path, args.device)
    else:
        # SimCLR teacher from SAS
        # (https://github.com/BigML-CS-UCLA/sas-data-efficient-contrastive-learning)
        from resnet import ResNet18, StemCIFAR
        target_model = ResNet18(stem=StemCIFAR).to(args.device)
        if args.teacher_ckpt is not None:
            ckpt_path = args.teacher_ckpt
        else:
            base = KRRST_TEACHER.get(args.dataset, args.dataset.lower())
            ckpt_path = os.path.join(args.ckpt_dir, f"{base}-resnet18-net.pt")
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                "simclr teacher checkpoint not found: %s (pass --teacher_ckpt "
                "or place it in --ckpt_dir)" % ckpt_path
            )
        loaded = torch.load(ckpt_path, map_location="cpu")
        state = loaded.state_dict() if isinstance(loaded, nn.Module) else loaded
        target_model.load_state_dict(state)

    target_model.eval()

    ''' Extract target representations in batches '''
    labels_all = []
    with torch.no_grad():
        for i in tqdm(range(0, len(images_all), args.batch_size), desc="Target reps"):
            batch = images_all[i:i + args.batch_size].to(args.device)
            rep = target_model(batch)
            labels_all.append(rep.detach().cpu())
    labels_all = torch.cat(labels_all, dim=0)

    out_path = os.path.join(output_dir, f"{args.dataset}_target_rep_train.pt")
    torch.save(labels_all, out_path)
    print("saved:", out_path)
    print("train label shape", tuple(labels_all.shape))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get Target Representation Parameter Processing')
    parser.add_argument('--dataset', type=str, default='CIFAR100', help='dataset')
    parser.add_argument('--data_path', type=str, default='/home/data', help='dataset path')
    parser.add_argument('--result_dir', type=str, default='target_rep', help='output root')
    parser.add_argument('--device', type=int, default=0, help='gpu number')
    parser.add_argument('--ssl_algorithm', type=str, default='barlow_twins', choices=['barlow_twins', 'simclr'], help='SSL algorithm used to get the target representation.')
    parser.add_argument('--ckpt_dir', type=str, default='./krrst_teacher_ckpt', help='directory holding the teacher checkpoints')
    parser.add_argument('--teacher_ckpt', type=str, default=None, help='full path to a teacher checkpoint; overrides --ckpt_dir mapping')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size for target-representation extraction')

    args = parser.parse_args()
    main(args)