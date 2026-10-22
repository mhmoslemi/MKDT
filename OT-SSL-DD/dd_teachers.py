from pathlib import Path

import torch
import torch.nn as nn

from eval import ROOT, reference_modules


def teacher_arguments(parser):
    parser.add_argument('--teacher_source', choices=['fresh', 'pretrained'], default='fresh')
    parser.add_argument('--teacher_seed', type=int, default=0)
    parser.add_argument('--teacher_ssl', choices=['simclr', 'barlowtwins'], default='simclr')
    parser.add_argument('--teacher_path', default=str(ROOT / 'Evaluate_benchamrk/full_ssl_checkpoints_1000ep_3seeds'))


class NormalizedTeacher(nn.Module):
    def __init__(self, network, source, target):
        super().__init__()
        self.network = network
        for name, values in zip(('source_mean', 'source_std', 'teacher_mean', 'teacher_std'), (*source, *target)):
            self.register_buffer(name, torch.tensor(values).view(1, 3, 1, 1))

    def embed(self, images):
        return self.network((images * self.source_std + self.source_mean - self.teacher_mean) / self.teacher_std)

    def forward(self, images):
        return self.embed(images)


def load_teacher(args, mean, std, im_size):
    if args.teacher_source == 'fresh':
        return None
    model = args.model.removesuffix('BN')
    path = Path(args.teacher_path) / args.dataset / model / 'full' / 'pt100' / args.teacher_ssl / ('seed_%03d' % args.teacher_seed) / 'encoder.pt'
    network, _, saved = reference_modules()['common'].load_encoder(path, device=args.device)
    if saved['model'] != model or saved['input_size'] != im_size[0]:
        raise ValueError('Teacher architecture or input size differs from this run')
    print('Loaded fixed teacher seed %d: %s' % (args.teacher_seed, path), flush=True)
    return NormalizedTeacher(network, (mean, std), saved['normalization']).to(args.device).eval().requires_grad_(False)
