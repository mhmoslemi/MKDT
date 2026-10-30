"""One similarity/dataset/size: resume MIPA, 2k SimCLR, and five probes per label budget."""
import importlib.util
import json
import os
from pathlib import Path
import sys

import torch

HERE = Path(__file__).resolve().parent
BENCH, ROOT = HERE.parent, HERE.parent.parent
SOURCES = ('CIFAR10', 'CIFAR100', 'SVHN', 'FashionMNIST')
METRICS = ('l1', 'l2', 'mmd', 'mmd-dist', 'dm', 'energy', 'whitened-cosine', 'sinkhorn')
COUNTS = dict(CIFAR10=50000, CIFAR100=50000, SVHN=73257, FashionMNIST=60000)
SYNTHETICS = ROOT / 'OT-SSL-DD/ours_syntehtic_similarity_lr0p002_b128_i10k'


def read(path):
    return json.loads(Path(path).read_text())


def generation_args(source, subset):
    teacher_folder = {'SVHN': 'digits_simclr/checkpoints',
                      'FashionMNIST': 'fashionmnist/full/checkpoints'}.get(source, 'full_ssl_checkpoints_1000ep_3seeds')
    return dict(dataset=source, percentage=subset, model='ResNet18BN', seed=0,
        Iteration=10000, lr_img=0.002, batch_real=128, temperature=0.2,
        disp_weight=0.0, pair_weight=0.0, num_random_networks=1, random_models='ResNet18BN',
        dsa=True, distill_aug_strategy='color_crop_cutout_flip_scale_rotate', distill_aug_mode='M',
        teacher_source='pretrained', teacher_seed=[0], teacher_ssl='simclr',
        teacher_path=str(BENCH / teacher_folder), feature_batch=128)


def configure(job, snapshot, data_root):
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / 'OT-SSL-DD'))
    spec = importlib.util.spec_from_file_location('method1', snapshot)
    method = importlib.util.module_from_spec(spec)
    sys.modules['method1'] = method
    spec.loader.exec_module(method)
    from Evaluate_benchamrk.mipa_random128 import worker as base

    reference = base.reference_eval.reference_modules()
    common = reference['common']
    for relative, expected in job['code_hashes'].items():
        path = snapshot if relative == 'OT-SSL-DD/method1.py' else ROOT / relative
        if common.file_hash(path) != expected:
            raise ValueError('Code changed since submission: ' + relative)
    origin = ROOT / job['selection']['config']
    assert common.file_hash(origin) == job['selection']['config_sha256']
    previous = read(origin)
    assert previous['identity']['seed'] == 0 and previous['identity']['selection_method'] == 'random'
    assert previous['identity']['selection']['indices'] == job['selection']['indices']
    teacher = ROOT / job['teacher']['path']
    assert common.file_hash(teacher) == job['teacher']['file_sha256']
    assert common.file_hash(teacher.parent / 'ssl_complete.json') == job['teacher']['marker_sha256']
    assert read(teacher.parent / 'ssl_complete.json')['complete']

    source, subset, metric = job['source'], job['subset'], job['similarity']
    base.HERE, base.SOURCES = HERE, SOURCES
    base.ITERATIONS = {0.2: 10000, 1: 10000, 2: 10000}
    base.method1 = method
    base.generation_args = generation_args
    base.synthetic_path = lambda source, subset: SYNTHETICS / source / 'ResNet18BN' / ('pt%g' % subset) / metric / (
        'res_IIC_%s_ResNet18BN_%gpercent_final.pt' % (source, subset))
    base.encoder_folder = lambda source, subset: HERE / 'results' / source / 'ResNet18BN' / ('pt%g' % subset) / metric / 'simclr/ssl_seed_000'
    original_contract = base.contract
    base.contract = lambda item: dict(original_contract(item), similarity=item['similarity'], metric_parameters=item['metric_parameters'])
    original_main = method.main

    def main_with_metric():
        sys.argv += ['--similarity', metric, '--init', 'random']
        original_main()

    method.main = main_with_metric
    reference['data'].SOURCE_COUNTS.update(COUNTS)
    if source in ('SVHN', 'FashionMNIST'):
        prepared_path = ROOT / job['prepared_file']
        assert common.file_hash(prepared_path) == job['prepared_sha256']
        prepared = read(prepared_path)
        assert prepared['datasets'][source]['train_fingerprint'] == job['selection']['source_fingerprint']
        adapter = reference['_fashion_dataset' if source == 'FashionMNIST' else 'datasets']
        reference['data'].RawDataset = lambda root, name, training: adapter.DigitDataset(root, name, training)
        original_read = base.read
        base.read = lambda path: json.loads(json.dumps(prepared['protocol'])) if Path(path).resolve() == BENCH / 'protocol.json' else original_read(path)

    original_summary = base.summarize

    def summarize(*arguments):
        original_summary(*arguments)
        target, labels = arguments[-2:]
        path = base.encoder_folder(source, subset) / ('summary_%s_lbl%d.json' % (target, labels))
        summary = read(path)
        summary.update(similarity=metric, metric_parameters=job['metric_parameters'])
        summary.pop('table_scope', None)
        common.atomic_json(path, summary)

    base.summarize = summarize
    job['data_root'] = str(data_root)
    return base, reference


def check(job, data_root, base, reference):
    """Read-only CPU preflight on staged data, saved teachers, and the shared similarity loss."""
    source = job['source']
    raw = reference['data'].RawDataset(data_root, source, True)
    assert raw.fingerprint == job['selection']['source_fingerprint']
    mean, std = base.reference_eval.SYNTHETIC_NORMALIZATION[source]
    transform = base.method1.transforms.Compose([base.method1.transforms.ToTensor(), base.method1.transforms.Normalize(mean, std)])
    dataset = reference['data'].ImageView(raw, transform)
    count = int(len(dataset) * job['subset'] / 100)
    expected = base.initial_images(job, dataset, count, 'cpu')
    actual = base.method1.deepcluster_initialize(dataset, count, 'cpu', 0, 3, raw.classes, (32, 32), init='random')
    assert torch.equal(actual, expected), 'Initialization must use exactly the saved Random images'
    args = base.method1.argparse.Namespace(**dict(job['generation_args'], teacher_seed=0, device='cpu'))
    teacher = base.method1.load_teacher(args, mean, std, (32, 32))
    embed = base.method1.frozen_embedder(teacher)
    real = actual[:2]
    synthetic = actual[2:4].clone().requires_grad_(True)
    param = base.method1.ParamDiffAug()
    augmenter = base.method1.make_diff_augmenter('color_crop_cutout_flip_scale_rotate', param, dsa=True)
    with torch.no_grad():
        ur, vr, ur2, vr2 = [embed(augmenter(real, seed)) for seed in (10, 11, 12, 13)]
    us, vs, us2, vs2 = [embed(augmenter(synthetic, seed)) for seed in (10, 11, 12, 13)]
    for metric in METRICS:
        values = (ur, vr, us, vs)
        if metric in ('mmd-dist', 'dm', 'energy', 'sinkhorn'):
            values = tuple(torch.stack(pair, dim=1) for pair in zip(values, (ur2, vr2, us2, vs2)))
        whitening = (torch.zeros(ur.shape[1]), torch.eye(ur.shape[1])) if metric == 'whitened-cosine' else None
        loss = base.method1.bipartite_view_transition_loss(*values, tau=0.2, similarity=metric, whitening=whitening)[0]
        gradient, = torch.autograd.grad(loss, synthetic, retain_graph=True)
        assert torch.isfinite(loss) and torch.isfinite(gradient).all()
        assert all(parameter.grad is None for parameter in teacher.parameters())
    protocol = base.read(BENCH / 'protocol.json')
    assert protocol['ssl']['batch_size'] == 256 and protocol['evaluation']['epochs'] == 100
    print('CHECK PASSED:', source, 'exact Random initialization, saved teacher, all 8 metric gradients, standard SSL/probe protocol', flush=True)


def main():
    manifest, data_root = (Path(value).resolve() for value in sys.argv[1:3])
    check_only = sys.argv[3:] == ['--check']
    assert check_only or not sys.argv[3:]
    assert manifest.is_relative_to(HERE / 'batches')
    snapshot = manifest.parent / 'method1.py'
    job = read(manifest)
    assert job['source'] in SOURCES and job['subset'] in (0.2, 1, 2) and job['similarity'] in METRICS
    assert job['generation_args'] == generation_args(job['source'], job['subset'])
    assert job['evaluation'] == dict(ssl='simclr', ssl_epochs=2000, ssl_seed=0, linear_epochs=100,
                                     probe_seeds=list(range(5)), main_labels=[1, 5])
    assert os.environ.get('SLURM_JOB_ID'), 'Use Slurm, including for CPU preflight'
    if not check_only:
        assert os.environ.get('SLURM_JOB_ACCOUNT') == job['account']
        assert os.environ.get('SLURM_CLUSTER_NAME', os.environ.get('CC_CLUSTER')) == 'vulcan'
        assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    base, reference = configure(job, snapshot, data_root)
    if check_only:
        check(job, data_root, base, reference)
        return
    with reference['common'].exclusive_run(HERE / 'sessions' / job['name']):
        base.generation(job, data_root, reference['common'])
        path = base.synthetic_path(job['source'], job['subset'])
        saved = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        assert saved['init'] == 'random' and saved['iic_params']['similarity'] == job['similarity']
        assert all(saved['iic_params'][name] == value for name, value in job['metric_parameters'].items())
        del saved
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        base.ssl_and_main(job, reference)


if __name__ == '__main__':
    main()
