import argparse
import hashlib
import importlib.util
import json
import os
import statistics
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import torch
from torchvision.transforms.functional import to_pil_image


SSL_EPOCHS = {
    # "CIFAR10": {0.2: 2000, 1: 8000, 2: 6000},
    "CIFAR10": {0.2: 2000, 1: 2000, 2: 4000},
    "CIFAR100": {0.2: 3000, 1: 8000, 2: 6000},
    "TinyImageNet": {0.2: 6000, 1: 5000, 2: 4000},
}
PROBE_EPOCHS = {1: 100, 5: 100}

# Undo OT utils.get_dataset normalization before applying the benchmark transforms.
SYNTHETIC_NORMALIZATION = {
    "CIFAR10": [[0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]],
    "CIFAR100": [[0.5071, 0.4866, 0.4409], [0.2673, 0.2564, 0.2762]],
    "TinyImageNet": [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]],
}

ROOT = Path(__file__).resolve().parents[1]
_REFERENCE = None


def reference_modules():
    """Reuse benchmark functions without replacing OT's imported networks/utils."""
    global _REFERENCE
    if _REFERENCE is None:
        paths = {
            "networks": ROOT / "networks.py",
            "downstream_data": ROOT / "benchmarks/downstream_data.py",
            **{name: ROOT / "Evaluate_benchamrk" / (name + ".py")
               for name in ("common", "data", "train_ssl", "evaluate")},
        }
        previous = {name: sys.modules.get(name) for name in paths}
        search_path = sys.path[:]
        modules = {}
        try:
            for name, path in paths.items():
                spec = importlib.util.spec_from_file_location(name, path)
                module = importlib.util.module_from_spec(spec)
                sys.modules[name] = module
                spec.loader.exec_module(module)
                modules[name] = module
        finally:
            sys.path[:] = search_path
            for name, module in previous.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module
        _REFERENCE = modules
    return _REFERENCE


class SyntheticImages(torch.utils.data.Dataset):
    def __init__(self, images):
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return to_pil_image(self.images[index]), -1


@contextmanager
def ssl_progress(module, every, epochs):
    """Throttle console progress only; keep every epoch in ssl.jsonl."""
    had_print = 'print' in module.__dict__
    original_print = module.__dict__.get('print', print)

    def progress_print(*values, **kwargs):
        if kwargs.get('file') is None and len(values) == 1 and isinstance(values[0], str) and values[0].startswith('SSL '):
            epoch = json.loads(values[0].split(': ', 1)[1])['epoch']
            if epoch not in (1, epochs) and epoch % every:
                return
        original_print(*values, **kwargs)

    module.print = progress_print
    try:
        yield
    finally:
        if had_print:
            module.print = original_print
        else:
            del module.print


def evaluate_synset_SSL(it_eval, model, images_train, args, output, normalization=None, seed=None):
    """Train a fresh reference SSL encoder and a frozen-encoder linear probe."""
    reference = reference_modules()
    common, data = reference["common"], reference["data"]
    seed = args.seed + it_eval if seed is None else seed
    ssl_epochs = args.epoch_eval_train if args.epoch_eval_train is not None else SSL_EPOCHS[args.dataset].get(args.percentage)
    probe_epochs = args.epoch_linear_train if args.epoch_linear_train is not None else PROBE_EPOCHS[args.label_percentage]
    ssl_log_every = getattr(args, "ssl_log_every", 500)
    if ssl_epochs is None:
        raise ValueError("No SSL epoch schedule for this size; set the SSL epoch override")
    if min(ssl_epochs, probe_epochs, args.batch_linear, ssl_log_every) < 1 or args.lr_linear <= 0:
        raise ValueError("Epochs, batch size, print interval and learning rate must be positive")
    model = {"ConvNetBN": "ConvNet", "VGG11BN": "VGG11", "ResNet18BN": "ResNet18"}.get(model, model)
    if model not in ("ConvNet", "VGG11", "ResNet18"):
        raise ValueError("Reference evaluation requires ConvNet, VGG11 or ResNet18 (BN), not AP/cover")

    # Fresh benchmark recipe: generation augmentation/temperature/projector args do not enter here.
    protocol = json.loads((ROOT / "Evaluate_benchamrk/protocol.json").read_text())
    protocol["ssl"]["subset_epochs"] = ssl_epochs
    protocol["ssl"]["checkpoint_every_epochs"] = 500
    protocol["evaluation"].update(epochs=probe_epochs, lr=args.lr_linear, batch_size=args.batch_linear)
    size = protocol["input_size"][args.dataset]
    count = int(data.SOURCE_COUNTS[args.dataset] * args.percentage / 100)
    train_set = data.RawDataset(args.data_path, args.dataset, True)
    test_set = data.RawDataset(args.data_path, args.dataset, False)
    selection_method = "random" if getattr(args, "random_subset", False) else "ours"
    if selection_method == "random":
        synthetic, selection = data.select_subset(
            train_set, "random", args.percentage, seed, protocol,
            Path(output).parent / "selections", workers=0, device=args.device)
        assert len(synthetic) == count
        image_metadata = dict(selection=selection, pixel_policy="original_PIL_uint8")
        print("Random subset: %d distinct training images, selection seed=%d" % (count, seed), flush=True)
    else:
        if not isinstance(images_train, torch.Tensor) or images_train.shape != (count, 3, size, size) or count < 2:
            raise ValueError("Synthetic images must match the dataset, percentage and native resolution")
        images = images_train.detach().cpu().contiguous()
        if not images.is_floating_point() or not torch.isfinite(images).all():
            raise ValueError("Synthetic images must be finite, normalized floating-point tensors")
        normalization = SYNTHETIC_NORMALIZATION[args.dataset] if normalization is None else normalization
        mean, std = (torch.tensor(value).view(1, 3, 1, 1) for value in normalization)
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
            raise ValueError("Invalid synthetic normalization")
        pixels = images * std + mean
        clipped_fraction = float(((pixels < 0) | (pixels > 1)).float().mean())
        synthetic = SyntheticImages(pixels.clamp(0, 1).contiguous())
        image_metadata = dict(synthetic_sha256=hashlib.sha256(images.numpy().tobytes()).hexdigest(),
                              synthetic_dtype=str(images.dtype), synthetic_normalization=normalization,
                              clipped_fraction=clipped_fraction, pixel_policy="denormalize_clamp_then_PIL_uint8")
    signature = dict(protocol=protocol, source=args.dataset, model=model, ssl=args.ssl_method,
                     subset=args.percentage, selection_method=selection_method, source_fingerprint=train_set.fingerprint,
                     **image_metadata,
                     code_hashes=common.code_hashes(), evaluation_driver_sha256=common.file_hash(__file__))
    identity = dict(signature=common.digest_json(signature), seed=seed, selection_method=selection_method)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    common.atomic_json(output / "config.json", dict(signature=signature, identity=identity))
    print("Reference SSL: %s/%s/%s, SSL epochs=%d, probe epochs=%d, labels=%g%%" %
          (args.dataset, model, args.ssl_method, ssl_epochs, probe_epochs, args.label_percentage), flush=True)

    # Do not let evaluation alter the distillation RNGs or numerical settings.
    import numpy as np
    rng, numpy_rng = common.rng_state(), np.random.get_state()
    settings = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
                torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic)
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with ssl_progress(reference["train_ssl"], ssl_log_every, ssl_epochs):
            checkpoint, ssl_saved = reference["train_ssl"].train(
                synthetic, protocol, args.dataset, model, args.ssl_method,
                seed, output, identity, workers=0, device=args.device)
        common.atomic_json(output / "ssl_complete.json", dict(
            complete=True, identity=identity, epochs=ssl_saved["epochs"],
            encoder_sha256=ssl_saved["encoder_sha256"], checkpoint_file_sha256=common.file_hash(checkpoint)))
        result = reference["evaluate"].linear_probe(
            checkpoint, identity, protocol, args.dataset, args.dataset,
            train_set, test_set, args.label_percentage, seed,
            output / ("%s_lbl%g.json" % (args.dataset, args.label_percentage)), workers=0, device=args.device)
    finally:
        common.restore_rng(rng)
        np.random.set_state(numpy_rng)
        (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
         torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic) = settings
    return checkpoint, result, result["accuracy_percent"] / 100


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(SSL_EPOCHS), default="CIFAR10")
    parser.add_argument("--percentage", type=float, default=2)
    parser.add_argument("--label-percentage", type=int, choices=list(PROBE_EPOCHS), default=1)
    parser.add_argument("--model", choices=["ConvNet", "ConvNetBN", "VGG11", "VGG11BN", "ResNet18", "ResNet18BN"], default="ResNet18BN")
    parser.add_argument("--ssl-method", choices=["simclr", "barlowtwins"], default="simclr")
    parser.add_argument("--ssl-epochs", dest="epoch_eval_train", type=int, default=None)
    parser.add_argument("--ssl-log-every", "--print-every", type=int, default=100, help="Print SSL progress every N epochs; always print first/final epochs")
    parser.add_argument("--probe-epochs", dest="epoch_linear_train", type=int, default=None)
    parser.add_argument("--lr-linear", type=float, default=0.1)
    parser.add_argument("--batch-linear", type=int, default=256)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--data-path", default=str(ROOT.parent / "data"))
    synthetic_source = parser.add_mutually_exclusive_group()
    synthetic_source.add_argument("--synthetic-checkpoint", type=Path, default=Path("/scratch/mmoslem3/MKDT/OT-SSL-DD/ours_syntehtic/CIFAR10/ResNet18BN/pt2/res_IIC_CIFAR10_ResNet18BN_2percent_final.pt"))
    synthetic_source.add_argument("--synthetic-root", type=Path, default=None, help="Legacy sizeN/res_IIC_DATASET_ConvNet_Npercent.pt layout")
    synthetic_source.add_argument("--random-subset", action="store_true", help="Use randomly selected real training images instead of a synthetic checkpoint")
    parser.add_argument("--save-path", type=Path, default=None)
    args = parser.parse_args()

    if args.epoch_eval_train is None:
        args.epoch_eval_train = SSL_EPOCHS[args.dataset].get(args.percentage)
    if args.epoch_linear_train is None:
        args.epoch_linear_train = PROBE_EPOCHS[args.label_percentage]
    if args.epoch_eval_train is None:
        parser.error("No SSL epoch schedule for this size; set --ssl-epochs explicitly")
    if min(args.epoch_eval_train, args.epoch_linear_train, args.runs, args.batch_linear, args.ssl_log_every) < 1 or args.lr_linear <= 0 or not 0 < args.percentage <= 100:
        parser.error("Epochs, runs, batches, print interval and learning rate must be positive; percentage must be in (0, 100]")
    
    checkpoint_path = args.synthetic_checkpoint
    if args.synthetic_root is not None:
        checkpoint_path = args.synthetic_root / f"size{args.percentage:g}" / f"res_IIC_{args.dataset}_ConvNet_{args.percentage:g}percent.pt"



    checkpoint_path = None if args.random_subset else Path(checkpoint_path)
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    saved = {} if args.random_subset else torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(saved, torch.Tensor):
        # MKDT images_*.pt files are raw tensors normalized by the root dataset loader.
        _, mean, std = reference_modules()["data"].SPECS[args.dataset]
        saved = {"data": saved, "normalization": [mean, std]}
    elif not isinstance(saved, dict):
        raise ValueError("Synthetic checkpoint must contain an image tensor or a dictionary with a data key")
    if saved.get("dataset", args.dataset) != args.dataset or saved.get("percentage", args.percentage) != args.percentage:
        raise ValueError("Synthetic checkpoint metadata disagrees with the requested dataset/percentage")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    prefix = "random" if args.random_subset else "manual"
    output = args.save_path if args.save_path else ROOT / "OT-SSL-DD/eval_results" / f"{prefix}_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(synthetic_checkpoint=str(checkpoint_path.resolve()) if checkpoint_path else None,
                  save_path=str(output.resolve()), std_ddof=1)
    common = reference_modules()["common"]
    common.atomic_json(output / "config.json", config)
    print(json.dumps(config, indent=2), flush=True)
    accuracies = []

    for run in range(args.runs):
        seed = args.seed + run
        _, _, accuracy = evaluate_synset_SSL(run, args.model, saved.get("data"), args,
                                             output / f"seed_{seed:03d}", saved.get("normalization"))
        accuracies.append(accuracy * 100)
        summary = dict(config=config, completed_runs=len(accuracies), expected_runs=args.runs,
                       complete=len(accuracies) == args.runs, accuracies_percent=accuracies,
                       mean_percent=statistics.mean(accuracies),
                       std_percent=statistics.stdev(accuracies) if len(accuracies) > 1 else None)
        common.atomic_json(output / "summary.json", summary)
        print(f"seed={seed} accuracy={accuracy * 100:.2f}%", flush=True)

    std = f"{summary['std_percent']:.2f}" if summary["std_percent"] is not None else "N/A"
    print(f"Mean ± sample std: {summary['mean_percent']:.2f} ± {std}%")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
