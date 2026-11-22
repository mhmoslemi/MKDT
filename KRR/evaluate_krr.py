"""Train one 2k SimCLR encoder on KRR-ST images, then five probes per label budget."""
import argparse
import fcntl
import json
import math
from pathlib import Path
import statistics
import sys

import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_pil_image

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "Evaluate_benchamrk"
sys.path.insert(0, str(BENCH))
from common import atomic_json, digest_json, exclusive_run, file_hash
from data import RawDataset, SOURCE_COUNTS
from evaluate import linear_probe
from train_ssl import train

NAMES = {"cifar10": "CIFAR10", "cifar100": "CIFAR100", "tinyimagenet": "TinyImageNet"}
NORMALIZATION = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": ((0.5071, 0.4866, 0.4409), (0.2673, 0.2564, 0.2762)),
    "tinyimagenet": ((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)),
}


class SyntheticImages(Dataset):
    def __init__(self, pixels):
        self.pixels = pixels

    def __len__(self):
        return len(self.pixels)

    def __getitem__(self, index):
        return to_pil_image(self.pixels[index]), -1


def pixels_from(images, dataset):
    if not isinstance(images, torch.Tensor) or images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("Expected an NCHW RGB tensor")
    if not images.is_floating_point() or not torch.isfinite(images).all():
        raise ValueError("Expected finite floating-point KRR-ST images")
    mean, std = NORMALIZATION[dataset]
    mean = torch.tensor(mean, dtype=images.dtype).view(1, 3, 1, 1)
    std = torch.tensor(std, dtype=images.dtype).view(1, 3, 1, 1)
    pixels = images.detach().cpu() * std + mean
    return pixels.clamp(0, 1).contiguous(), float(((pixels < 0) | (pixels > 1)).float().mean())


def read(path):
    return json.loads(Path(path).read_text())


def requested_probes(stage, labels=None, seed=None):
    if stage == "all":
        return tuple((label, probe_seed) for label in (1, 5) for probe_seed in range(5))
    if stage == "probe":
        return ((labels, seed),)
    return ()


def summarize(folder, source, identity):
    summaries = {}
    for labels in (1, 5):
        scores = []
        for seed in range(5):
            path = folder / "evaluations" / f"{source}_lbl{labels}_probe{seed}.json"
            if not path.is_file():
                continue
            result = read(path)
            if not result.get("complete") or result["expected"]["identity"] != identity:
                raise ValueError(f"Incompatible probe result: {path}")
            score = result["accuracy_percent"]
            if not math.isfinite(score):
                raise ValueError(f"Non-finite probe result in {path}")
            scores.append(score)
        entry = {"completed_probes": len(scores), "expected_probes": 5, "scores": scores}
        if len(scores) == 5:
            entry.update(mean_percent=statistics.mean(scores), std_percent=statistics.stdev(scores))
            print(f"KRR-ST {source} lbl{labels}: {entry['mean_percent']:.2f} +/- "
                  f"{entry['std_percent']:.2f}", flush=True)
        summaries[str(labels)] = entry
    atomic_json(folder / "summary.json", {"complete": all(v["completed_probes"] == 5 for v in summaries.values()),
                "ssl_seed": 0, "probe_seeds": list(range(5)), "std_ddof": 1, "labels": summaries})


def main(args):
    if not args.x_syn.is_file():
        raise FileNotFoundError(args.x_syn)
    source = NAMES[args.dataset]
    count = round(SOURCE_COUNTS[source] * args.percentage / 100)
    images = torch.load(args.x_syn, map_location="cpu", weights_only=True, mmap=True)
    pixels, clipped = pixels_from(images, args.dataset)
    if len(pixels) != count:
        raise ValueError(f"Expected {count} images for {source} {args.percentage:g}%, got {len(pixels)}")
    size = 64 if args.dataset == "tinyimagenet" else 32
    if tuple(pixels.shape[1:]) != (3, size, size):
        raise ValueError(f"Expected native {size}x{size} images")
    synthetic = SyntheticImages(pixels)
    protocol = read(BENCH / "protocol.json")
    protocol["ssl"].update(subset_epochs=2000, checkpoint_every_epochs=500)
    protocol["subset_seeds"] = [0]
    metadata = {"path": str(args.x_syn.resolve()), "sha256": file_hash(args.x_syn),
                "normalization": NORMALIZATION[args.dataset], "clipped_fraction": clipped,
                "count": count, "percentage": args.percentage}
    train_set, test_set = RawDataset(args.data_root, source, True), RawDataset(args.data_root, source, False)
    signature = {"experiment": "krr_st_main_table_2k_1ssl5probes", "source": source,
                 "model": "ResNet18", "subset": args.percentage, "ssl": "simclr",
                 "selection_method": "krr_st", "synthetic": metadata, "protocol": protocol,
                 "source_fingerprint": train_set.fingerprint, "driver_sha256": file_hash(Path(__file__))}
    identity = {"signature": digest_json(signature), "seed": 0, "selection_method": "krr_st"}
    folder = args.output / source / ("pt%g" % args.percentage) / "simclr" / "ssl_seed_000"
    config = {"signature": signature, "identity": identity}
    config_path = folder / "config.json"
    if args.stage in ("all", "ssl"):
        with exclusive_run(folder):
            if config_path.exists() and read(config_path) != json.loads(json.dumps(config)):
                raise ValueError(f"Refusing to mix a different run in {folder}")
            atomic_json(config_path, config)
            checkpoint, saved = train(synthetic, protocol, source, "ResNet18", "simclr", 0,
                                      folder, identity, workers=args.workers, device="cuda")
            atomic_json(folder / "ssl_complete.json", {"complete": True, "identity": identity,
                        "epochs": saved["epochs"], "encoder_sha256": saved["encoder_sha256"],
                        "checkpoint_file_sha256": file_hash(checkpoint)})
        if args.stage == "ssl":
            return
    if not config_path.is_file() or read(config_path) != json.loads(json.dumps(config)):
        raise ValueError(f"Missing or incompatible SSL configuration in {folder}")
    checkpoint = folder / "encoder.pt"
    marker = read(folder / "ssl_complete.json")
    if (not marker.get("complete") or marker["identity"] != identity or marker["epochs"] != 2000
            or marker["checkpoint_file_sha256"] != file_hash(checkpoint)):
        raise ValueError(f"Incomplete or changed SSL encoder in {folder}")
    for labels, seed in requested_probes(args.stage, args.labels, args.probe_seed):
        name = f"{source}_lbl{labels}_probe{seed}"
        with exclusive_run(folder / "probe_locks" / name):
            output = folder / "evaluations" / f"{name}.json"
            linear_probe(checkpoint, identity, protocol, source, source, train_set,
                         test_set, labels, seed, output, args.workers, "cuda")
        with (folder / ".summary.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            summarize(folder, source, identity)


def self_test():
    for dataset, size in (("cifar10", 32), ("cifar100", 32), ("tinyimagenet", 64)):
        images = torch.zeros(2, 3, size, size)
        pixels, clipped = pixels_from(images, dataset)
        mean = torch.tensor(NORMALIZATION[dataset][0]).view(1, 3, 1, 1)
        assert pixels.shape == images.shape and torch.allclose(pixels[:, :, :1, :1], mean)
        assert clipped == 0.0
    assert len(set(requested_probes("all"))) == 10
    assert requested_probes("probe", 5, 3) == ((5, 3),)
    assert requested_probes("ssl") == ()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", nargs="?", choices=tuple(NAMES))
    parser.add_argument("percentage", nargs="?", type=float)
    parser.add_argument("x_syn", nargs="?", type=Path)
    parser.add_argument("--data-root", type=Path, default=ROOT.parent / "data")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "evaluations")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--stage", choices=("all", "ssl", "probe"), default="all")
    parser.add_argument("--labels", type=int, choices=(1, 5))
    parser.add_argument("--probe-seed", type=int, choices=range(5))
    parser.add_argument("--self-test", action="store_true")
    parsed = parser.parse_args()
    if parsed.self_test:
        self_test()
    elif parsed.dataset is None or parsed.percentage is None or parsed.x_syn is None:
        parser.error("dataset, percentage, and x_syn are required")
    else:
        if parsed.stage == "probe" and (parsed.labels is None or parsed.probe_seed is None):
            parser.error("--labels and --probe-seed are required for --stage probe")
        main(parsed)
