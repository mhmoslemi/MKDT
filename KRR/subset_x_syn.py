"""Make reproducible nested 1% and 0.2% subsets from released KRR-ST 2% data."""
import argparse
from pathlib import Path

import torch


def main(source: Path, output_1: Path, output_0p2: Path) -> None:
    images = torch.load(source, map_location="cpu", weights_only=True, mmap=True)
    counts = {(1000, 3, 32, 32): (500, 100), (2000, 3, 64, 64): (1000, 200)}
    if not isinstance(images, torch.Tensor) or tuple(images.shape) not in counts:
        raise ValueError(f"Expected CIFAR100 or TinyImageNet 2% KRR-ST tensor, got {getattr(images, 'shape', None)}")
    if not images.is_floating_point() or not torch.isfinite(images).all():
        raise ValueError("Expected finite floating-point synthetic images")
    permutation = torch.randperm(len(images), generator=torch.Generator().manual_seed(0))
    count_1, count_0p2 = counts[tuple(images.shape)]
    for count, output in ((count_1, output_1), (count_0p2, output_0p2)):
        if output.exists():
            raise FileExistsError(f"Refusing to replace {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(images[permutation[:count]].clone(), output)
        torch.save(permutation[:count], output.with_name("source_indices_seed0.pt"))
        print(f"Saved {count} images to {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output_1", type=Path)
    parser.add_argument("output_0p2", type=Path)
    args = parser.parse_args()
    main(args.source, args.output_1, args.output_0p2)
