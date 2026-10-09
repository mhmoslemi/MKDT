import argparse
import json
import os
import statistics
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path("/home/mmoslem3/scratch/MKDT")
sys.path.insert(0, str(ROOT / "benchmarks"))

import torch
import benchmark_cifar10 as bench


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--percentage", type=int, choices=[1, 2, 5], required=True)
    parser.add_argument("--label-percentage", type=int, choices=[1, 5], default=1)
    parser.add_argument("--model", choices=["ConvNet", "VGG11", "ResNet18"], default="ConvNet")
    parser.add_argument("--ssl-method", choices=["simclr", "barlowtwins"], default="simclr")
    parser.add_argument("--ssl-log-every", type=int, default=200)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-path", default="/scratch/mmoslem3/data")
    # parser.add_argument("--synthetic-root", default=str(ROOT / "benchmarks/ours/batch_20261008T195254Z_EkW378/yiweilu_mCLPIh/synthetic"))
    parser.add_argument("--synthetic-root", default=str(ROOT / "benchmarks/ours/batch_20261008T195254Z_EkW378/yiweilu_mCLPIh/synthetic"))
    # 
    parser.add_argument("--save-path", default=None)
    args = parser.parse_args()




    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    budgets = json.loads((ROOT / "OT-SSL-DD/training_epochs.json").read_text())
    ssl_epochs = 5000
    probe_epochs = 300
    checkpoint_path = Path(args.synthetic_root) / f"size{args.percentage}" / f"res_IIC_CIFAR10_ConvNet_{args.percentage}percent.pt"
    checkpoint_path = 'result2/res_IIC_CIFAR10_ConvNet_1percent.pt'
    synthetic = torch.load(checkpoint_path, map_location="cpu", weights_only=True)["data"].to("cuda")
    if synthetic.shape != (500 * args.percentage, 3, 32, 32):
        raise ValueError(f"Unexpected synthetic shape: {tuple(synthetic.shape)}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = Path(args.save_path) if args.save_path else ROOT / "benchmarks/ours" / f"manual_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    config = dict(vars(args), ssl_epochs=5000, probe_epochs=500, batch_size=256, lr_ssl=0.01, lr_probe=0.1, momentum=0.9, weight_decay=0.0005, temperature=0.2, barlow_lambda=0.005, projection_dim=128, augmentation=bench.STRATEGY, augmentation_mode="S", synthetic_checkpoint=str(checkpoint_path), std_ddof=1)
    bench.write_json(output / "config.json", config)
    print(json.dumps(config, indent=2), flush=True)

    train, labels, test, test_labels = bench.load_data(args.data_path, "cuda")
    labels_cpu = labels.cpu().numpy()
    accuracies = []

    for seed in range(args.seed, args.seed + args.runs):
        chosen = torch.tensor(bench.labeled_indices(labels_cpu, args.label_percentage, seed), device="cuda")
        network = bench.get_network(args.model, 3, 10, (32, 32), seed=seed + 3000)
        bench.train_ssl(network, synthetic, args.ssl_method, ssl_epochs, seed, log_every=args.ssl_log_every)
        accuracy = bench.linear_probe(network, train[chosen], labels[chosen], test, test_labels, probe_epochs, seed)
        accuracies.append(accuracy)
        bench.write_json(output / f"seed_{seed:03d}.json", dict(seed=seed, accuracy_percent=accuracy, labeled_indices=chosen.cpu().tolist()))
        summary = dict(config=config, completed_runs=len(accuracies), expected_runs=args.runs, complete=len(accuracies) == args.runs, accuracies_percent=accuracies, mean_percent=statistics.mean(accuracies), std_percent=statistics.stdev(accuracies) if len(accuracies) > 1 else None)
        bench.write_json(output / "summary.json", summary)
        print(f"seed={seed} accuracy={accuracy:.2f}%", flush=True)
        del network

    std = f"{summary['std_percent']:.2f}" if summary["std_percent"] is not None else "N/A"
    print(f"Mean ± sample std: {summary['mean_percent']:.2f} ± {std}%")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
