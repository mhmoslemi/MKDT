# MNIST / FashionMNIST / SVHN MKDT completion

```bash
bash Evaluate_benchamrk/submit_digits_mkdt_finish_vulcan.sh
```

Reuses active prerequisites, resumes the MNIST Barlow teacher, and restores missing
MNIST feature/expert/initialization jobs. Existing MIPA and full SimCLR jobs are untouched.

For each dataset at 0.2%, 1%, and 2%: the existing original ConvNet MKDT algorithm
runs 5,000 updates in five dependent 1,000-update chunks. Full images, labels,
learned LR, optimizers, expert cursor and RNG are checkpointed every 100 updates.
Only dataset loading/normalization changes through the existing digit adapters.

Each final synthetic set feeds one ResNet18 BN SimCLR encoder for 2,000 epochs,
then five independent 100-epoch linear probes for each of the 1% and 5% label budgets.
SSL resumes every 500 epochs; completed probes are reused. An interrupted head restarts.
Main table only: nine synthetic sets, nine encoders, 18 cells, 90 probes.

Outputs: `synthetic/<dataset>/ConvNet/pt<size>/` and
`results/<dataset>/ResNet18/pt<size>/simclr/ssl_seed_000/`.
Submission receipts and exact dependencies are in `batches/`; logs are in `logs/`.
MNIST prerequisites retain Lu; new distillation/evaluation jobs use Wang.
GPU jobs request one L40S, one CPU and 6 GB; initialization assembly is CPU-only.
Distillation limits extrapolate historical CIFAR MKDT timings; SSL/probes use the
matched completed Random runs. Limits include 120 seconds startup/I/O and a one-minute buffer.
