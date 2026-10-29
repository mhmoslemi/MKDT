# SVHN / MNIST: full-data SimCLR stage

Submit or resume on Vulcan:

```bash
bash Evaluate_benchamrk/submit_digits_full_simclr_vulcan.sh
```

The preparation job downloads MNIST through torchvision (official archive MD5
checks), verifies the already staged SVHN train/test splits, measures actual
training/probe time on an L40S, then submits six jobs. Three training jobs charge
Wang and three charge Lu. Each requests one L40S, one CPU, and 6 GB RAM. Requested
walltimes project the measured epoch time to 1,000 epochs, add measured probe
time, two minutes for startup/checkpoint I/O, and a one-minute buffer.

Each dataset uses full-data encoder seeds 0, 1, and 2, matching the existing
CIFAR10 full-data runs. Each encoder gets one frozen 100-epoch linear probe with
1% labels and another with 5% labels. Completed encoders/probes are validated and
reused; interrupted SSL resumes from the atomically replaced `training.pt`, saved
every 50 epochs. An interrupted head restarts only that head. Final checkpoints
are in:

```
digits_simclr/checkpoints/<SVHN|MNIST>/ResNet18/full/pt100/simclr/seed_00{0,1,2}/
```

These are exactly the existing root ResNet18 **BN, non-AP** backbone and shared
`train_ssl.train` / `evaluate.linear_probe` implementations. The CIFAR10 SimCLR
recipe (including augmentations) is unchanged: 32px inputs, batch 256, LARS,
lr 0.4 scaled by effective batch, temperature 0.2, projector 2048→256, ten-epoch
warmup then cosine decay. Probes use SGD lr 0.1, momentum 0.9, weight decay zero,
cosine decay, and a frozen encoder including BN buffers. No new "official digit
recipe" is claimed; this is the requested CIFAR10-recipe extension.

SVHN uses only its official 73,257-image training split, not `extra`. Torchvision
maps label 10 to digit 0. MNIST uses its official 60,000-image training split,
resizes 28px grayscale to 32px with PIL bicubic, then repeats grayscale into RGB.
Both use channel means/standard deviations computed from **training pixels only**;
the same processing is used at test time. All shared source files remain unchanged
so the older running experiments keep their original code fingerprints.

One-epoch timing encoders/probes live only under `calibration/`; they never enter
the table or serve as distillation teachers. Full-data means/sample standard
deviations are under `summaries/<dataset>/summary.json` once all three seeds finish.

This launcher submits the full-data teacher/baseline stage only. The added main
table rows reserve all three subset sizes (0.2%, 1%, 2%) and both label budgets.
MIPA, Random, K-means and MKDT subset jobs are not submitted by this first-stage
launcher. Agreed subsequent MIPA settings are the production CIFAR10 settings:
batch_real 1024, Adam lr_img 0.001, displacement/pair coefficients zero, saved
teacher seed 0, and 10k/8k/6k distillation iterations by size. Small-set SimCLR
uses 2k epochs. MKDT needs its own dataset-specific experts/generation, not MIPA
data relabeled as MKDT.
