# FashionMNIST

Submit or resume this dataset only:

```bash
bash Evaluate_benchamrk/submit_fashionmnist_vulcan.sh
```

The official train/test archives are staged under `/scratch/mmoslem3/data/FashionMNIST/raw`.
Their four MD5 checksums match torchvision. The dataset adapter uses the same PIL
bicubic 28-to-32 resize and grayscale-to-RGB conversion as the existing MNIST runs;
normalization is calculated from FashionMNIST's 60,000 training images only.

- Full SimCLR: ResNet18 BN, 1,000 epochs, three seeds; one 100-epoch frozen linear
  probe per seed for each of the 1% and 5% label budgets.
- Random and K-means: 0.2%, 1%, 2%; one 2,000-epoch SimCLR encoder per setting,
  five 100-epoch frozen probes per label budget. K-means uses the existing
  random ConvNetGAP-BN selection recipe, not the full teacher.
- MKDT prerequisites: the unchanged historical 1,000-epoch Barlow Twins teacher
  recipe, real feature targets, 100 original 20-epoch ConvNet regression experts
  in 20 five-expert shards, and high-loss initialization indices for all three sizes.
  This does not yet distill or evaluate MKDT synthetic images.

The existing workers are reused through process-local adapters. No MNIST/SVHN
worker, prepared configuration or runtime file is edited. All new output lives
under this directory (`full`, `subset`, `mkdt`); provenance checks remain enabled.
Completed encoders/probes/expert shards are reused; interrupted SSL resumes the
rolling checkpoint. An interrupted linear head or unfinished expert restarts only
that head/expert. Full/Barlow checkpoints replace every 50 epochs, subsets every 500.

The 32 experiment/prerequisite jobs alternate Wang/Lu (16 each), plus CPU data
preparation. GPU jobs use one L40S, one CPU and 6 GB RAM. Walltimes reuse measured
MNIST runtimes for the identical 60k/10k shapes, with the same one-minute buffer;
they are estimates, not measured FashionMNIST runtimes. Submission receipts and
logs are saved here. Tables are not automatically populated from partial results.
