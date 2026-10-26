# A²MIM full-data teachers

Three independent, from-scratch ResNet18BN teachers: CIFAR10, CIFAR100,
TinyImageNet. One seed (0), 300 epochs each. These jobs only train teachers.

Submit on Vulcan:

```bash
bash Evaluate_benchamrk/submit_a2mim_teachers_vulcan.sh
```

CIFAR jobs use `aip-boyuwang`, TinyImageNet uses `aip-yiweilu`. Each requests
one L40S, one CPU and 6 GB host RAM. Requested limits are 5 h for each CIFAR
and 23 h for Tiny: rounded from the existing full-data SimCLR throughput
proxy, not measurements of this new method.

## Recipe and upstream provenance

This is a PyTorch port of the authors' 300-epoch CNN configuration, not a
claim that their ImageNet/ResNet50 results transfer unchanged to ResNet18.
Pinned official code/configuration and Apache-2.0 license are in `reference/`;
their commits and file checksums are in `sources.json`.

Preserved: 60% block masking with ceil rounding, per-image mean fill, residual
learnable feature token with gamma initialized to 1e-6 before layer4, 1x1
convolution plus PixelShuffle reconstruction head, focal L1 pixel loss
(alpha .2, gamma 1, sigmoid), and .5 times focal frequency loss (average
spectrum, logarithmic batch weights, mixed detached replacement, .001
unmasked weight). AdamW: betas (.9,.999), eps 1e-8, weight decay .05;
no decay on BN/bias/mask token; mask-token LR multiplier .1. Float32.
RandomResizedCrop scale (.8,1), ratio (.75,4/3), bilinear interpolation,
horizontal flip. Ten warm-up epochs then official StepFix cosine scheduling.
Peak LR is .0012 * batch / 2048 = .00015 at batch 256; minimum LR .00001.

Explicit small-image adaptations: existing repository ResNet18BN with a
3x3/stride1 stem, no initial maxpool, and output stride 8; native 32px CIFAR
and 64px Tiny inputs; dataset-specific benchmark normalization; 8px mask
blocks; 256-channel token before layer4; stride-8 decoder. Single-GPU BN
replaces multi-GPU SyncBN. Batch 256 replaces global 2048, with official LR
scaling. All training images are used; labels are ignored. No validation or
test images participate in training or checkpoint selection.

Each job first compares the port's loss AND gradients with pinned official
A2MIMHead/FocalFrequencyLoss definitions and checks both native-size backbones.
No OpenMixup/MMCV installation or internet is needed in the jobs.

## Checkpoints and resume

Final frozen-backbone-compatible output:

`checkpoints/<dataset>/ResNet18/full/pt100/a2mim/seed_000/encoder.pt`

`training.pt` is atomically replaced every 10 epochs, retaining model,
decoder/token, AdamW, counters and RNG. A Slurm termination warning saves
the current batch; resubmitting the same launcher resumes it. Partial-epoch
resume regenerates and skips already consumed shuffled inputs, then restores
the saved RNG before training continues. Complete and currently queued jobs
are skipped. `encoder.pt` uses the existing `common.load_encoder` format;
the reconstruction head and masking token are omitted from this export.
Full resume state remains separately in `training.pt`.

The existing Ours CLI currently restricts teacher methods to SimCLR/Barlow;
selecting these A²MIM teachers there will require adding the `a2mim` choice
when generation is requested. Existing synthetic-generation/evaluation code
and its benchmark fingerprints have not been changed.
