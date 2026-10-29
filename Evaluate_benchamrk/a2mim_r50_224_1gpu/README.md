# ResNet50 A²MIM, original OpenMixup implementation

Pinned upstream: `Westlake-AI/openmixup@37e1372446ab35c03fcf77302539bd4a5bee47eb`.
Recipe: `configs/selfsup/a2mim/cifar100/r50_l3_sz224_init_4xb256_accu2_cos_ep1000.py`.

CIFAR10 and CIFAR100 each have one full-data encoder (1,000 epochs) and Random,
MKDT and MIPA at 0.2% and 1% (10,000 epochs). These reuse the original seed-0
random indices and saved synthetic pixels, not the newer large-batch MIPA sets.

The agreed one-GPU adaptation uses 64 images per microbatch, eight-way gradient
accumulation, local BatchNorm and zero data-loader workers. Original 224px crops,
32px masks, stride-32 ResNet50, decoder, reconstruction/frequency objectives,
AdamW parameter groups, warmup, cosine schedule and AMP remain upstream code.
The executable four-GPU recipe uses 64 images/GPU and two-way accumulation;
this is not an exact four-GPU reproduction. Incomplete final batches are kept.
For 100-image sets, eight microbatches contain 400 sampled images, not 512.
Single-rank Gloo avoids a native startup crash in the legacy NCCL build;
model computation remains on the GPU, with no other ranks to synchronize.

Full training saves a rolling optimizer/scaler checkpoint every 10 epochs;
subset training saves every 500 epochs. The launcher uses upstream auto-resume.
Completed encoders and completed probe JSONs are reused after validation.
An interrupted linear head restarts; already completed heads are retained.

Each encoder receives five independent 100-epoch frozen linear heads for each
target (source dataset, other CIFAR dataset, TinyImageNet) and 1%/5% labels.
Probes use 224px crops (scale 0.2–1), horizontal flips, a 256px resize/224px
center crop at test time, and the upstream training normalization. SGD uses
momentum 0.9, zero weight decay and cosine learning rate: 0.1 for CIFAR10,
1.0 for CIFAR100, 0.1 for TinyImageNet (the last is our transfer adaptation).

Total: 14 encoders, 84 cells, 420 heads; 14 training and 28 bundled probe jobs.
Jobs alternate Wang/Lu and request one L40S, one CPU, 6GB host RAM. Walltimes
come from the allocated-GPU training/probe measurements, including MMCV's
two-second epoch overhead, startup and a one-minute buffer.

Submit or resume from the repository root:

```bash
bash Evaluate_benchamrk/submit_a2mim_r50_224_1gpu_vulcan.sh
```

The setup and dispatcher validate the original upstream model, gradient updates,
strict encoder export, actual AMP trainer and optimizer checkpoints before
releasing the cohort. Submission receipts are under `batches/`; logs under
`logs/`; final encoders and per-cell summaries under `checkpoints/`.
Existing ResNet18 tables and results are not overwritten.
