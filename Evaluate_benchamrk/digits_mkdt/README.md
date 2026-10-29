# SVHN / MNIST MKDT prerequisites

```bash
bash Evaluate_benchamrk/submit_digits_mkdt_vulcan.sh
```

This does not cancel, alter, or depend on the separate SimCLR jobs.
It prepares exactly one MKDT teacher and one reusable expert bank per dataset.
It does not submit synthetic distillation or SSL evaluation yet.

The short preparation job runs and saves the first **production** Barlow epoch
for each dataset with the complete 1,000-epoch LR schedule. That epoch is resumed,
not discarded. Measured epoch times determine the two remaining teacher job
walltimes. It then submits 46 ordinary jobs, no arrays:

- Two Barlow Twins teacher jobs, one seed 0 per dataset.
- Two dependent, unaugmented full-training-split feature extraction jobs.
- Forty dependent expert/ranking jobs: 20 per dataset, five experts per job.
- Two CPU jobs assembling high-loss initialization indices for 0.2%, 1%, 2%.

SVHN jobs charge Wang; MNIST jobs charge Lu, matching the previous per-dataset
account split. GPU stages request one L40S, one CPU, 6 GB RAM. Assembly requests
one CPU and 6 GB, no GPU. Rerunning the command preserves completed stages,
reuses active job IDs, and submits missing dependent stages.

## The existing MKDT recipe, not the newer benchmark Barlow recipe

Teacher components and schedule come from root `train_teacher.py` unchanged:
torchvision ResNet18 with 3x3 stride-one stem and no max-pool, BN, 512-dimensional
backbone features; projector 512→512(BN/ReLU)→1024; original Barlow normalization
and objective with lambda 0.0078125. Adam lr 0.01, weight decay 1e-6, batch 256,
ten-epoch warmup, cosine decay to 0.001 times base LR, 1,000 epochs, seed 0.
The same `TwoViewTransform` is used. One rolling training checkpoint, including
optimizer and RNG state, is replaced every 50 epochs; final backbone saved once.
The teacher jobs resume the saved first epoch with the unchanged schedule.

Digit pixels/splits/order and train-only normalization come from the already
prepared SimCLR dataset adapter: SVHN train only (no extra split); MNIST resized
28→32 with bicubic interpolation and repeated into RGB. This is the necessary
new-dataset adapter, not a change to the MKDT loss or training schedule.

Feature extraction uses that final teacher in eval mode, without augmentation,
and keeps the 512-dimensional training features in dataset order. No class labels
are used as regression targets. Final teacher file, feature tensor, and their
provenance/checksums are retained.

Expert training and scoring call the existing `buffer.main` and `sort_dset.main`
without changing their algorithms. Only `get_dataset` is supplied process-locally
for these new datasets. Experts use MKDT's original **ConvNet with instance norm**,
not the ConvNetBN used for benchmark evaluation. Each of 100 independent seeds
trains for 20 epochs, MSE to the fixed teacher features, SGD lr 0.1, momentum 0.9,
weight decay 1e-4, batch 256. Each buffer stores initialization plus 20 trained
parameter snapshots. Scoring uses epoch-one weights of every expert; shard means
are averaged equally before selecting highest-loss initialization indices.
The same 100-expert bank is reused across all three synthetic sizes.

Completed experts survive shard retries; only an interrupted expert is repeated,
as in the previous implementation. Each expert is exported atomically. Targets,
code, dataset order and completed buffers are validated before reuse. Source files
already used by other jobs are not modified.

## Outputs (all under this directory)

```
teachers/<dataset>/barlow_twins_resnet18_<dataset-lowercase>.pt
teachers/<dataset>/training.pt
target_rep/barlow_twins/<dataset>_target_rep_train.pt
work/<dataset>/buffers/<dataset>/ConvNet/replay_buffer_0.pt ... replay_buffer_99.pt
work/<dataset>/losses_000.pt ... losses_095.pt
work/<dataset>/<dataset>_barlow_twins_{0.2,1,2}_high_loss_indices.pkl
work/<dataset>/init_complete.json
logs/
submitted_*.tsv
```

Teacher limits are actual measured first-epoch projections plus startup/checkpoint
overhead and a one-minute buffer. Expert limits conservatively scale the old
CIFAR10 job 1939145 (one newly trained expert plus five-expert scoring/validation)
to five experts and the digit training-set size, plus one minute. Those expert
limits are estimates, not measured full new-dataset shard times. Feature and CPU
assembly jobs have three- and four-minute limits respectively.
