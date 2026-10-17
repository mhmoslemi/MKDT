MKDT SYNTHETIC-IMAGE SSL -- KILLARNEY, SIX SEEDS

Run from MKDT on Killarney after syncing these new files:
  bash Evaluate_benchamrk/mkdt_ssl_killarney.sh --dry-run
  bash Evaluate_benchamrk/mkdt_ssl_killarney.sh
The launcher has NOT been submitted by Codex. No evaluation or distillation jobs
are included; existing random/kmeans launchers and shared Python are unchanged.

The launcher now passes the absolute Evaluate_benchamrk directory explicitly to
each Slurm worker. Worker mode never derives it from BASH_SOURCE (which points
to Slurm's spool copy). It checks the worker, manifest, protocol and venv paths
before loading modules. Sync the corrected launcher to Killarney, then rerun
the same submission command to replace the failed/cancelled original jobs.
Existing synthetic images are reused; no synthetic generation is needed.

INPUTS
The default generated root is:
  benchmarks/mkdt_multidataset/results/batch_20261010T160046Z_Y6dqZS/synthetic/generated
Set MKDT_GENERATED_ROOT to change this root (the directory containing ConvNet/
and VGG11/). The exact 12 required tensors are enumerated in tasks.tsv:
  GENERATOR/CIFAR10/CIFAR10_{1,2}pct/images_5000.pt
  GENERATOR/CIFAR100/CIFAR100_{1,2}pct/images_5000.pt
  GENERATOR/TinyImageNet/Tiny_{1,2}pct/images_10000.pt
where GENERATOR is ConvNet or VGG11. File existence is checked ON KILLARNEY for
every required input BEFORE any job is submitted; tensor validation happens in
the GPU jobs, not on the login node. Only some artifacts are copied to Vulcan.

Mapping follows the generation/refresh scripts and saved configurations:
  ConvNet-generated images -> ConvNet and ResNet18 SSL backbones.
  VGG11-generated images -> VGG11 SSL backbone.
Native image size is CIFAR 32x32 and TinyImageNet 64x64 (no old 64->32 resize).
Uses data.SyntheticDataset: shape/count/floating/finite checks; inverse of the
repository GENERATION normalization; explicit clamp to [0,1], recording clipped
fraction; float-to-PIL uint8 conversion; current train_ssl.py fresh-view SSL
augmentations and protocol SSL normalization. CIFAR generation normalization
and current SSL normalization have different standard deviations: conversion
through RGB handles this intentionally. No normalized tensors reinterpreted as
raw pixels. Synthetic labels_*.pt and synlr_*.pt are neither loaded nor needed.

SCOPE
3 sources x 3 BN backbones x 2 sizes (1%,2%) x 2 SSL objectives x 6 seeds = 216
retained encoders. Six seeds mean independent SSL initializations (0..5), NOT
six independently distilled datasets. Seeds reuse each saved synthetic set.
No 0.2% generation/artifact was found in the scripts/input registry; those MKDT
TeX rows are blank multicolumn placeholders. No subset of a 1% tensor is silently
called a distilled 0.2% set. Existing 5% tensors are not used because those table
rows are disabled and the current random/kmeans schedule does not include 5%.
The older table_coverage.tsv mechanically lists 0.2% MKDT cells despite the TeX
placeholders; this launcher explicitly covers only the available active 1%/2%.

RECIPE -- THE SAME train_ssl.train USED BY CURRENT RANDOM/KMEANS
  CIFAR10/CIFAR100: 1% -> 8000 epochs, 2% -> 6000 epochs per seed.
  TinyImageNet:    1% -> 5000 epochs, 2% -> 4000 epochs per seed.
  ConvNetBN, VGG11BN, normal ResNet18 (BN; not AP).
  Batch 256, LARS, warmup/cosine, loss/projector/augmentation settings unchanged
  from protocol.json. No legacy benchmark_mkdt_refresh/multidataset training.
  Overwrite training.pt every 500 epochs, including optimizer/RNG state. Rerun
  resumes with the same full schedule. Final encoder.pt is strictly reloaded.
  Both the final encoder and ONE latest rolling checkpoint are retained.
  Real source TRAIN data is fingerprinted for future evaluation provenance;
  only the synthetic dataset is supplied to SSL. No downstream data is loaded.

JOBS AND TIMING
  144 ordinary jobs: 72 pairs + 72 single-seed jobs; no arrays/dependencies.
  Two seeds per job unless the projected time + 15min exceeds 5h; those configs
  get one seed/job. Thus Tiny 1%/2% use single seeds; CIFAR uses pairs.
  A single seed can still exceed 5h. No truncation or additional training stages.
  Estimated requested walltimes: 01:55:29--07:17:40, including exactly 15min extra.
  timing_estimates.json records current-trainer Killarney random-subset reference
  rates and the formula. These are NOT measured synthetic-image runtimes.
  MKDT_SSL_TIME=HH:MM:SS can override all requested times without recipe changes.
  1 L40S GPU, 1 CPU, 6G, workers=0, numerical threads=1.
  115 Wang / 29 Lu (~80/20, rounded as for the Killarney random launcher).
  Alliance Slurm/software skill checks: verify cluster, own account access,
  exact GPU type and Python module before submission; no forced partition.

OUTPUTS
  Evaluate_benchamrk/mkdt_ssl_checkpoints_cifar8k6k_tiny5k4k_6seeds/
    SOURCE/MODEL/mkdt/ptSIZE/SSL/seed_NNN/
      encoder.pt, training.pt, config.json, ssl_complete.json, ssl.jsonl
    logs/ and submitted_*.tsv
Configuration records generator, input path/SHA256, conversion diagnostics,
source training fingerprint, complete effective recipe, code hashes and seed.
New worker code lives in this subdirectory so the top-level code hashes of
already-running random/kmeans jobs do not change. No existing files are deleted.

BENCH_VENV (default $HOME/ENV), BENCH_DATA_ROOT (default MKDT/../data), and
BENCH_OUTPUT_ROOT (dedicated folder inside Evaluate_benchamrk) can be overridden.
The launcher skips matching active jobs or groups with all final files present.
An unfinished group validates/resumes its existing seeds. Changed inputs/code
cause a provenance error, not silent mixing. Use a new output root for a changed
recipe or regenerated input. The current random/kmeans evaluation launchers do
not include MKDT; these saved encoders are for a subsequent MKDT probe batch.

EVIDENCE CHECKED
  benchmarks/killarney_short_jobs/check_mkdt_distilled_data.sh
  benchmarks/killarney_short_jobs/synthesis_part3_job.sh
  benchmarks/killarney_mkdt_refresh/job.sh and its saved encoder config.json files
  distill.py and the generation snapshot's utils.py normalization constants
  Evaluate_benchamrk/data.py, train_ssl.py, protocol.json
