FULL-DATA SSL -- VULCAN, THREE SEEDS, 90% WANG / 10% LU

Run from MKDT on the Vulcan login node:
  bash Evaluate_benchamrk/full_ssl_vulcan.sh
Optional read-only preview:
  bash Evaluate_benchamrk/full_ssl_vulcan.sh --dry-run

54 ordinary jobs, exactly ONE seed per job:
  3 datasets x 3 BN backbones x 2 SSL methods x seeds 0,1,2.
  CIFAR10, CIFAR100, TinyImageNet; ConvNetBN, VGG11BN, ResNet18BN.
  49 jobs use aip-boyuwang and 5 use aip-yiweilu (~90%/10%).
  1 L40S GPU, 1 CPU, 6G; workers=0, numerical threads=1.
  No arrays, automatic submissions by the worker, dependencies, probes, or generation.

TRAINING
  Use 100% of the source TRAIN split, not test/validation data.
  Labels are not used by the SSL objective. Batch shuffle/drop_last follows
  the existing train_ssl.py implementation.
  1000 SSL epochs, batch 256, LARS and the existing method-specific recipe:
  warmup/cosine, losses, projectors, augmentations, native CIFAR 32px / Tiny 64px.
  Reuses common.py, data.RawDataset and train_ssl.train(..., full=True).
  No changes to shared Python, networks.py, or protocol.json.
  Only the new worker's effective protocol sets full_seeds=[0,1,2] and
  checkpoint_every_epochs=50; these settings are saved with each checkpoint.
  This is the agreed literature-based CIFAR recipe adapted to our benchmark,
  not a claim of exact official ImageNet reproduction.

CHECKPOINTS AND RESUME
  Evaluate_benchamrk/full_ssl_checkpoints_1000ep_3seeds/
    SOURCE/MODEL/full/pt100/SSL/seed_NNN/
      training.pt       latest SSL model/projector/optimizer/RNG state
      encoder.pt        final encoder, saved only after training completes
      config.json       effective protocol, input/code fingerprints, seed
      ssl_complete.json final metadata and checkpoint hashes
      ssl.jsonl         per-epoch training statistics
      environment.json  runtime metadata
    logs/ and submitted_*.tsv

  ONE training.pt per seed is atomically REPLACED at epochs 50,100,...,1000.
  No epoch-numbered checkpoint accumulation. Atomic saves use a temporary file
  before replacement; an interrupted write leaves the prior checkpoint intact.
  Retain the final encoder and the latest rolling checkpoint. Delete nothing.
  A rerun skips matching live jobs and existing final encoder/marker pairs.
  Unfinished workers validate identity and resume from training.pt. Work after
  the latest successfully saved checkpoint must be repeated after interruption.
  An interruption before epoch 50 has no periodic checkpoint to resume.
  A final encoder saved just before a crash can recover its missing marker.
  No automatic requeue: rerun the same launcher when a failed job is no longer live.

PATHS
  BENCH_VENV defaults to $HOME/ENV.
  BENCH_DATA_ROOT defaults to the data directory alongside MKDT.
  BENCH_OUTPUT_ROOT can select a separate directory inside Evaluate_benchamrk.
  The submit-side absolute repository path is passed to every worker, so Slurm's
  spool copy cannot redirect worker.py into a nonexistent spool subdirectory.
  Source files, software, both accounts, GPU type, and disk usage are checked before
  submission. Full dataset content checks happen in the allocated GPU jobs.

TIMING
  tasks.tsv contains all 54 jobs; timing_estimates.json records source log paths,
  measured tail windows and calculations. Requested times are full-data step
  projections plus exactly 15 minutes, NOT measured full-data walltimes:
    CIFAR: about 12--18 hours per seed.
    TinyImageNet: about 44--75 hours per seed.
  Estimates use the median steady-state step rate across completed Vulcan 2%
  reference seeds. Per-epoch timers exclude checkpoint writes; the +15min margin
  must also cover startup/checkpoint overhead and may not cover node slowdowns.
  Every request is below the observed 7-day GPU batch maximum. No partition is
  forced; Slurm routes by walltime. Queue delay is not included.
