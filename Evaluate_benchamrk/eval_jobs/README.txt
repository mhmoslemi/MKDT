CURRENT SSL CHECKPOINT EVALUATION -- NEW LAUNCHERS

Run from the MKDT repository, AFTER the SSL runs finish:
  Vulcan:   bash Evaluate_benchamrk/eval_kmeans_vulcan.sh
  Killarney: bash Evaluate_benchamrk/eval_random_killarney.sh
Append --dry-run to preview every job without writing files or contacting Slurm.
Sync the NEW launchers/eval_jobs directory AND the same shared code used for SSL
to Killarney first. Checkpoints remain on their original server; Git is not a
checkpoint transfer mechanism. These scripts have not been submitted by Codex.

Scope (120 ordinary jobs PER SERVER, 240 total; no arrays or dependencies):
  kmeans: 3 sources x 3 BN backbones x 3 subset sizes x 2 SSL methods x 6 seeds
          = 324 retained encoders, 1,944 probes, 324 table cells.
  random: same configurations x 10 seeds
          = 540 retained encoders, 3,240 probes, 324 table cells.
Every encoder gets its main-dataset 1% and 5% probes. ResNet18 also gets both
label budgets on all six downstream targets. ConvNet and VGG11 have only their
main-dataset cells in the existing table contract. No full-data/MKDT evaluations.

Resource/account settings preserved from the respective SSL launchers:
  1 L40S GPU, 1 CPU, 6G RAM, 0 loader workers, 1 numerical-library thread.
  Vulcan kmeans: 108 Wang / 12 Lu. Killarney random: 96 Wang / 24 Lu.
The Alliance Slurm/software skill checks are implemented at submission: verify
cluster, the user's two accounts, L40S inventory, and Python/3.11.5 availability.
No partition is forced. Default venv is $HOME/ENV; data is MKDT/../data.

INPUTS (not the obsolete 1000-epoch random checkpoint directory):
  kmeans_ssl_checkpoints_cifar10k8k6k_tiny6k5k4k_6seeds/
  random_ssl_checkpoints_cifar10k8k6k_tiny6k5k4k/
Seed paths: SOURCE/MODEL/METHOD/ptSIZE/SSL/seed_NNN/
Require config.json, ssl_complete.json and final encoder.pt for EVERY seed
before any job is submitted. training.pt is NOT accepted as a finished encoder.
Workers check completed epochs, config/identity hashes, code hashes, actual
encoder tensors, source data fingerprint and the original SSL worker digest.
They expect CIFAR 10000/8000/6000 and Tiny 6000/5000/4000 epochs for .2/1/2%.
Adding a new top-level Python file would change the SSL code inventory: this
driver lives in eval_jobs/ deliberately. Existing SSL/evaluate/data/network
code, protocol.json, old launchers and TeX tables are NOT changed.

PROBING -- reuse the checked implementation in ../evaluate.py:
  Fresh linear head per target, label budget and encoder seed.
  100 epochs, SGD lr=0.1, momentum=0.9, weight_decay=0, batch<=256.
  Cosine per-step LR; exact nested class-balanced 1%/5% training-label budgets.
  Random resized crop and flip every training epoch; source image resolution
  (CIFAR 32, Tiny 64) and source normalization. Deterministic test transforms.
  Encoder parameters AND BatchNorm running statistics frozen throughout.
  Test once after the final epoch, never choose a head using test accuracy.
  This is the selected small-label benchmark protocol, not a claim that the
  original SimCLR/Barlow ImageNet papers share one universal probing recipe.
  Very small budgets can omit classes: e.g. Flowers official train has 1020
  images, so 1% is 10 labels, not one label per class. No silent budget expansion.

TIMING (approximate, not measured evaluation walltimes):
  kmeans tasks: 01:08:59--01:17:13; random tasks: 01:48:37--01:49:58.
  Estimated total probe work is unchanged, balanced across six times as many
  jobs. Each encoder's probes stay together; indivisible encoder workloads and
  the fixed 15-minute margin prevent an exact sixfold walltime reduction.
  Exact task lists and time requests are in *_tasks.tsv and *_jobs.tsv.
  timing_estimates.json records the source logs, formula and assumptions.
  No current production-probe timings were found. The proxy is native SSL
  seconds per two processed image views, halved to approximate one probe view,
  plus an UNMEASURED 5ms/image JPEG decode allowance and 10s/probe load overhead.
  SSL backprop/projector costs make this proxy conservative for native images;
  JPEG sizes, storage contention and cross-cluster differences remain uncertain.
  Each job requests its summed estimated work + exactly 900 seconds. No claim
  is made that these walltimes are minimal or guaranteed. The old cached-feature
  probe timings are not applicable to repeated fresh-crop encoder forwards.
  BENCH_EVAL_TIME=HH:MM:SS can override walltime without changing the protocol.

OUTPUTS, all inside Evaluate_benchamrk:
  kmeans_eval_results_current_6seeds_120jobs/
  random_eval_results_current_10seeds_120jobs/
  New output/job names prevent mixing completion markers with the 20-job plan.
  logs/                     Slurm stdout/stderr
  submitted_*.tsv            submitted job IDs/accounts
  plan.json                 immutable plan, code hashes and SSL worker hash
  jobs/                     environment, completion and failure/elapsed reports
  SOURCE/MODEL/METHOD/.../seed_NNN/evaluations/  per-probe JSON + epoch logs
  checklist.tsv             each seed/probe and its status
  table_summary.tsv         mean +/- sample SD, only when ALL 6/10 seeds validate
  progress.json             expected/completed encoders, probes and table cells
  probe_times.tsv           actual completed-probe seconds for future sizing
  SOURCE/MODEL/METHOD/.../summary.json           per-configuration summaries
No table cell is automatically written into TeX; no checkpoint is deleted.

Rerun the same launcher after a failure/timeout. Active and completed jobs are
skipped. Completed probes within an unfinished job are validated and reused.
An unfinished probe restarts its linear head, not SSL. Missing inputs or changed
code/identity are errors, not silently skipped seeds. Keep the evaluation code
fixed until this batch finishes; use a fresh results directory for a new recipe.

Path overrides: BENCH_VENV, BENCH_DATA_ROOT, BENCH_CHECKPOINT_ROOT,
BENCH_EVAL_OUTPUT_ROOT (must be inside Evaluate_benchamrk, outside checkpoints).
These launchers never contact the other server; run each on its intended cluster.
