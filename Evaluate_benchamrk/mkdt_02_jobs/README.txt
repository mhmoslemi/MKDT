MKDT 0.2% generation on Killarney (Wang account)
================================================

From MKDT on Killarney:
    bash Evaluate_benchamrk/submit_mkdt_02_wang_killarney.sh

Generation only: no SSL evaluation or table editing is submitted.
Current ResNet18 evaluation uses ConvNet-generated MKDT images, as before.
Generation models are the original MKDT ConvNet (instance norm), not the
benchmark ConvNetBN used to evaluate the resulting images.

Fresh submission: 83 ordinary jobs, no arrays:
 - 60 expert/ranking jobs: 20 per dataset, five complete experts per job.
 - 3 CPU initialization-assembly jobs, depending on all 20 expert shards.
 - 20 sequential GPU distillation jobs: five CIFAR10, five CIFAR100, ten Tiny.
Each distillation job does up to 1,000 ADDITIONAL updates of the SAME dataset,
not a separate synthetic dataset. All jobs use aip-boyuwang, 1 CPU, 6G RAM.
GPU jobs request one L40S and exclude kn101. Initialization assembly needs no GPU.

Walltimes (include one 2m30s buffer per job; approximate old-log estimates):
                         Expert/ranking   CPU assembly   1000 distill updates
 CIFAR10 / CIFAR100       00:12:30         00:03:30       00:27:30
 Tiny ImageNet           00:32:30         00:03:30       00:37:30
Tiny 0.2% timing has no direct measured run: its limit is an extrapolation.
Streaming input and buffer file loading differ from the old preloaded timing.
These estimates are not a runtime guarantee. Re-run the submitter on failure.

Pipeline and fixed recipe:
 1. Reuse saved Barlow-Twins full-data teacher feature tensors; no teacher retraining.
 2. Build all 100 independent KD regression expert trajectories per dataset.
    Each expert: 20 epochs, MSE against cached teacher features, SGD lr=.1,
    momentum=.9, weight decay=1e-4, batch=256; seed=expert index.
    CIFAR10/100 use ConvNet; Tiny uses ConvNetD4 at native 64px.
 3. Score every real train image using epoch-1 weights of EVERY rebuilt expert.
    Average the 20 equally sized five-expert shard score vectors and select
    the highest-loss 0.2%: 100 CIFAR images or 200 Tiny images.
    This recomputes initialization against the new banks, not old 1% images.
 4. Run MKDT parameter-trajectory matching. Optimize synthetic pixels,
    synthetic regression labels and a learned student LR, with SGD momentum=.5.
    MSE inner regression; normalized squared parameter-distance outer loss.
    Match two expert epochs; start epoch uniformly in {0,1}; initial student
    LR=.1; image LR=1000 CIFAR / 100000 Tiny; label LR=1; LR optimizer=1e-4.
    CIFAR: 5000 outer updates, 40 synthetic steps/update.
    Tiny: 10000 outer updates, 10 synthetic steps/update.
    Synthetic batch cap=256, so all 100/200 images fit in one synthetic batch.
    No new augmentation, VAE, labels from real classes, or additional losses.

Correctness fixes to distill.py:
 - Actually load the next replay-buffer file when advancing the expert cursor.
   Previously only the filename advanced; the old expert stayed loaded.
 - Export images AFTER their numbered update, not before it.
 - Resume images, synthetic labels, learned LR, all three optimizer momenta,
   expert file order/cursor and Python/NumPy/CPU/CUDA RNG states.
 - Replace training.pt atomically every 100 updates and at each chunk boundary.
   A timeout loses at most the updates since the last rolling save.
 - Load only selected real initialization pixels instead of the whole float dataset.

Recovery:
Run the SAME submission command again. It reuses active job IDs, skips completed
expert/ranking and distillation chunks, and adds afterok dependencies to missing
work. A killed expert is retrained from its fixed seed; finished experts remain.
An interrupted distillation resumes training.pt, not just the exported image tensor.
An allocated first expert shard checks resumed versus uninterrupted optimization
on a tiny artificial example before touching production expert training.
Changing generation source code after submission is rejected to prevent mixing
recipes; code is snapshotted per submission. Existing data are never deleted.

Outputs:
 Inputs and final image sets remain in:
 benchmarks/mkdt_multidataset/results/batch_20261010T160046Z_Y6dqZS/synthetic
 Final new sets are:
 generated/ConvNet/CIFAR10/CIFAR10_0.2pct/images_5000.pt
 generated/ConvNet/CIFAR100/CIFAR100_0.2pct/images_5000.pt
 generated/ConvNet/TinyImageNet/Tiny_0.2pct/images_10000.pt
 Each folder also retains synthetic labels/LR, initial/final PNG grids,
 intermediate image exports, rolling training.pt and complete_0.2.json.
 New expert banks, ranking vectors and launch receipts/logs live under:
 Evaluate_benchamrk/mkdt_02_jobs/runs/
 This is outside the old delete_mkdt_teachers.sh buffer globs.

Optional path overrides: MKDT_SYNTH_ROOT, BENCH_DATA_ROOT, BENCH_VENV.
Default venv is $HOME/ENV; Python module is StdEnv/2023 python/3.11.5.
The submitter verifies Killarney, L40S availability and your Wang association there.

Method source: https://github.com/jiayini1119/MKDT
0.2% is our smaller-budget extension of the existing 1%/2% configuration,
not a separately published/tuned official 0.2% recipe. Existing 1%/2% images
are not regenerated or altered. Existing evaluators still exclude 0.2% MKDT;
adding the new generation outputs to evaluation is a separate next step.

Local lightweight launcher check (no Python, GPU or Slurm calls):
    node Evaluate_benchamrk/mkdt_02_jobs/check_launcher.js
