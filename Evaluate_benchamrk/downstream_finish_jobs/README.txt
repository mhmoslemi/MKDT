Finish empty downstream cells on Killarney, all Wang
=================================================

    bash Evaluate_benchamrk/submit_remaining_downstream_wang_killarney.sh

Only these three tables are scanned, and their values are not edited:
 CVPR_tables/cifar10_downstream.tex
 CVPR_tables/cifar100_downstream.tex
 CVPR_tables/tinyimagenet_simclr.tex

Filled numbers, --, disabled 5% rows, and every 2% Barlow Twins cell are excluded.
A blank cell with any existing completed probes is also excluded if it was not
previously approved by this new launcher. The first actual submission records
approved_cells.json. Thus rerunning can resume NEW work without picking up the
OLD partial cells you asked to ignore. Current raw-probe partials, not merely
the TeX display, determine this exclusion. The Tiny Random/Aircraft 2% SimCLR
cell has 9/10 probes and is intentionally not scheduled.

Current synchronized inventory (refreshes on Killarney at submission):
 162 blank nonexcluded-size cells.
 14 old partial cells ignored.
 148 untouched empty cells selected, including 60 MKDT 0.2% cells.
 19 missing/unfinished SSL seed jobs and 80 probe jobs, 888 probes in total.
 Without MKDT 0.2%: 88 cells, 9 SSL jobs, 48 probe jobs, 528 probes.
 Random and k-means need no jobs here: their other cells are filled.

MKDT 0.2% depends on the generation pipeline prepared immediately before this.
The submitter reuses queued mk02_* final-generation job IDs. If necessary it
first invokes submit_mkdt_02_wang_killarney.sh to submit/reuse the generation
pipeline, then submits SSL jobs afterok generation and probes afterok SSL.
The generation pipeline can add up to 83 jobs if none has been submitted yet;
those are separate from the 99 evaluation jobs above. No sleep/polling jobs.
To leave MKDT 0.2% out entirely for now:
    bash Evaluate_benchamrk/submit_remaining_downstream_wang_killarney.sh --skip-mkdt02

Saved encoders are reused. An unfinished/missing encoder resumes/trains using
the SAME train_ssl.train implementation; no new SSL objective or augmentation.
Each new cell uses two independent SSL seeds x three 100-epoch frozen linear
probes. SimCLR: MKDT 2k; Ours 2k at 0.2%/1%, retained downstream 1k at 2%.
Barlow Twins: 10k. Ours 2% main-table selection remains 2k and is not changed.
All evaluation is ResNet18BN; MKDT uses the existing ConvNet-generated tensors.
Main-table probes are not scheduled. Synthetic images are never overwritten.

Resources: aip-boyuwang, 1 CPU, 6G RAM, 1 L40S, exclude kn101, no arrays.
Time = measured p90 SSL/probe time + exactly 150 seconds per submitted job.
Up to five short three-probe bundles are packed together; long bundles stay
separate. SSL jobs are per seed, with rolling resume every 500 epochs.
Completed probe JSONs are validated/reused; an interrupted head restarts.
Console SSL progress uses the existing 200-step filter; projector_std and seconds
remain available in raw ssl.jsonl but are removed from console progress.

Manifests, skipped-cell coverage and receipts:
 Evaluate_benchamrk/downstream_finish_jobs/batches/<batch>/
Logs: downstream_finish_jobs/logs/
Six-probe summaries: downstream_finish_jobs/summaries/
Raw results stay in the existing MKDT/Ours evaluation directories for TeX updates.

Dry-run (does not submit generation, SSL or probe jobs):
    bash Evaluate_benchamrk/submit_remaining_downstream_wang_killarney.sh --dry-run

Lightweight checks (no real Slurm calls, no Python or GPU execution):
    node Evaluate_benchamrk/downstream_finish_jobs/check_plan.js
    node Evaluate_benchamrk/downstream_finish_jobs/check_launcher.js

Killarney-side file presence and metadata are checked again there. Actual GPU
execution and cross-server binary equivalence were not checked on the login node.
