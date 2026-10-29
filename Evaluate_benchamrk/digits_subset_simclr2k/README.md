# MNIST / SVHN Random and K-means main-table runs

```bash
bash Evaluate_benchamrk/submit_digits_random_kmeans_simclr2k_vulcan.sh
```

Twelve ordinary independent Vulcan jobs: two datasets × two selection methods ×
three sizes (0.2%, 1%, 2%). Each job uses one L40S, one CPU, 6 GB RAM, trains one
fresh ResNet18BN (non-AP) SimCLR encoder for 2,000 epochs, then trains five frozen
100-epoch heads at 1% labels and five at 5% labels. Selection/SSL seed is zero;
probe seeds are 0–4. Totals: 12 encoders, 120 probes, 24 five-probe table cells.
No downstream evaluation or Barlow training is included. Existing jobs untouched.
SVHN charges Wang; MNIST charges Lu (six jobs per account).

All algorithms reuse the benchmark's existing `select_subset`, `train_ssl.train`,
and `evaluate.linear_probe`. Random selection is uniform without replacement.
K-means uses the existing random ConvNetGAP-BN normalized features, 50 Lloyd
iterations and unique nearest real exemplars—not a full-data trained teacher.
The classifier trained for evaluation is ResNet18BN for both selection methods.

Input preprocessing, splits, train-only normalization, and the complete SSL/probe
recipe come from the digit full-data setup. MNIST is resized to 32px and repeated
into RGB; SVHN excludes the extra split. Percentages use each actual split size:
MNIST selects 120/600/1,200 images; SVHN selects 146/732/1,465 images. SSL batch
size is min(256, subset count), with the same drop-last and LR scaling policy as
the existing benchmark. SSL checkpoints replace `training.pt` every 500 epochs.

Rerunning skips active or finished configurations. An unfinished job reuses its
saved selection and resumes SSL including optimizer state. After SSL completion,
its encoder is retained and reused for every probe. Valid completed heads are
skipped; only an interrupted head restarts. Logs print SSL progress every 200
epochs without projector_std/seconds; the detailed JSONL remains intact.

Results are under:
`results/<dataset>/ResNet18/<random|kmeans>/pt<size>/simclr/ssl_seed_000/`.
Each has the encoder, rolling training state, config, individual probe JSONs, and
`summary_lbl1.json` / `summary_lbl5.json` (mean ± sample std once five heads finish).
No TeX files are automatically overwritten by these jobs.

Walltimes use the last 200 logged epoch timings (p95) from the existing 2k CIFAR10
ResNet18 runs, scaled to the actual effective batch/step count, plus five times
the already measured digit 1%+5% probe times, 20 seconds startup, and one minute
buffer. K-means includes two estimated minutes for feature extraction/clustering.
These are estimates; the rolling checkpoints allow recovery after timeouts.
