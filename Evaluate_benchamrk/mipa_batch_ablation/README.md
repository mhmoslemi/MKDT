CIFAR10, 1% synthetic data (500 images), ResNet18BN.

Run `bash Evaluate_benchamrk/submit_mipa_batch_ablation_vulcan.sh` on Vulcan.
The six real-data batches are 128, 256, 512, 2048, 4096 and 8192. Batch 1024
already exists and has no new generation, SSL or probe job in this launcher.
Jobs alternate Wang/Lu (three each on the first submission).

Generation calls the existing `OT-SSL-DD/method1.py` unchanged: 8000 iterations,
Adam lr 0.001, seed 0, pretrained full-data SimCLR teacher seed 0, temperature
0.2, DSA, feature chunks of 128, displacement/pair weights both zero. The same
random-feature K-means initialization and seed are used for every batch.

Each job then trains one reference ResNet18 SimCLR encoder (seed 0, 2000 epochs,
SSL batch 256), followed by five 100-epoch frozen probes (seeds 0--4, 1% labels).
Each summary reports mean and sample standard deviation. No downstream probes.

Synthetic outputs: `OT-SSL-DD/ours_syntehtic_batch_ablation/CIFAR10/ResNet18BN/pt1/batch_<N>/`.
Each holds one replacing PT checkpoint plus first/final PNGs. Evaluation outputs:
`Evaluate_benchamrk/mipa_batch_ablation/results/batch_<N>/simclr/ssl_seed_000/`.
Logs, manifests and submission receipts are under this ablation directory.

Rerunning skips active/completed jobs and resumes partial jobs. Generation saves
Adam/sampling/RNG state every 100 iterations; SSL saves every 500 epochs.
Completed probe JSONs are reused; an interrupted head restarts. Do not edit the
recorded implementation or teacher between submission and completion.

Walltimes use prior CIFAR10 generation/SSL/probe timings plus one minute buffer.
The batch-size scaling is approximate, and each manifest records its estimate.
The main tables are not automatically changed by these ablation jobs.
