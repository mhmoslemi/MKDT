MIPA initialization ablation: CIFAR10, 0.2% (100 images), ResNet18BN.

Submit on Vulcan: `bash Evaluate_benchamrk/submit_mipa_init_ablation_vulcan.sh`.

Two distillation jobs:

- Random real-image initialization, three sets (initialization seeds 0, 1, 2).
- Random-feature K-means initialization, one set (seed 0), maximum 100 Lloyd
  iterations. The convergence check and nearest-real-image rule match method1.
  The actual iteration count is recorded in initialization.json.

The existing five-iteration K-means set in
`OT-SSL-DD/ours_syntehtic/CIFAR10/ResNet18BN/pt0.2/` is the reference; no job
regenerates or retrains that reference.

All four new sets use the original 0.2% recipe: 10000 distillation iterations,
Adam lr 0.001, real batch 1024, teacher ResNet18BN SimCLR seed 0, temperature
0.2, DSA, displacement/pair weights zero. Only initialization varies. The
distillation sampling/augmentation seed stays 0, including all three random
initializations; initialization has its own random seed.

Each set gets a dependent evaluation job: one 2000-epoch reference SimCLR
encoder, SSL seed 0, followed by five 100-epoch frozen probes with 1% labels,
probe seeds 0--4. This gives two generation jobs and four evaluation jobs,
three Wang and three Lu. Each uses one L40S, one CPU and 6 GB.

Synthetic outputs:
`OT-SSL-DD/ours_syntehtic_init_ablation/CIFAR10/ResNet18BN/pt0.2/<variant>/`.
Each retains one replacing PT and first/final PNGs, plus provenance JSONs.
Evaluation outputs: `Evaluate_benchamrk/mipa_init_ablation/results/<variant>/`.

Rerun the submitter to skip active/completed work and resume interrupted runs.
Distillation saves Adam/sampling/RNG state every 100 iterations; SSL checkpoints
every 500 epochs. Completed probes are reused; an interrupted probe restarts its
head. The worker overrides initialization only within its own process, so the
existing method1 file and other running experiments are unchanged.

Walltimes use matched prior 0.2% generation/SSL/probe logs plus a 60-second
buffer. The 100-iteration K-means initialization allowance is approximate.
