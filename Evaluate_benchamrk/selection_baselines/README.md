# SAS and ProbCover main-table baselines

Run `bash Evaluate_benchamrk/selection_baselines/submit_lu.sh` on Vulcan.
This submits three shared selection jobs followed by eighteen evaluations, all
on Lu (`aip-yiweilu`, one L40S). Each evaluation trains one fresh ResNet18 encoder
for 2,000 SimCLR epochs and five 100-epoch probes at each of 1% and 5% labels.
It reuses `Evaluate_benchamrk/{data,train_ssl,evaluate,common}.py` and the existing
protocol. Training checkpoints resume automatically; completed probes are reused.
Jobs save summaries but do not rewrite any tables.

Selection uses only the training images and the frozen full-data SimCLR teacher:
CIFAR10/CIFAR100 seed 0, TinyImageNet seed 1, all completed 1,000-epoch teachers.
Features are pre-projection, extracted with the deterministic reference test
transform and L2-normalized. Both methods use Euclidean k-means latent groups
(10/100/200 groups, seed 0, maximum 1,000 iterations). No ground-truth class
assignments, class names, probe labels or test images enter selection.

## SAS

Official repository: https://github.com/sjoshi804/sas-data-efficient-contrastive-learning
Pinned revision: `5e5741884c2e5dcdea65467989b1ab0720430eca`.
The two files in `upstream/sas/` are unchanged copies of the released
`sas-pip/sas/{subset_dataset,submodular_maximization}.py`.
We use its `SubsetSelectionObjective` (threshold zero) and `lazy_greedy` directly,
with cosine similarity calculated by normalized matrix multiplication. The
release ranks a whole cluster before slicing a prefix; we stop at the largest
required prefix, which gives the same selection. Per-cluster fractional budgets
use largest-remainder rounding so every saved subset has exactly the requested
100/500/1,000 CIFAR or 200/1,000/2,000 TinyImageNet images. This runs the released
selector; no additional refinement absent from the released code is introduced.

## ProbCover

Official repository: https://github.com/avihu111/TypiClust
Pinned revision: `4097a71c348f60492ab22be0c4c9da224e637af6`.
Reference: `deep-al/pycls/al/prob_cover.py` and paper Section 3.2:
https://arxiv.org/abs/2205.11320.
We start with an empty selected set and use its maximum-uncovered-neighborhood
greedy rule, including deterministic index-order tie breaking. A sparse CSR/CSC
graph replaces the repeated pandas filtering, without changing the rule.
We select the largest strict radius keeping at least 95% of balls cluster-pure,
computed exactly from each point's nearest point in another unlabeled cluster.
This implements the paper's radius criterion on our teacher embeddings rather
than reusing a radius tuned for the authors' feature space. Partial final batches
are included and already selected vertices remain excluded after full coverage.

## Artifacts and checks

`selections/DATASET/features.pt` saves the features, clusters and teacher identity;
`{sas,probcover}_ptSIZE.json` saves exact indices, fingerprint and recipe.
`results/DATASET/METHOD/ptSIZE/simclr/ssl_seed_000/` saves the encoder, individual
probes and mean +/- sample SD summaries. `submitted.tsv` records dependencies.
The manifest hashes the source and teachers. It refuses to mix changed code or
configurations into an existing run.

`test_selection.py`, run inside a Slurm GPU allocation, checks SAS against the
released full-order prefixes, exact budgets, radius purity, sparse selection
against a dense reference, graph remainder batches, full-coverage uniqueness,
and a short real-data SSL/checkpoint/frozen-probe round trip.
