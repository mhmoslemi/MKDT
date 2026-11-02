MIPA versus Random: CIFAR10/CIFAR100 at 0.2%, 1%, and 2%

Sources are the synthetic sets referenced by the current main table:
Evaluate_benchamrk/mipa_cvpr_color_crop/manifest.json.
Random uses baseline seed 0 (selection RNG seed 1000), verified against
the synthetic checkpoint's actual initial pixels. Nothing is retrained.

Per dataset:
  real_features.npy  All 50,000 real training images' normalized teacher features.
  real_labels.npy    True classes, used only for point colors.

Per dataset/pt{0.2,1,2}:
  inputs.json       Exact checkpoints, hashes, generation settings and provenance.
  features.npz      Random/MIPA features, Random indices, initial class labels.
  tsne.npz          real/random/mipa 2D coordinates, real_labels, random_indices.
  projection.json   Fixed projection settings, versions and coordinate checksum.
  comparison.*      Side-by-side Random versus MIPA (PNG and PDF).
  random.* / mipa.* Individual panels (PNG and PDF).

Each comparison fits ONE joint PCA(50) + t-SNE on all 50,000 real images
and the MIPA prototypes. Random points reference their existing real rows;
they are not duplicated during fitting. Both panels have identical axes
and class-colored backgrounds. Red points identify Random images or MIPA
prototypes, not class labels. The same seed (0), perplexity (30), and
iteration count (1000) are used for all six comparisons. No labels enter
feature extraction or dimensionality reduction. Each size has its own fit.

Replot an existing comparison on CPU, with no dataset/teacher/GPU required:
  python Evaluate_benchamrk/mipa_tsne/compare.py plot --dataset CIFAR10 --percentage 0.2
Use a CPU allocation on this cluster, or a local CPU machine. For plotting,
only numpy/matplotlib and inputs.json/tsne.npz are needed. Change the plot()
function to restyle figures without recomputing features or t-SNE.

Submit/retry extraction and all six projections on Vulcan:
  bash Evaluate_benchamrk/mipa_tsne/submit.sh
Completed projections are verified and reused. Active duplicate submissions
are skipped. Logs: Evaluate_benchamrk/mipa_tsne/logs/.

These are qualitative teacher-space visualizations, not accuracy estimates
or proof of coverage/non-collapse in the original feature space. Differences
in 2D density or separation should not be interpreted as quantitative gains.
