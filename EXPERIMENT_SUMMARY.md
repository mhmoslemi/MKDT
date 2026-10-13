# Experiment and table summary

This note summarizes the benchmark/table work and the submitted Slurm runs. It is intentionally high level; the TSV manifests listed below are the authoritative job records.

## Tables

The final editable tables are in `CVPR_tables/`:

- `cifar10_main.tex` and `cifar10_downstream.tex`
- `cifar100_main.tex` and `cifar100_downstream.tex`
- `tinyimagenet_main.tex` and `tinyimagenet_downstream.tex`

The CIFAR10 result table was renamed to `cifar10_main.tex`, matching tables were added for CIFAR100 and Tiny ImageNet, and the downstream tables were kept separate. KSS-ST was removed from every table. Old duplicate table versions were archived/removed. Cells are filled only from complete, verified runs; intentionally incompatible evaluations remain empty.

## Shared benchmark protocol

- Synthetic-data sizes: **1%, 2%, and 5%**.
- SSL methods: **SimCLR** and **Barlow Twins**.
- SimCLR epochs for 1/2/5%: **5000 / 4000 / 3000**.
- Barlow Twins epochs for 1/2/5%: **10000 / 8000 / 6000**.
- Linear-probe label budgets: **1% and 5%**.
- Each reported configuration uses **10 seeds/runs**.
- Main-table backbones are ConvNet, VGG11, and ResNet18 where applicable.
- Downstream transfer is evaluated with **ResNet18 only**.

## Run groups

### Our synthetic data

The `benchmarks/synthetic_new/` batch evaluates the available synthetic images for the main and downstream tables. The main-table runs cover ConvNet, VGG11, and ResNet18; downstream runs use ResNet18. The completed batch and its protocol are under:

`benchmarks/synthetic_new/results/batch_20261010T004533Z_wSzguy/`

### Full-data reference

The full-data refresh covers CIFAR10, CIFAR100, and Tiny ImageNet with ConvNet, VGG11, and ResNet18. SimCLR uses 300 epochs and Barlow Twins uses 600 epochs. Long ResNet18 jobs were later split by seed where cleanly possible. Remaining Tiny ImageNet jobs were canceled when the active scope was narrowed to CIFAR10/CIFAR100.

- Original jobs: `1576243--1576280`
- Batch: `benchmarks/full_data_refresh/results/batch_20261010T160040Z_73NiHv/`
- Original manifest: `submitted.tsv`
- Split/resubmitted jobs: `split_resubmitted.tsv`

### MKDT baseline

MKDT is a separate baseline from our method. Its pipeline is:

`target representations -> expert trajectories -> high-loss sorting -> 1/2/5% distillation -> SSL pretraining -> evaluation -> table report`

The benchmark uses only the generated MKDT image tensors for SSL evaluation; it does not use MKDT synthetic labels or synthetic learning rates. ConvNet and ResNet18BN generator runs were submitted first. VGG11 generation and same-backbone VGG11 evaluation were then added for CIFAR10/CIFAR100. No new Tiny ImageNet VGG11 chain was submitted.

- Main MKDT batch: `benchmarks/mkdt_multidataset/results/batch_20261010T160046Z_Y6dqZS/`
- Main manifest: `submitted.tsv`
- Seed-split replacements: `split_resubmitted.tsv`
- VGG11 extension manifest: `vgg11_submitted.tsv`
- C10/C100 ResNet18BN distillation: `1576322--1576324`, `1576340--1576342`
- C10/C100 VGG11 extension: `1599496--1599537`
- Combined VGG-aware report: `1599538`

The large ResNet18BN and VGG11 distillation jobs remain unsplit one-day jobs because the distillation code does not have a clean, verified checkpoint/resume boundary. Shorter pretraining jobs use measured wall times. The VGG11 extension submitted 42 GPU jobs: 38 on `aip-boyuwang` and 4 on `aip-yiweilu`.

## Outputs and monitoring

- Table sources: `CVPR_tables/`
- Full-data logs/results: `benchmarks/full_data_refresh/results/batch_20261010T160040Z_73NiHv/`
- MKDT logs: `benchmarks/mkdt_multidataset/results/batch_20261010T160046Z_Y6dqZS/logs/`
- MKDT synthetic tensors: `.../synthetic/generated/<generator>/<dataset>/<dataset>_<size>pct/images_5000.pt`
- MKDT evaluations: `.../runs/<source>_<generator>_<evaluator>_size<size>_<ssl>/`
- Generated MKDT table variants: `.../tables/<generator>/`

Jobs are dependency-linked, so evaluation starts only after its synthetic data and encoder are complete. Reports run after the relevant evaluation chains finish. Wang is the primary account; selected jobs are assigned to Lu according to the requested account split.
