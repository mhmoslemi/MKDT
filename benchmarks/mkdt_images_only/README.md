# MKDT images-only all-backbone benchmark

MKDT synthesis uses ConvNet. The resulting image tensors are then used to fill
**every MKDT cell** in `CVPR_tables/cifar10_main.tex` and
`CVPR_tables/cifar10_downstream.tex`.

Protocol:

- Source: MKDT CIFAR-10 `images_5000.pt` at 1%, 2%, and 5%.
- Evaluation encoder: a fresh ConvNet, VGG11, or ResNet18 for every one of 10 seeds.
- Pretraining: SimCLR or Barlow Twins, using only the synthetic image tensor.
- Explicitly excluded: MKDT `labels_5000.pt` and `synlr_5000.pt`.
- Main evaluation: frozen linear probes on 1% and 5% labeled CIFAR-10.
- Downstream evaluation: frozen linear probes on 1% and 5% labeled CIFAR-100,
  Aircraft, CUB-2011, Dogs, Flowers, and Tiny ImageNet.
- Epoch budgets and label selection match the existing table protocol.

MKDT's published repository provides 2% and 5% distillation settings. The 1%
row is therefore a new extension and must not be described as a published MKDT
configuration.

Expected image paths:

```
logged_files/CIFAR10/CIFAR10_1pct/images_5000.pt
logged_files/CIFAR10/CIFAR10_2pct/images_5000.pt
logged_files/CIFAR10/CIFAR10_5pct/images_5000.pt
```

Paths can be overridden with `MKDT_IMAGE_1`, `MKDT_IMAGE_2`, and
`MKDT_IMAGE_5`. Once all three tensors exist, submit with:

```
bash benchmarks/mkdt_images_only/submit.sh
```

The report job always writes table copies under `<batch>/tables/`. It updates
the two source files in `CVPR_tables/` only when all 252 result cells are
complete; partial results are never applied.
