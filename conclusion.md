| `lr_img` | Best accuracy | Iteration |
|---|---:|---:|
| **0.005 (baseline)** | **41.38%** | 500 |
| 0.01 | 41.17% | 250 |
| 0.02 | 40.43% | 250 |
| 0.05 | 39.01% | 250 |

| `batch_real` | Best accuracy | Iteration |
|---|---:|---:|
| **512** | **41.80%** | 750 |
| 128 | 41.51% | 750 |
| 2048 | 41.51% | 500 |
| 1024 (baseline) | 41.38% | 500 |
| 256 | 41.24% | 750 |

| `num_random_networks` | Best accuracy | Iteration |
|---|---:|---:|
| **5** | **41.91%** | 750 |
| 1 | 41.69% | 1000 |
| 10 | 41.45% | 500 |
| 40 | 41.40% | 750 |
| 20 (baseline) | 41.38% | 500 |

| `distill_aug_strategy` | Best accuracy | Iteration |
|---|---:|---:|
| **`color_crop_flip_scale_rotate`** | **41.79%** | 750 |
| `crop_flip` | 41.58% | 500 |
| `color_crop_flip` | 41.51% | 750 |
| `color_crop_cutout_flip_scale_rotate` (baseline) | 41.38% | 500 |
| `none` | 41.36% | 500 |
| `color_crop_cutout_flip` | 41.24% | 1000 |

| `distill_aug_mode` | Best accuracy | Iteration |
|---|---:|---:|
| **M** | **41.59%** | 750 |
| S (baseline) | 41.38% | 500 |
