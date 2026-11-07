#!/bin/bash
#SBATCH --job-name=mipa-figure-style
#SBATCH --account=aip-boyuwang
#SBATCH --cpus-per-task=1
#SBATCH --mem=512M
#SBATCH --time=00:05:00
#SBATCH --output=/scratch/mmoslem3/MKDT/CVPR_tables/ablations/figures/original_style_20261103/render-%j.log
set -euo pipefail
cd /scratch/mmoslem3/MKDT
module load StdEnv/2023 python/3.11.5 scipy-stack/2025a
export XDG_CACHE_HOME="$SLURM_TMPDIR/cache"
export MPLCONFIGDIR="$SLURM_TMPDIR/matplotlib"
export PYTHONDONTWRITEBYTECODE=1
python - <<'PY'
import contextlib
import copy
import io
import json
from pathlib import Path
import runpy
import shutil

plot = runpy.run_path('Evaluate_benchamrk/mipa_batch_ablation/plot.py')
plt = plot['plt']
figures = Path('CVPR_tables/ablations/figures')
backup = figures / 'original_style_20261103'
staging = backup / 'restyled'
staging.mkdir(exist_ok=True)
cases = [
    ('mipa_temperature_cifar10_p0p2_lbl1', 'temperature', r'Assignment temperature $\tau$'),
    ('mipa_batch_size_cifar10_p1_lbl1', 'batch', 'Distillation batch size'),
]
close = plt.close
plt.close = lambda *args: None
for stem, variable, xlabel in cases:
    original = json.loads((backup / (stem + '.json')).read_text())
    snapshot = copy.deepcopy(original)
    ticks = [row[variable] for row in snapshot['completed_mipa']]
    with contextlib.redirect_stdout(io.StringIO()):
        plot['save_plot'](snapshot, staging / stem, variable, xlabel, ticks)
    assert (staging / (stem + '.csv')).read_bytes() == (backup / (stem + '.csv')).read_bytes()
    assert {k: v for k, v in snapshot.items() if k != 'colors'} == {k: v for k, v in original.items() if k != 'colors'}
    fig = plt.gcf()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ax = fig.axes[0]
    texts = [ax.xaxis.label, ax.yaxis.label, *ax.get_xticklabels(), *ax.get_yticklabels(), *ax.get_legend().get_texts()]
    for text in texts:
        if not text.get_visible() or not text.get_text():
            continue
        # Locator ticks beyond the displayed range are not rendered.
        if text in ax.get_yticklabels() and not ax.get_ylim()[0] <= text.get_position()[1] <= ax.get_ylim()[1]:
            continue
        box = text.get_window_extent(renderer)
        assert box.x0 >= 0 and box.y0 >= 0 and box.x1 <= fig.bbox.width and box.y1 <= fig.bbox.height, (stem, text.get_text(), box)
    legend_box = ax.get_legend().get_window_extent(renderer)
    assert ax.bbox.contains(legend_box.x0, legend_box.y0)
    assert ax.bbox.contains(legend_box.x1, legend_box.y1)
    assert ax.get_legend().get_frame_on()
    assert tuple(fig.get_size_inches()) == (3.5, 2.8)
    labels = [text.get_window_extent(renderer) for text in ax.get_xticklabels()]
    assert all(left.x1 < right.x0 for left, right in zip(labels, labels[1:]))
    close(fig)
    print(stem + ': unchanged data and ranges; text bounds, tick spacing, and boxed interior legend and identical canvas size passed')
for stem, _, _ in cases:
    for suffix in ('.pdf', '.png', '.json'):
        shutil.copyfile(staging / (stem + suffix), figures / (stem + suffix))
PY
for stem in mipa_temperature_cifar10_p0p2_lbl1 mipa_batch_size_cifar10_p1_lbl1; do
  pdffonts "CVPR_tables/ablations/figures/$stem.pdf"
  pdftoppm -scale-to 1200 -png -singlefile "CVPR_tables/ablations/figures/$stem.pdf" "CVPR_tables/ablations/figures/original_style_20261103/restyled/$stem-pdf-preview"
done
