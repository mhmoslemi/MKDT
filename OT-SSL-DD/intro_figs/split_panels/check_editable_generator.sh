#!/bin/bash
#SBATCH --job-name=intro_editable_check
#SBATCH --account=aip-yiweilu
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:01:00
#SBATCH --chdir=/scratch/mmoslem3/MKDT/OT-SSL-DD/intro_figs
#SBATCH --output=/scratch/mmoslem3/MKDT/OT-SSL-DD/intro_figs/split_panels/%x-%j.out
set -euo pipefail
bash split_panels/job.sh
sha256sum -c split_panels/editable_generator_reference.sha256
module --force purge
module load StdEnv/2023 python/3.11.5
source /home/mmoslem3/ENV/bin/activate
export PYTHONDONTWRITEBYTECODE=1
python - <<'PY'
import hashlib
import os
from pathlib import Path
import tempfile
import generate_figures as g

# Exercise real rendering after changing values, images, geometry, formatting,
# colors and a panel-specific font. Temporary outputs leave the final plots alone.
with tempfile.TemporaryDirectory(dir=os.environ['SLURM_TMPDIR']) as directory:
    g.OUTPUT_DIR = Path(directory)
    g.PREVIEW_PIXELS = 500
    g.STYLE.update(page_size=280, image_radius=118, image_size=24,
                   center_image_size=24, circle_radius=22, bar_inner=32,
                   bar_length=45, bar_half_angle=13, label_size=8.5,
                   label_gap=1, number_format='.0f', label_suffix=r'\%',
                   grid_radii=(55, 80), label_rotation_offset=-80)
    g.PANELS[0]['values'][0] = 8.8
    g.PANELS[0]['values'][5] = 28.5
    g.PANELS[0]['fill_rgb'] = (210, 230, 210)
    g.PANELS[0]['images'][0], g.PANELS[0]['images'][1] = g.PANELS[0]['images'][1], g.PANELS[0]['images'][0]
    g.PANELS[1]['style'] = {'font_family': r'\sffamily', 'label_weight': r'\mdseries'}
    g.PANELS[1]['angles'] = [92-36*i for i in range(10)]
    g.main()
    for panel in g.PANELS:
        filename = panel['name']+'_preview.png'
        assert (Path(directory)/filename).is_file()
        assert hashlib.sha256((Path(directory)/filename).read_bytes()).digest() != hashlib.sha256((g.HERE/filename).read_bytes()).digest()
print('PASS: settings changed successfully in both rendered panels; default outputs preserved.')
PY
