#!/bin/bash
#SBATCH --job-name=cvpr_downstream_figures
#SBATCH --account=aip-yiweilu
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:02:00
#SBATCH --chdir=/scratch/mmoslem3/MKDT
#SBATCH --output=/scratch/mmoslem3/MKDT/PLOTS/downstream/render-%j.out
set -euo pipefail
module --force purge
module load StdEnv/2023 python/3.11.5
source /home/mmoslem3/ENV/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
export MPLCONFIGDIR="${SLURM_TMPDIR:?}/mplconfig"
python PLOTS/downstream/plot_downstream.py
python - <<'PY'
import hashlib
import json
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

folder = Path('PLOTS/downstream').resolve()
script = folder/'plot_downstream.py'
blocks = script.read_text().split('# %%')
cells = [dict(cell_type='markdown', id='intro', metadata={}, source=[
    'Individual downstream PDFs for the CVPR figure. Run all cells from this directory or the repository root.\n',
    'Edit the first code cell to change typography or canvas size. The code cells mirror plot_downstream.py.\n',
    'The verified data file is figures/downstream_results.csv; only PDF images are exported.\n'])]
for i, block in enumerate(blocks):
    if i:
        heading, block = block.split('\n', 1)
        cells.append(dict(cell_type='markdown', id=f'heading-{i}', metadata={}, source=[heading.strip()]))
    cells.append(dict(cell_type='code', id=f'code-{i}', metadata={}, execution_count=None,
                      outputs=[], source=block.splitlines(keepends=True)))
notebook = dict(cells=cells, metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'),
                language_info=dict(name='python', version='3.11')), nbformat=4, nbformat_minor=5)
path = folder/'plot_downstream.ipynb'
path.write_text(json.dumps(notebook, indent=2)+'\n')
files = sorted((folder/'figures/individual').glob('*.pdf'))
assert len(files) == 12
before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
namespace = {'__name__': '__main__'}
os.chdir(folder)
for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        exec(compile(''.join(cell['source']), str(path), 'exec'), namespace)
assert before == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}, 'Notebook/script PDF mismatch'
for file in files:
    info = subprocess.check_output(['pdfinfo', str(file)], text=True)
    assert '118.8 x 97.2 pts' in info, (file.name, info)
    xml = subprocess.check_output(['pdftohtml', '-xml', '-zoom', '1', '-hidden', '-stdout', str(file)], text=True)
    root = ET.fromstring(xml)
    sizes = {int(font.attrib['size']) for font in root.iter('fontspec')}
    assert {8, 9} <= sizes, (file.name, sizes)
    texts = [''.join(t.itertext()) for t in root.iter('text')]
    assert not any('pretraining' in t.lower() or 'Accuracy' in t or 'Random' in t for t in texts)
print('PASS: notebook code executed; 12 byte-identical PDFs; equal native sizes; 8/9 pt fonts; no legends or y labels.')
PY
mkdir -p PLOTS/downstream/cvpr_test
export TEXINPUTS="$PWD/PLOTS/downstream:${TEXINPUTS:-}"
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=PLOTS/downstream/cvpr_test PLOTS/downstream/cvpr_preview.tex > PLOTS/downstream/cvpr_test/build.stdout 2>&1
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=PLOTS/downstream/cvpr_test PLOTS/downstream/cvpr_preview.tex >> PLOTS/downstream/cvpr_test/build.stdout 2>&1
if rg -n 'Overfull|Float too large|LaTeX Error' PLOTS/downstream/cvpr_test/cvpr_preview.log; then
    exit 1
fi
pdfseparate -f 2 -l 2 PLOTS/downstream/cvpr_test/cvpr_preview.pdf PLOTS/downstream/cvpr_test/cvpr_layout-%d.pdf
mv PLOTS/downstream/cvpr_test/cvpr_layout-2.pdf PLOTS/downstream/cvpr_test/cvpr_layout.pdf
python - <<'PY'
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
path = Path('PLOTS/downstream/cvpr_test/cvpr_layout.pdf')
xml = subprocess.check_output(['pdftohtml', '-xml', '-zoom', '1', '-hidden', '-stdout', str(path)], text=True)
root = ET.fromstring(xml)
sizes = {int(f.attrib['size']) for f in root.iter('fontspec') if 'DejaVuSans' in f.attrib.get('family', '')}
assert {8, 9} <= sizes, sizes
assert len([t for t in root.iter('text') if ''.join(t.itertext()) in ('Aircraft', 'Dogs')]) == 6
print('PASS: compiled CVPR layout preserves 8/9 pt panel fonts and all twelve downstream panels.')
PY
pdfinfo PLOTS/downstream/cvpr_test/cvpr_preview.pdf
pdffonts PLOTS/downstream/cvpr_test/cvpr_preview.pdf
printf 'INDIVIDUAL_PDFS_NOTEBOOK_AND_CVPR_TEST_COMPLETE\n'
