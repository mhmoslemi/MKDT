#!/usr/bin/env python3
"""Regenerate the two current square figures exactly, using their TikZ sources.

Run from any directory:
    python /scratch/mmoslem3/MKDT/OT-SSL-DD/intro_figs/generate_figures.py

Keep this file beside intro_left.tex, intro_right.tex, intro_panel.tex, images/,
mapping_tau0025_flip/assignments.json, and split_panels/verify.py.
Requires Python's standard library, pdflatex with the existing TeX packages,
and Poppler (pdftoppm, pdfinfo, pdftotext). No model inference is performed.
The left panel includes four documented manual adjustments for illustration.

Outputs beside this script:
    intro_left.pdf, intro_right.pdf                 (252 x 252 pt)
    intro_left_preview.png, intro_right_preview.png (1200 x 1200 px)
"""
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parent
    missing = [name for name in ('pdflatex', 'pdftoppm', 'pdfinfo', 'pdftotext')
               if shutil.which(name) is None]
    if missing:
        raise SystemExit('Missing required commands: ' + ', '.join(missing))
    required = [root/name for name in ('intro_left.tex', 'intro_right.tex',
        'intro_panel.tex', 'mapping_tau0025_flip/assignments.json', 'split_panels/verify.py')]
    for view in (1, 2):
        required.extend(root/'images'/f'view{view}'/name for name in
            ['real.png', *[f'prototype-{i:02d}.png' for i in range(1, 11)]])
    for path in required:
        if not path.is_file():
            raise SystemExit(f'Missing figure input: {path}')

    for side in ('left', 'right'):
        log = root/'split_panels'/f'{side}_compile.stdout'
        with log.open('w') as output:
            completed = subprocess.run(['pdflatex', '-interaction=nonstopmode',
                '-halt-on-error', f'intro_{side}.tex'], cwd=root,
                stdout=output, stderr=subprocess.STDOUT)
        if completed.returncode:
            raise SystemExit(f'LaTeX compilation failed; see {log}')

    subprocess.run([sys.executable, '-B', str(root/'split_panels/verify.py')],
                   cwd=root, check=True)
    for side in ('left', 'right'):
        subprocess.run(['pdftoppm', '-singlefile', '-scale-to', '1200', '-png',
            f'intro_{side}.pdf', f'intro_{side}_preview'], cwd=root, check=True)
        print(root/f'intro_{side}.pdf')
        print(root/f'intro_{side}_preview.png')


if __name__ == '__main__':
    main()
