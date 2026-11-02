import hashlib
import json
import os
from pathlib import Path
import re
from PIL import Image

assert os.environ.get('SLURM_JOB_ID'), 'Run the image export in a CPU allocation.'
root = Path('/scratch/mmoslem3/MKDT')
source = root / 'CVPR_tables/ablations/figures/mipa_components_cifar100_pt1'
destination = root / 'OT-SSL-DD/cifar100_1percent_pairs'
selection = json.loads((source / 'paper_examples.json').read_text())
assert selection['indices_zero_based'] == [64, 192, 279, 22, 332, 376, 453, 483]
indices = [64, 192, 196, 22, 332, 376, 453, 483, 228, 428]
names = ['bowl', 'telephone', 'turtle', 'sweet_pepper', 'worm', 'rose', 'orchid', 'lizard', 'lamp', 'bridge']
inputs = json.loads((source / 'inputs.json').read_text())
assert len(set(indices)) == len(set(names)) == 10
assert names == [inputs['classes'][inputs['initial_labels'][index]].lower().replace(' ', '_') for index in indices]
exported = []
for version, filename, expected_hash in zip(('original', 'distilled'), ('initial_all.png', 'final_all.png'), selection['source_grid_sha256']):
    path = source / filename
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_hash
    with Image.open(path) as grid:
        assert grid.mode == 'RGB' and grid.size == (342, 1702)
        for position, (index, name) in enumerate(zip(indices, names), 1):
            left, top = 2 + 34 * (index % 10), 2 + 34 * (index // 10)
            tile = grid.crop((left, top, left + 32, top + 32))
            output = destination / f'{version}_{position:02d}_{name}.png'
            if output.exists():
                with Image.open(output) as previous:
                    assert previous.size == tile.size and previous.mode == tile.mode and previous.tobytes() == tile.tobytes(), output
            else:
                tile.save(output)
            with Image.open(output) as saved:
                assert saved.size == (32, 32) and saved.mode == 'RGB' and saved.tobytes() == tile.tobytes()
            exported.append(output)
references = re.findall(r'\\includegraphics\[[^\]]+\]\{([^}]+)\}', (destination / 'figure.tex').read_text())
references = [name for name in references if not Path(name).name.startswith('noise_')]
assert len(references) == len(exported) == 20
assert [root / 'OT-SSL-DD' / name for name in references] == exported
for version in ('original', 'distilled'):
    replaced = destination / f'{version}_03_television.png'
    if replaced.exists():
        archived = destination / 'archive' / replaced.name
        archived.parent.mkdir(exist_ok=True)
        assert not archived.exists(), archived
        replaced.rename(archived)
contact = Image.new('RGB', (342, 70), 'white')
for position, path in enumerate(exported):
    row, column = divmod(position, 10)
    with Image.open(path) as tile:
        contact.paste(tile, (2 + 34 * column, 2 + 34 * row))
contact.resize((2052, 420), Image.Resampling.NEAREST).save(Path(__file__).with_name('pairs_preview.png'))
print(f'Exported and verified 20 exact RGB crops: {destination}', flush=True)
print('Selected prototype indices: ' + str(indices), flush=True)
print('LaTeX references verified: originals first row, distilled second row.', flush=True)
