from pathlib import Path
import json
import shutil
import hashlib
import torch
from PIL import Image

torch.set_num_threads(1)
root = Path(__file__).resolve().parent
figure_root = root.parent / 'intro_figs'
images_dir = figure_root / 'images'
images_dir.mkdir(exist_ok=True)
chosen = [44, 133, 114, 141, 102, 142, 156, 157, 17, 75]
for slot, index in enumerate(chosen, 1):
    shutil.copyfile(root / 'fig' / f'image_{index:03d}.png', images_dir / f'prototype-{slot:02d}.png')
archive = torch.load('/scratch/mmoslem3/data/tinyimagenet.pt', map_location='cpu', weights_only=True, mmap=True)
index = 9759
assert archive['classes'][int(archive['labels_val'][index])] == 'n01855672'
center = archive['images_val'][index].permute(1, 2, 0).numpy()
Image.fromarray(center).save(images_dir / 'real.png')
source = root / 'res_IIC_TinyImageNet_ResNet18BN_0.2percent_final.pt'
manifest = dict(synthetic_checkpoint=str(source), checkpoint_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    prototypes=[dict(slot=i, index=j, bird_like=i in (3, 5)) for i,j in enumerate(chosen,1)],
    center=dict(archive='/scratch/mmoslem3/data/tinyimagenet.pt', split='val', index=index, wnid='n01855672',
        matched_to='User supplied goose reference screenshot'),
    probabilities='Illustrative values preserved from supplied intro.tex; not measured model assignments.')
(figure_root / 'image_sources.json').write_text(json.dumps(manifest, indent=2)+'\n')
print(json.dumps(manifest))
