"""Measure the pictured views with the frozen distillation teacher; then render."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
FIGURE = HERE.parent
PROJECT = FIGURE.parent
sys.path.insert(0, str(PROJECT))
from dd_teachers import load_teacher
from eval import reference_modules
from utils import DiffAugment, ParamDiffAug

TAU = 0.025
AUGMENTATION_SEED = 0

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_ACCOUNT') == 'aip-yiweilu'
assert torch.cuda.is_available()
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
source = json.loads((HERE/'input_sources.json').read_text())
checkpoint_path = Path(source['synthetic_checkpoint'])
assert sha(checkpoint_path) == source['checkpoint_sha256'], 'Synthetic checkpoint changed'
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
assert checkpoint['dataset'] == 'TinyImageNet' and checkpoint['teacher_source'] == 'pretrained'
assert checkpoint['teacher_seed'] == 1 and checkpoint['complete']
assert checkpoint['data'].shape == (200, 3, 64, 64)
slots = source['prototypes']
assert [s['slot'] for s in slots] == list(range(1, 11))
indices = [s['index'] for s in slots]
assert len(set(indices)) == 10
mean, std = checkpoint['normalization']
mean_t, std_t = (torch.tensor(value).view(1, 3, 1, 1) for value in (mean, std))
normalized = checkpoint['data'].float()
assert torch.isfinite(normalized).all()
pixels = ((normalized * std_t + mean_t).clamp(0, 1)*255+.5).byte()
synthetic = [Image.fromarray(image.permute(1, 2, 0).numpy()) for image in pixels]
for slot, index in enumerate(indices, 1):
    with Image.open(FIGURE/'images'/f'prototype-{slot:02d}.png') as im:
        assert np.array_equal(np.asarray(im.convert('RGB')), np.asarray(synthetic[index]))
archive = torch.load(source['center']['archive'], map_location='cpu', weights_only=True, mmap=True)
real_index = source['center']['index']
assert archive['classes'][int(archive['labels_val'][real_index])] == source['center']['wnid']
real = Image.fromarray(archive['images_val'][real_index].permute(1, 2, 0).numpy())
with Image.open(FIGURE/'images'/'real.png') as im:
    assert np.array_equal(np.asarray(real), np.asarray(im.convert('RGB')))
del archive
teacher_root = PROJECT.parent/'Evaluate_benchamrk/full_ssl_checkpoints_1000ep_3seeds'
teacher_path = teacher_root/'TinyImageNet/ResNet18/full/pt100/simclr/seed_001/encoder.pt'
assert sha(teacher_path) == '08b60ce0d69c44b788445eaf382c10e6a4531b81a69100d7d795697c59833d66'
args = argparse.Namespace(teacher_source='pretrained', model='ResNet18BN',
    teacher_path=str(teacher_root), dataset='TinyImageNet', teacher_ssl='simclr',
    teacher_seed=1, device='cuda')
teacher = load_teacher(args, mean, std, (64, 64))
common = reference_modules()['common']
state_before = common.state_hash(teacher)
assert not teacher.training and all(not p.requires_grad for p in teacher.parameters())
# One known case checks normalization and the softmax temperature convention.
eye = torch.eye(2)
expected = torch.tensor([1., math.exp(-1/TAU)])
expected /= expected.sum()
assert torch.allclose(F.softmax((eye[:1] @ eye.T)/TAU, dim=1)[0], expected)
assert torch.equal(torch.rot90(torch.arange(9).reshape(3, 3), -1),
    torch.tensor([[6, 3, 0], [7, 4, 1], [8, 5, 2]]))

results = []
tensor = torch.from_numpy(np.stack([np.asarray(im) for im in [real, *synthetic]]).copy()).permute(0, 3, 1, 2).float()/255
normalized_input = (tensor - mean_t)/std_t
param = ParamDiffAug()
param.aug_mode = 'M'
augmentation = dict(strategy='scale_cutout', seed=AUGMENTATION_SEED,
    ratio_scale=param.ratio_scale, ratio_cutout=param.ratio_cutout,
    fill='zero in source-normalized coordinates, i.e. dataset-mean RGB',
    code=str(PROJECT/'utils.py'), code_sha256=sha(PROJECT/'utils.py'))
for number in (1, 2):
    if number == 1:
        transformed = torch.rot90(normalized_input, -1, dims=(2, 3))
    else:
        transformed = DiffAugment(normalized_input, 'scale_cutout', seed=AUGMENTATION_SEED, param=param)
        repeated = DiffAugment(normalized_input[:1].repeat(2, 1, 1, 1),
            'scale_cutout', seed=AUGMENTATION_SEED, param=param)
        assert torch.equal(repeated[0], repeated[1])
        assert torch.allclose(repeated[:1], transformed[:1], atol=1e-6)
    # Quantize once so the encoder sees exactly the pixels exported to the figure.
    augmented_pixels = ((transformed * std_t + mean_t).clamp(0, 1)*255+.5).byte()
    augmented = [Image.fromarray(im.permute(1, 2, 0).numpy()) for im in augmented_pixels]
    encoder_input = (augmented_pixels.float()/255 - mean_t)/std_t
    with torch.inference_mode():
        features = torch.cat([teacher.embed(batch.cuda()).cpu() for batch in encoder_input.split(32)])
        assert features.shape == (201, 512) and torch.isfinite(features).all()
        features = F.normalize(features, dim=1)
        similarity = (features[0:1] @ features[1:].T)[0]
        probability_all = F.softmax(similarity / TAU, dim=0)
        probability = F.softmax(similarity[indices] / TAU, dim=0)
        assert torch.allclose(probability.sum(), torch.tensor(1.))
        assert torch.allclose(probability, probability_all[indices]/probability_all[indices].sum(), atol=1e-6)
        if number == 1:
            previous = json.loads((FIGURE/'mapping_tau02/assignments.json').read_text())
            assert torch.allclose(similarity[indices], torch.tensor(previous['results'][0]['cosine_similarity']), atol=1e-6)
    folder = FIGURE/'images'/'scale_cutout'/f'view{number}'
    folder.mkdir(parents=True, exist_ok=True)
    augmented[0].save(folder/'real.png')
    for slot, index in enumerate(indices, 1):
        augmented[index+1].save(folder/f'prototype-{slot:02d}.png')
    results.append(dict(view=number, cosine_similarity=similarity[indices].tolist(),
        cosine_similarity_all200=similarity.tolist(),
        probability_displayed10=probability.tolist(), probability_all200=probability_all.tolist(),
        displayed10_mass_in_all200=float(probability_all[indices].sum()),
        image_sha256={p.name: sha(p) for p in sorted(folder.glob('*.png'))}))
assert common.state_hash(teacher) == state_before, 'Frozen encoder changed'
result = dict(complete=True, slurm_job_id=os.environ['SLURM_JOB_ID'], tau=TAU,
    normalization_scope='softmax over the ten displayed prototypes',
    formula='softmax(L2(f(view(real))) @ L2(f(view(synthetic))).T / tau)',
    teacher=str(teacher_path), teacher_sha256=sha(teacher_path), teacher_seed=1,
    representation='frozen encoder features (512-D), no projection head',
    synthetic_checkpoint=str(checkpoint_path), synthetic_sha256=sha(checkpoint_path),
    checkpoint_iteration=checkpoint['iteration'], normalization=checkpoint['normalization'],
    prototype_indices=indices, center=source['center'],
    views_spec=['90 degrees clockwise', 'shared DiffAugment scale_cutout, multi mode, seed0'],
    augmentation=augmentation, image_directory=str(FIGURE/'images'/'scale_cutout'),
    input_pixels='same uint8 RGB pixels as figure PNGs; resulting augmented PNGs are rendered directly',
    full200_note='Diagnostic also uses literal tau0.025; no prototype-count temperature rescaling.',
    worker_sha256=sha(__file__), results=results)
common.atomic_json(HERE/'assignments.json', result)
with (HERE/'assignments.csv').open('w', newline='') as out:
    writer = csv.writer(out)
    writer.writerow(['slot','checkpoint_index','view1_cosine','view2_cosine','view1_percent','view2_percent'])
    for i, index in enumerate(indices):
        writer.writerow([i+1,index,*[r['cosine_similarity'][i] for r in results],
            *[100*r['probability_displayed10'][i] for r in results]])

text = (HERE/'figure_template.tex').read_text()
for i, row in enumerate(slots):
    p1, p2 = [100*r['probability_displayed10'][i] for r in results]
    text, count = re.subn(rf'prototype-{i+1:02d}/[\d.]+/[\d.]+', f'prototype-{i+1:02d}/{p1:.1f}/{p2:.1f}', text)
    assert count == 1
max_probability = max(100*p for r in results for p in r['probability_displayed10'])
scale = max(10, 5*math.ceil(max_probability/5))
text = re.sub(r'\\def\\MIPAMaxPercent\{[\d.]+\}', lambda _: f'\\def\\MIPAMaxPercent{{{scale}}}', text)
caption = (r'\caption{\textbf{Soft assignments across augmentations.} '
    r'A frozen TinyImageNet ResNet-18 encoder assigns the real image to the ten shown synthetic prototypes '
    r'using softmax over $\ell_2$-normalized feature similarities ($\tau=0.025$). '
    r'Each panel applies the same transform to all images. Probabilities are normalized over these ten prototypes; '
    f'both panels use a 0--{scale}' + r'\% radial scale.}')
text = re.sub(r'\\caption\{.*?(?=\n\\label)', lambda _: caption, text, flags=re.S)
assert 'illustrative' not in text.lower()
assert '{0}{0}{28}' in text and '{0}{0}{48}' not in text
assert text.count('{0}{0}{28}') == 2
assert 24+39 < 86-math.sqrt(2)*16
assert 'flip' not in text.lower() and r'\tau=0.025' in text
assert (FIGURE/'intro.tex').read_text() == (HERE/'input_figure.tex').read_text(), 'Figure edited while calculation was running'
(FIGURE/'intro.tex').write_text(text)
source['probabilities'] = dict(kind='measured', tau=TAU, normalization_scope='ten displayed prototypes',
    result_file=str(HERE/'assignments.json'),
    view1_percent=[100*p for p in results[0]['probability_displayed10']],
    view2_percent=[100*p for p in results[1]['probability_displayed10']])
source['views'] = dict(image_directory=result['image_directory'], specs=result['views_spec'], augmentation=augmentation)
common.atomic_json(FIGURE/'image_sources.json', source)
print('MEASURED_PERCENTAGES', json.dumps(source['probabilities']), flush=True)
print('PASS: teacher unchanged; probabilities normalized; image provenance verified; image sizes identical; bars clear thumbnails.', flush=True)
