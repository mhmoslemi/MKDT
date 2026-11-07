"""Recompute pictured assignments from saved similarities and tighten the plot."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
FIGURE = HERE.parent
TAU = 0.05


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def softmax(logits):
    assert logits and all(math.isfinite(x) for x in logits)
    weights = [math.exp(x - max(logits)) for x in logits]
    return [x / sum(weights) for x in weights]


assert os.environ.get('SLURM_JOB_ID')
assert softmax([0., 0.]) == [0.5, 0.5]
assert math.isclose(softmax([0., math.log(3)])[1], 0.75)
previous = FIGURE/'mapping_tau02/assignments.json'
result = json.loads(previous.read_text())
assert result['complete'] and result['tau'] == 0.2
old_tau = result['tau']
indices = result['prototype_indices']
for view in result['results']:
    for name, digest in view['image_sha256'].items():
        assert sha(FIGURE/'images'/f"view{view['view']}"/name) == digest
    old_probability = view['probability_displayed10']
    probability = softmax([x / TAU for x in view['cosine_similarity']])
    # Temperature changes can reuse softmax outputs: p_new is proportional to p_old**(old_tau/new_tau).
    probability_all = softmax([math.log(p) * old_tau / TAU for p in view['probability_all200']])
    mass = sum(probability_all[i] for i in indices)
    assert all(math.isclose(p, probability_all[i] / mass, abs_tol=1e-6)
               for p, i in zip(probability, indices))
    assert math.isclose(sum(probability), 1.) and math.isclose(sum(probability_all), 1.)
    assert max(probability) > max(old_probability)
    view.update(probability_displayed10=probability, probability_all200=probability_all,
                displayed10_mass_in_all200=mass)
result.update(tau=TAU, inference_slurm_job_id=result['slurm_job_id'],
    slurm_job_id=os.environ['SLURM_JOB_ID'], worker_sha256=sha(Path(__file__)),
    source_result=str(previous), source_result_sha256=sha(previous),
    computation='Reuse saved cosine similarities; encoder and image views unchanged.',
    full200_note='Temperature-rescaled from saved full200 probabilities; literal tau0.05, no prototype-count rescaling.')

text = (HERE/'input_figure.tex').read_text()
assert (FIGURE/'intro.tex').read_text() == text, 'Figure changed since preparation'
text = text.replace('35.5', '24').replace('29.5', '41')
text = text.replace(r'\MIPAAngle-7', r'\MIPAAngle-12').replace(r'\MIPAAngle+7', r'\MIPAAngle+12')
text = text.replace('tau0.2', 'tau0.05').replace(r'\tau=0.2', r'\tau=0.05')
views = result['results']
for i in range(10):
    p1, p2 = [100 * v['probability_displayed10'][i] for v in views]
    text, count = re.subn(rf'prototype-{i+1:02d}/[\d.]+/[\d.]+',
                         f'prototype-{i+1:02d}/{p1:.1f}/{p2:.1f}', text)
    assert count == 1
scale = 5 * math.ceil(max(100 * p for v in views for p in v['probability_displayed10']) / 5)
text = re.sub(r'\\def\\MIPAMaxPercent\{[\d.]+\}', lambda _: f'\\def\\MIPAMaxPercent{{{scale}}}', text)
text = re.sub(r'The shared radial scale is 0--\d+', f'The shared radial scale is 0--{scale}', text)
assert text.count('{0}{0}{28}') == 2 and r'\tau=0.05' in text
assert math.sqrt(2)*15.1 < 24 and 24+41 < 90-math.sqrt(2)*16
assert 2*12 < 36
source = json.loads((HERE/'input_sources.json').read_text())
source['probabilities'].update(tau=TAU, result_file=str(HERE/'assignments.json'),
    view1_percent=[100*p for p in views[0]['probability_displayed10']],
    view2_percent=[100*p for p in views[1]['probability_displayed10']])
(HERE/'assignments.json').write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
with (HERE/'assignments.csv').open('w', newline='') as out:
    writer = csv.writer(out)
    writer.writerow(['slot','checkpoint_index','view1_cosine','view2_cosine','view1_percent','view2_percent'])
    for i, index in enumerate(indices):
        writer.writerow([i+1,index,*[v['cosine_similarity'][i] for v in views],
                         *[100*v['probability_displayed10'][i] for v in views]])
(FIGURE/'intro.tex').write_text(text)
(FIGURE/'image_sources.json').write_text(json.dumps(source, indent=2, sort_keys=True)+'\n')
print('MEASURED_PERCENTAGES', json.dumps(source['probabilities']))
print('PASS: normalized probabilities; cached image hashes; temperature identity; equal image sizes; clear bars.')
