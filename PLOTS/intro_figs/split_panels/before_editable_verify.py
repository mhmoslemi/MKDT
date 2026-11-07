"""Verify label geometry, square PDF sizes, measured values and explicit display edits."""
import json
import math
from pathlib import Path
import re
import subprocess

root = Path(__file__).resolve().parent.parent
result = json.loads((root/'mapping_tau0025_flip/assignments.json').read_text())
source = (root/'intro_panel.tex').read_text()
values = re.findall(r'prototype-\d+/([\d.]+)/([\d.]+)', source)
assert len(values) == 10 and result['tau'] == 0.025
assert 'width=28bp,height=28bp' in source
dimensions = []
left_display_edits = {2: 14.2, 5: 29.5, 7: 9.4, 9: 6.6}
for view, name in enumerate(('left', 'right'), 1):
    measured = [round(100*p, 1) for p in result['results'][view-1]['probability_displayed10']]
    for i, pair in enumerate(values):
        expected = left_display_edits.get(i, measured[i]) if view == 1 else measured[i]
        assert float(pair[view-1]) == expected
    assert math.isclose(sum(float(pair[view-1]) for pair in values), sum(measured), abs_tol=1e-9)
    info = subprocess.check_output(['pdfinfo', str(root/f'intro_{name}.pdf')], text=True)
    size = re.search(r'Page size:\s+([\d.]+) x ([\d.]+) pts', info)
    assert size and re.search(r'Pages:\s+1\b', info)
    dimensions.append(tuple(map(float, size.groups())))
    assert all(abs(v-252) < 0.02 for v in dimensions[-1]), dimensions[-1]
    log = (root/f'intro_{name}.log').read_text()
    assert 'Overfull' not in log and 'LaTeX Error' not in log
    labels = re.findall(r'MIPA_LABEL:(\d+):(\d+):([\d.]+)pt:([\d.]+)pt:([\d.]+)pt:([\d.]+)', log)
    assert len(labels) == 10
    anchors = re.findall(r'MIPA_BASE:(\d+):(\d+):(-?[\d.]+)pt:(-?[\d.]+)pt:(-?[\d.]+)pt:(-?[\d.]+)pt', log)
    assert len(anchors) == 10
    for logged_view, index, bx, by, tx, ty in anchors:
        assert int(logged_view) == view
        angle = math.radians(90-36*int(index))
        radius = 34+52*float(values[int(index)][view-1])/35+.6
        bx, by, tx, ty = (float(v)*72/72.27 for v in (bx, by, tx, ty))
        assert math.hypot(bx-radius*math.cos(angle), by-radius*math.sin(angle)) < .01, 'Bottom of number must meet bar tip'
        assert (tx-bx)*math.cos(angle)+(ty-by)*math.sin(angle) > 6, 'Number must face outward from its baseline'
    boxes = []
    for logged_view, index, width, height, depth, radius in labels:
        assert int(logged_view) == view
        index = int(index)
        angle = math.radians(90-36*index)
        radius = float(radius)
        probability = float(values[index][view-1])
        assert math.isclose(radius, 34+52*probability/35, abs_tol=0.002)
        half_width = float(width)*72/72.27/2
        half_height = (float(height)+float(depth))*72/72.27/2
        assert half_width > 5 and half_height > 3
        assert 2*half_width < 2*radius*math.sin(math.radians(12))-.8, (name, index, 'Number wider than bar cap')
        label_angle = angle-math.pi/2
        cx, cy = ((radius+.6+half_height)*f(angle) for f in (math.cos, math.sin))
        corners = [(cx+u*math.cos(label_angle)-v*math.sin(label_angle),
                    cy+u*math.sin(label_angle)+v*math.cos(label_angle))
                   for u in (-half_width, half_width) for v in (-half_height, half_height)]
        box = (min(x for x,y in corners), min(y for x,y in corners),
               max(x for x,y in corners), max(y for x,y in corners))
        assert all(abs(c) < 126 for c in box), 'Label clipped by page'
        assert math.hypot(max(box[0], -box[2], 0), max(box[1], -box[3], 0)) > 24.3
        for j in range(10):
            image_angle = math.radians(90-36*j)
            ix, iy = 110*math.cos(image_angle), 110*math.sin(image_angle)
            assert box[2] < ix-15.1 or ix+15.1 < box[0] or box[3] < iy-15.1 or iy+15.1 < box[1], 'Number overlaps thumbnail'
        boxes.append(box)
    for i, a in enumerate(boxes):
        for b in boxes[i+1:]:
            assert a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1], 'Percentage labels overlap'
    text = subprocess.check_output(['pdftotext', '-raw', str(root/f'intro_{name}.pdf'), '-'], text=True)
    assert re.sub(r'\s+', '', text) == ''.join(pair[view-1] for pair in values), text
    assert '%' not in text
    assert all(word not in text.lower() for word in ('rotation', 'flip', 'crop', 'figure'))
assert dimensions[0] == dimensions[1]
assert 86.2 < 110-math.sqrt(2)*15.1
assert 110+15.1 < 126
assert math.sqrt(2)*15.1 < 23.8
images = [(110*math.cos(math.radians(90-36*i)), 110*math.sin(math.radians(90-36*i))) for i in range(10)]
for i, (x, y) in enumerate(images):
    for xx, yy in images[i+1:]:
        assert abs(x-xx) > 30.2 or abs(y-yy) > 30.2
print('PASS: identical square PDFs; all 20 numbers have their bottom against the bar tip and face outward, narrower than their caps; no percent signs, clipping, or label/image collisions; only the four requested left-panel values adjusted, preserving their total.')
