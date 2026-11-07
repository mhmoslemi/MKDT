"""Reproduce comparison.pdf/png/svg and the three equal-size individual panels.

Run with your activated environment: python time_stoare_compare.py
Paths resolve relative to this file, so the working directory does not matter.
ACCURACY_LIMITS=None fits the actual values and error bars automatically.
Each of time.pdf, storage.pdf, and accuracy.pdf is a separate figure.
"""
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.transforms import Bbox
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BENCH = ROOT/'Evaluate_benchamrk'
MIPA = BENCH/'mipa_cvpr_color_crop/results/CIFAR10/ResNet18BN/pt0.2/simclr/ssl_seed_000/summary_CIFAR10_lbl1.json'
MKDT = BENCH/'partial_probe_results/mkdt_CIFAR10_p0p2_simclr/CIFAR10_lbl1/summary.json'
COST_TABLE = ROOT/'CVPR_tables/efficiency/generation_cost.tex'
ACCURACY_LIMITS = None
HEADROOM = 1.14


def read(path):
    return json.loads(path.read_text())


def evidence(path):
    return dict(path=str(path.relative_to(ROOT)), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


arrow_labels = []


def labeled_arrow(ax, start, end, text, fontsize=13.5, linewidth=3, offset_points=14):
    """Center text along the arrow, offset perpendicularly in physical points."""
    start_px, end_px = ax.transData.transform([start,end])
    dx, dy = end_px-start_px
    length = math.hypot(dx,dy)
    assert length > 0
    ax.annotate('',xy=end,xytext=start,
        arrowprops=dict(arrowstyle='-|>',color='#c1c3c5',linewidth=linewidth,
                        mutation_scale=12,shrinkA=0,shrinkB=0),zorder=4)
    midpoint = ((start[0]+end[0])/2,(start[1]+end[1])/2)
    label = ax.annotate(text,xy=midpoint,
        xytext=(-dy/length*offset_points,dx/length*offset_points),textcoords='offset points',
        ha='center',va='center',fontsize=fontsize,
        rotation=math.degrees(math.atan2(dy,dx)),rotation_mode='anchor',zorder=6)
    arrow_labels.append((ax,start,end,label))


block = COST_TABLE.read_text().split(r'\multirow{2}{*}{CIFAR10}', 1)[1].split(r'\midrule', 1)[0]
costs = {}
for line in block.splitlines():
    fields = [v.strip() for v in re.sub(r'\\textbf\{([^}]+)\}', r'\1', line).replace(r'\\', '').split('&')]
    if len(fields) == 7 and fields[1] in ('MKDT', 'MIPA'):
        costs[fields[1]] = dict(time_minutes=float(fields[2]), storage_gib=float(fields[6]))
assert costs == {'MKDT': {'time_minutes':47.82, 'storage_gib':10.65},
                 'MIPA': {'time_minutes':6.43, 'storage_gib':0.042}}
rows = []
for name, path in [('MKDT', MKDT), ('MIPA', MIPA)]:
    saved = read(path)
    assert saved['complete'] and saved['ssl_epochs'] == 2000
    assert (saved['source'], saved['target'], saved['subset'], saved['labels'], saved['model'], saved['ssl']) == ('CIFAR10', 'CIFAR10', 0.2, 1, 'ResNet18', 'simclr')
    scores = []
    for sample in saved['samples']:
        probe = read(Path(sample['result_file']))
        assert probe['complete'] and probe['encoder_unchanged']
        assert probe['accuracy_percent'] == sample['accuracy_percent']
        assert probe['expected']['evaluation']['epochs'] == 100
        scores.append(probe['accuracy_percent'])
    assert math.isclose(statistics.mean(scores), saved['mean_percent'])
    assert math.isclose(statistics.stdev(scores), saved['std_percent'])
    rows.append(dict(method=name, accuracy_percent=saved['mean_percent'], std_percent=saved['std_percent'],
        probes=len(scores), ssl_seeds=sorted({s['ssl_seed'] for s in saved['samples']}),
        **costs[name], summary=evidence(path)))
mkdt, mipa = rows
assert [row['probes'] for row in rows] == [6, 5]
speedup = mkdt['time_minutes']/mipa['time_minutes']
storage_ratio = mkdt['storage_gib']/mipa['storage_gib']
gain = mipa['accuracy_percent']-mkdt['accuracy_percent']
accuracy_low = math.floor(min(r['accuracy_percent']-r['std_percent'] for r in rows))-1
accuracy_high = accuracy_low+HEADROOM*(max(r['accuracy_percent']+r['std_percent'] for r in rows)-accuracy_low)
accuracy_limits = ACCURACY_LIMITS or (accuracy_low,accuracy_high)
snapshot = dict(dataset='CIFAR10', data_percent=0.2, labels_percent=1, synthetic_images=100,
    ssl_epochs=2000, probe_epochs=100, rows=rows, time_speedup=speedup,
    storage_ratio=storage_ratio, accuracy_gain_pp=gain,
    timing_definition='Total CIFAR10 0.2% distillation minutes copied directly from the requested cost table.',
    storage_definition='Auxiliary storage in GiB, as reported in the requested table; not RAM or GPU memory.',
    exclusions=['initialization', 'teacher training'], accuracy_axis_limits=list(accuracy_limits),
    source_note='Cost table uses the historical random128 MIPA configuration; accuracy uses current color_crop MIPA. Cost is not measured for the evaluated checkpoint.',
    uncertainty='Error bars are sample SD across probes: MKDT 2 SSL seeds x 3 probes; MIPA 1 SSL seed x 5 probes.',
    cost_table=evidence(COST_TABLE), plot_source=evidence(Path(__file__)))
(HERE/'comparison.json').write_text(json.dumps(snapshot, indent=2)+'\n')
with (HERE/'comparison.csv').open('w', newline='') as out:
    writer = csv.DictWriter(out, fieldnames=['method','time_minutes','storage_gib','accuracy_percent','std_percent','probes'], extrasaction='ignore')
    writer.writeheader()
    writer.writerows(rows)

plt.rcParams.update({'font.family':'STIXGeneral', 'mathtext.fontset':'stix', 'font.size':9,
    'axes.titlesize':11, 'axes.labelsize':8, 'xtick.labelsize':9, 'ytick.labelsize':7.5,
    'pdf.fonttype':42, 'ps.fonttype':42, 'svg.fonttype':'none', 'axes.linewidth':0.6,
    'text.color':'#282c31', 'axes.labelcolor':'#5b656f', 'xtick.color':'#282c31', 'ytick.color':'#5b656f'})
fig, axes = plt.subplots(1, 3, figsize=(7.05, 3.1))
fig.subplots_adjust(left=.003, right=.997, bottom=.145, top=.901, wspace=.10)
fills, edges = ['#f8deda', '#d9e6f4'], ['#c76a64', '#5b84b4']
centers, bar_width = [82, 215], 52
specs = [('Distillation time', 'time_minutes', 0, HEADROOM*max(r['time_minutes'] for r in rows)),
         ('Storage', 'storage_gib', 0, HEADROOM*max(r['storage_gib'] for r in rows)),
         ('Accuracy', 'accuracy_percent', *accuracy_limits)]
for ax, (title, field, low, high) in zip(axes, specs):
    ax.set_xlim(20,277)
    ax.set_ylim(low,high)
    ax.set_axis_off()
    ax.set_facecolor('white')
    ax.text(.5,1.06,title,transform=ax.transAxes,ha='center',va='center',fontsize=15,color='#282c31')
    ax.plot([40,253],[low,low],color='#8ca8c7',linewidth=1.3,zorder=5,clip_on=False)
    for i, (x,row) in enumerate(zip(centers,rows)):
        height = row[field]-low
        assert 0 < height < high-low
        ax.add_patch(Rectangle((x-bar_width/2,low),bar_width,height,
                              facecolor=fills[i],edgecolor=edges[i],linewidth=.55,zorder=3))
        label = (f"{row[field]:.2f} min" if field == 'time_minutes' else
                 f"{row[field]:g} GiB" if field == 'storage_gib' else f"{row[field]:.2f}%")
        label_y = row[field]+(row['std_percent'] if field == 'accuracy_percent' else 0)
        ax.annotate(label,xy=(x,label_y),xytext=(0,6),textcoords='offset points',
                    ha='center',va='bottom',fontsize=13.5,zorder=6)
        ax.text(x,-.09,row['method'],transform=ax.get_xaxis_transform(),ha='center',va='top',fontsize=15)
    if field != 'accuracy_percent':
        maximum = max(r[field] for r in rows)
        ratio = speedup if field == 'time_minutes' else storage_ratio
        labeled_arrow(ax,(122,.98*maximum),(204,mipa[field]+.18*maximum),
                      f'{ratio:.1f} times '+('faster' if field == 'time_minutes' else 'smaller'))
    else:
        ax.errorbar(centers,[r[field] for r in rows],yerr=[r['std_percent'] for r in rows],fmt='none',
                    ecolor='#5b656f',elinewidth=.6,capsize=2,capthick=.6,zorder=4)
        arrow_start = (125,mkdt[field]+.17*(high-low))
        arrow_end = (182,mipa[field]+.02*(high-low))
        labeled_arrow(ax,arrow_start,arrow_end,f'+{gain:.2f} pp gain',
                      fontsize=13,linewidth=2.6,offset_points=16)
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
for ax in axes:
    for text in ax.texts:
        bounds = text.get_window_extent(renderer)
        assert 0 <= bounds.y0 <= bounds.y1 <= fig.bbox.height, 'Text would be clipped vertically'
for ax,start,end,label in arrow_labels:
    start_px,end_px = ax.transData.transform([start,end])
    dx,dy = end_px-start_px
    center = label.get_window_extent(renderer).get_points().mean(axis=0)
    offset = center-(start_px+end_px)/2
    assert abs((offset[0]*dx+offset[1]*dy)/math.hypot(dx,dy)) < .1, 'Text is not centered along arrow'
positions = [ax.get_position() for ax in axes]
assert all(math.isclose(p.width,positions[0].width) and math.isclose(p.height,positions[0].height) for p in positions)
for suffix in ['pdf','svg','png']:
    fig.savefig(HERE/f'comparison.{suffix}', dpi=300, facecolor='white')
panel_width = positions[0].width*fig.get_figwidth()
panel_height = fig.get_figheight()
for ax, position, name in zip(axes, positions, ['time','storage','accuracy']):
    bounds = Bbox.from_bounds(position.x0*fig.get_figwidth(),0,panel_width,panel_height)
    for suffix in ['pdf','svg','png']:
        fig.savefig(HERE/f'{name}.{suffix}', dpi=300, facecolor='white',bbox_inches=bounds,pad_inches=0)
sizes = []
for name in ['time','storage','accuracy']:
    with Image.open(HERE/f'{name}.png') as im:
        sizes.append(im.size)
assert len(set(sizes)) == 1, f'Panel sizes differ: {sizes}'
plt.close(fig)
(HERE/'figure.tex').write_text(r'''\begin{figure*}[t]
\centering
\includegraphics[width=.32\linewidth]{PLOTS/intro_figs/mipa_mkdt_cifar10_p0p2_lbl1/time.pdf}\hfill
\includegraphics[width=.32\linewidth]{PLOTS/intro_figs/mipa_mkdt_cifar10_p0p2_lbl1/storage.pdf}\hfill
\includegraphics[width=.32\linewidth]{PLOTS/intro_figs/mipa_mkdt_cifar10_p0p2_lbl1/accuracy.pdf}
\caption{\textbf{MIPA versus MKDT on CIFAR-10 (0.2\% data, 1\% labels).} Left and middle: total distillation time and auxiliary storage from the reported cost table; times exclude initialization and teacher training, and storage denotes retained auxiliary files, not GPU memory. Right: current 2k-epoch SimCLR ResNet-18 accuracy (mean and sample SD); the accuracy baseline is 38\%, with the upper limit fitted to the data. MKDT pools six probes from two SSL seeds; MIPA uses five probes from one SSL seed. Cost and accuracy panels use their respective reported MIPA configurations.}
\label{fig:mipa-mkdt-cifar10-efficiency}
\end{figure*}
''')
print(json.dumps({k:snapshot[k] for k in ['time_speedup','storage_ratio','accuracy_gain_pp']}, indent=2))
print('PASS: table and accuracy verified; arrow text centered within 0.1 pixels along each arrow; fitted accuracy range', accuracy_limits, '; separate equal panel dimensions', sizes[0])
