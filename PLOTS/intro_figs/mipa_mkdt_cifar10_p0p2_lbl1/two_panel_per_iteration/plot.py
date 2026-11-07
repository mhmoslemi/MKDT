"""Compare the exact CIFAR-10 0.2% sets used in the current main table."""
import csv
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BENCH = ROOT/'Evaluate_benchamrk'
MIPA = BENCH/'mipa_cvpr_color_crop/results/CIFAR10/ResNet18BN/pt0.2/simclr/ssl_seed_000/summary_CIFAR10_lbl1.json'
MKDT = BENCH/'partial_probe_results/mkdt_CIFAR10_p0p2_simclr/CIFAR10_lbl1/summary.json'
MIPA_LOG = BENCH/'mipa_cvpr_color_crop/logs/cvprcropg_CIFAR10_pt0p2-2054186.out'


def read(path):
    return json.loads(path.read_text())


def evidence(path):
    return dict(path=str(path.relative_to(ROOT)), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


assert os.environ.get('SLURM_JOB_ID')
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
    config = read(Path(saved['samples'][0]['result_file']).parents[1]/'config.json')['signature']
    synthetic = Path(config['synthetic_path'] if name == 'MIPA' else config['synthetic']['path'])
    if name == 'MIPA':
        assert hashlib.sha256(synthetic.read_bytes()).hexdigest() == config['synthetic_sha256']
        log = MIPA_LOG.read_text()
        start = re.search(r'\[([^\]]+)\] distillation begins', log)[1]
        end = re.search(r'\[([^\]]+)\] iter = 10000,', log)[1]
        seconds = (datetime.fromisoformat(end)-datetime.fromisoformat(start)).total_seconds()
        iterations, logs = 10000, [MIPA_LOG]
        assert str(synthetic) in log and read(synthetic.parent/'complete.json')['complete']
    else:
        logs = [BENCH/f'mkdt_02_jobs/runs/logs/mk02_CIFAR10_dist_{step}-194354{i}.err'
                for i, step in enumerate(range(1000, 5001, 1000))]
        elapsed = [re.findall(r'1000/1000 \[(\d+):(\d+)<00:00', p.read_text())[-1] for p in logs]
        seconds = sum(int(m)*60+int(s) for m, s in elapsed)
        iterations = 5000
        last_log = logs[-1].with_suffix('.out').read_text()
        assert str(synthetic.parent) in last_log and 'RETAINED state after 5000/5000 updates' in last_log
    rows.append(dict(method=name, accuracy_percent=saved['mean_percent'], std_percent=saved['std_percent'],
        probes=len(scores), ssl_seeds=sorted({s['ssl_seed'] for s in saved['samples']}),
        optimization_seconds=seconds, iterations=iterations, seconds_per_iteration=seconds/iterations,
        summary=evidence(path), synthetic=evidence(synthetic), timing_logs=[evidence(p) for p in logs]))
hardware = (HERE/'hardware.psv').read_text().splitlines()
assert len(hardware) == 6 and all('|vulcan|COMPLETED|' in line and 'gres/gpu:l40s=1' in line for line in hardware)
mkdt, mipa = rows
assert mkdt['optimization_seconds'] == 2869 and mipa['optimization_seconds'] == 365
assert [row['probes'] for row in rows] == [6, 5]
speedup = mkdt['seconds_per_iteration']/mipa['seconds_per_iteration']
gain = mipa['accuracy_percent']-mkdt['accuracy_percent']
snapshot = dict(dataset='CIFAR10', data_percent=0.2, labels_percent=1, synthetic_images=100,
    ssl_epochs=2000, probe_epochs=100, rows=rows, per_iteration_speedup=speedup,
    total_optimization_speedup=mkdt['optimization_seconds']/mipa['optimization_seconds'], accuracy_gain_pp=gain,
    hardware='Historical Vulcan runs, each allocated one L40S, one CPU, and 6G RAM.',
    timing_definition='Observed optimization elapsed / completed updates; MKDT sums five sequential chunks.',
    exclusions=['initialization', 'teacher/expert preparation', 'queue gaps', 'downstream SSL and probes'],
    uncertainty='Error bars are sample SD across probes: MKDT 2 SSL seeds x 3 probes; MIPA 1 SSL seed x 5 probes.',
    limitation='Historical runs on different nodes/dates; no repeated controlled runtime benchmark.',
    hardware_source=evidence(HERE/'hardware.psv'), plot_source=evidence(Path(__file__)))
(HERE/'comparison.json').write_text(json.dumps(snapshot, indent=2)+'\n')
with (HERE/'comparison.csv').open('w', newline='') as out:
    writer = csv.DictWriter(out, fieldnames=['method','accuracy_percent','std_percent','probes','optimization_seconds','iterations','seconds_per_iteration'], extrasaction='ignore')
    writer.writeheader()
    writer.writerows(rows)

plt.rcParams.update({'font.family':'STIXGeneral', 'mathtext.fontset':'stix', 'font.size':9,
    'axes.titlesize':11, 'axes.labelsize':8, 'xtick.labelsize':9, 'ytick.labelsize':7.5,
    'pdf.fonttype':42, 'ps.fonttype':42, 'svg.fonttype':'none', 'axes.linewidth':0.6,
    'text.color':'#282c31', 'axes.labelcolor':'#5b656f', 'xtick.color':'#282c31', 'ytick.color':'#5b656f'})
fig, axes = plt.subplots(1, 2, figsize=(4.8, 3.1))
fig.subplots_adjust(left=.095, right=.98, bottom=.16, top=.76, wspace=.48)
fig.text(.5, .95, 'CIFAR-10', ha='center', va='top', fontsize=12, weight='bold')
fig.text(.5, .875, '0.2% data (100 images)  ·  1% labels', ha='center', va='top', fontsize=9, color='#5b656f')
fills, edges = ['#d9e6f4', '#f8deda'], ['#5b84b4', '#c76a64']
values = [[r['seconds_per_iteration']*1000 for r in rows], [r['accuracy_percent'] for r in rows]]
for ax, vals, title in zip(axes, values, ['Time per iteration', 'Accuracy']):
    ax.set_axisbelow(True)
    ax.grid(axis='y', color='#e8ebee', linewidth=.5)
    ax.bar([0,1], vals, width=.52, color=fills, edgecolor=edges, linewidth=.9, zorder=3)
    ax.set_xticks([0,1], ['MKDT', 'MIPA (ours)'])
    ax.set_xlim(-.52, 1.52)
    ax.set_title(title, pad=10)
    ax.tick_params(axis='both', length=0, pad=5)
    for side in ['top','right','left']:
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color('#929ba5')
axes[0].set_ylim(0, 740)
axes[0].set_yticks([0,200,400,600])
axes[0].set_ylabel('Milliseconds')
for x, y, label in [(0, values[0][0], f'{values[0][0]:.0f} ms'), (1, values[0][1], f'{values[0][1]:.1f} ms')]:
    axes[0].text(x, y+17, label, ha='center', va='bottom', fontsize=10)
axes[0].annotate('', xy=(.75,105), xytext=(.27,505),
                 arrowprops=dict(arrowstyle='->', color='#929ba5', linewidth=1.1))
axes[0].text(.50, 665, f'{speedup:.1f}'+r'$\times$ faster', ha='center', fontsize=11, weight='bold')
axes[1].set_ylim(0, 61)
axes[1].set_yticks([0,10,20,30,40,50])
axes[1].set_ylabel('Top-1 accuracy (%)')
axes[1].errorbar([0,1], values[1], yerr=[r['std_percent'] for r in rows],
                 fmt='none', ecolor='#5b656f', elinewidth=.7, capsize=2, capthick=.7, zorder=4)
for x, row in enumerate(rows):
    axes[1].text(x, row['accuracy_percent']+row['std_percent']+1.4,
                 f"{row['accuracy_percent']:.2f}%", ha='center', va='bottom', fontsize=10)
axes[1].annotate('', xy=(1,52), xytext=(0,52),
                 arrowprops=dict(arrowstyle='->', color='#929ba5', linewidth=1.1))
axes[1].text(.5, 55, f'+{gain:.2f} pp', ha='center', fontsize=11, weight='bold')
for suffix in ['pdf','svg','png']:
    fig.savefig(HERE/f'comparison.{suffix}', dpi=300, facecolor='white')
plt.close(fig)
(HERE/'figure.tex').write_text(r'''\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{intro_figs/mipa_mkdt_cifar10_p0p2_lbl1/comparison.pdf}
\caption{\textbf{MIPA versus MKDT on CIFAR-10.} Both use 100 synthetic images (0.2\% data), followed by 2k-epoch SimCLR with ResNet-18 and a 1\%-label linear probe. Left: observed optimization time per update on one L40S, excluding initialization and teacher/expert preparation. MKDT uses 5k updates (47.82 min total); MIPA uses 10k (6.08 min). Right: mean accuracy with sample SD; MKDT pools six probes from two SSL seeds, and MIPA uses five probes from one SSL seed. Timings are historical observations from different nodes and dates.}
\label{fig:mipa-mkdt-cifar10-efficiency}
\end{figure}
''')
print(json.dumps({k:snapshot[k] for k in ['per_iteration_speedup','total_optimization_speedup','accuracy_gain_pp']}, indent=2))
print('PASS: exact sets and probes verified; timing logs parsed; shared L40S allocation verified; figure exported.')
