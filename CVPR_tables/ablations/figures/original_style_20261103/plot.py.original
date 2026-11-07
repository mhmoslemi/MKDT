"""Plot CIFAR10 batch-size and temperature ablations against matched Random baselines."""
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter, FixedLocator, MultipleLocator

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
ROOT = BENCH.parent
OUTPUT = ROOT / 'CVPR_tables/ablations/figures/mipa_batch_size_cifar10_p1_lbl1'
BATCHES = [128, 256, 512, 1024, 2048, 4096]
RANDOM = BENCH / 'cifar_small_simclr2k_1ssl5probes/results/CIFAR10/ResNet18/random/pt1/simclr/ssl_seed_000'
EXISTING = BENCH / 'ours_eval_results_simclr2k_barlow10k_2ssl3probes_main/CIFAR10/ResNet18BN/pt1/simclr/ssl_seed_000'


def read(file):
    return json.loads(Path(file).read_text())


def file_hash(file):
    digest = hashlib.sha256()
    with Path(file).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def point(folder, batch, seeds, summary_file=None, subset=1):
    config = read(folder / 'config.json')
    signature, identity = config['signature'], config['identity']
    assert (signature['source'], signature['model'], signature['subset'], signature['ssl']) == ('CIFAR10', 'ResNet18', subset, 'simclr')
    assert identity['seed'] == 0
    protocol = signature['protocol']
    assert protocol['ssl']['subset_epochs'] == 2000 and protocol['evaluation']['epochs'] == 100
    marker = read(folder / 'ssl_complete.json')
    assert marker['complete'] and marker['epochs'] == 2000 and marker['identity'] == identity
    assert file_hash(folder / 'encoder.pt') == marker['checkpoint_file_sha256']
    scores, files = [], []
    for seed in seeds:
        file = folder / 'evaluations' / ('CIFAR10_lbl1_probe%d.json' % seed)
        probe = read(file)
        expected = probe['expected']
        assert probe['complete'] and probe['encoder_unchanged']
        assert probe['encoder_sha256'] == marker['encoder_sha256']
        assert expected['identity'] == identity and expected['evaluation'] == protocol['evaluation']
        assert expected['evaluation_code_hashes'] == signature['code_hashes']
        assert (expected['source'], expected['target'], expected['label_percentage'], expected['seed']) == ('CIFAR10', 'CIFAR10', 1, seed)
        assert math.isfinite(probe['accuracy_percent']) and 0 <= probe['accuracy_percent'] <= 100
        scores.append(probe['accuracy_percent'])
        files.append(str(file.relative_to(ROOT)))
    mean, std = statistics.mean(scores), statistics.stdev(scores)
    if summary_file:
        saved = read(summary_file)
        assert saved['complete'] and saved['completed_probes'] == len(seeds) == saved['expected_probes']
        assert saved['std_ddof'] == 1 and saved['ssl_epochs'] == 2000 and saved['linear_epochs'] == 100
        assert math.isclose(saved['mean_percent'], mean, abs_tol=1e-10)
        assert math.isclose(saved['std_percent'], std, abs_tol=1e-10)
    return dict(method='Random' if batch is None else 'MIPA', batch=batch,
                mean_percent=mean, std_percent=std, probe_count=len(seeds), ssl_seed=0,
                ssl_epochs=2000, linear_epochs=100, labels=1, scores=scores,
                config_file=str((folder / 'config.json').relative_to(ROOT)), probe_files=files), protocol


def main():
    baseline, protocol = point(RANDOM, None, range(5), RANDOM / 'summary_lbl1.json')
    completed, missing = [], []
    for batch in BATCHES:
        if batch == 1024:
            row, reference = point(EXISTING, batch, range(3))
            row['existing_run'] = True
        else:
            folder = HERE / ('results/batch_%d/simclr/ssl_seed_000' % batch)
            summary = folder / 'summary_lbl1.json'
            if not summary.exists():
                missing.append(batch)
                continue
            args = read(summary)['generation_args']
            assert (args['batch_real'], args['Iteration'], args['lr_img'], args['disp_weight'], args['pair_weight']) == (batch, 8000, 0.001, 0, 0)
            row, reference = point(folder, batch, range(5), summary)
            row['existing_run'] = False
        for key in ('ssl', 'evaluation', 'input_size', 'normalization'):
            assert reference[key] == protocol[key], 'Random/MIPA protocol mismatch: ' + key
        assert reference['methods']['simclr'] == protocol['methods']['simclr']
        completed.append(row)
    assert completed
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    snapshot = dict(created_utc=datetime.now(timezone.utc).isoformat(), dataset='CIFAR10',
                    subset_percent=1, synthetic_images=500, model='ResNet18', labels_percent=1,
                    distillation_iterations=8000, generation_lr=0.001,
                    baseline=baseline, completed_mipa=completed, unfinished_batches=missing,
                    excluded_batches=[8192],
                    plotted_statistic='mean accuracy with plus/minus one sample standard deviation',
                    uncertainty='sample standard deviation over linear probes; one SSL encoder per point',
                    existing_batch_1024_note='Existing SSL seed 0 and its three completed probes; no rerun')
    save_plot(snapshot, OUTPUT, 'batch', 'Distillation batch size', BATCHES)


def temperature():
    values = [0.05, 0.1, 0.2, 0.4, 0.8]
    random = RANDOM.parents[2] / 'pt0.2/simclr/ssl_seed_000'
    baseline, protocol = point(random, None, range(5), random / 'summary_lbl1.json', subset=0.2)
    baseline['temperature'] = baseline.pop('batch')
    completed, common_args = [], None
    for value in values:
        variant = 'tau_' + str(value).replace('.', 'p') + '_lr0p001'
        folder = BENCH / 'mipa_temperature_ablation/results' / variant / 'simclr/ssl_seed_000'
        summary = folder / 'summary_lbl1.json'
        saved = read(summary)
        args = dict(saved['generation_args'])
        assert args.pop('temperature') == value
        assert (args['percentage'], args['Iteration'], args['batch_real'], args['lr_img'], args['disp_weight'], args['pair_weight']) == (0.2, 10000, 256, 0.001, 0, 0)
        assert saved['initialization'] == dict(method='random', seed=0, kmeans_iterations=0)
        if common_args is None:
            common_args = args
        assert args == common_args, 'Temperature runs differ in other generation settings'
        row, reference = point(folder, value, range(5), summary, subset=0.2)
        row['temperature'] = row.pop('batch')
        for key in ('ssl', 'evaluation', 'input_size', 'normalization'):
            assert reference[key] == protocol[key], 'Random/MIPA protocol mismatch: ' + key
        assert reference['methods']['simclr'] == protocol['methods']['simclr']
        completed.append(row)
    snapshot = dict(created_utc=datetime.now(timezone.utc).isoformat(), dataset='CIFAR10',
                    subset_percent=0.2, synthetic_images=100, model='ResNet18', labels_percent=1,
                    distillation_iterations=10000, generation_lr=0.001, generation_batch=256,
                    initialization=dict(method='random', seed=0, kmeans_iterations=0),
                    baseline=baseline, completed_mipa=completed,
                    plotted_statistic='mean accuracy with plus/minus one sample standard deviation',
                    uncertainty='sample standard deviation over five linear probes; one SSL encoder per point')
    output = OUTPUT.with_name('mipa_temperature_cifar10_p0p2_lbl1')
    save_plot(snapshot, output, 'temperature', r'Assignment temperature $\tau$', values)


def save_plot(snapshot, output, variable, xlabel, ticks):
    baseline, completed = snapshot['baseline'], snapshot['completed_mipa']
    columns = ['method', variable, 'mean_percent', 'std_percent', 'probe_count', 'ssl_seed', 'ssl_epochs', 'linear_epochs', 'labels']
    with output.with_suffix('.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows([baseline] + completed)

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
                         'axes.labelsize': 8.5, 'axes.linewidth': 0.65,
                         'xtick.labelsize': 7, 'ytick.labelsize': 7.5,
                         'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
                         'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, ax = plt.subplots(figsize=(3.4, 2.55))
    blue = '#1f77b4'
    color = '#2ca02c' if variable == 'temperature' else '#ff7f0e'
    ax.set_xscale('log', base=2)
    xs = [row[variable] for row in completed]
    ys = [row['mean_percent'] for row in completed]
    deviations = [row['std_percent'] for row in completed]
    ax.plot([min(xs), max(xs)], [baseline['mean_percent']] * 2,
            color=blue, linewidth=1.3, label='Random')
    for sign in (-1, 1):
        ax.plot([min(xs), max(xs)], [baseline['mean_percent'] + sign * baseline['std_percent']] * 2,
                color=blue, linewidth=0.7, linestyle=':', alpha=0.8)
    ax.errorbar(xs, ys, yerr=deviations, color=color, linewidth=1.3,
                marker='o', markersize=2.8, markeredgewidth=0,
                elinewidth=0.8, capsize=2.2, capthick=0.8, label='MIPA', zorder=3)
    ax.set_xlim(min(xs) / 1.12, max(xs) * 1.12)
    lower = min([baseline['mean_percent'] - baseline['std_percent']] +
                [row['mean_percent'] - row['std_percent'] for row in completed])
    upper = max([baseline['mean_percent'] + baseline['std_percent']] +
                [row['mean_percent'] + row['std_percent'] for row in completed])
    padding = (upper - lower) * 0.04
    ax.set_ylim(lower - padding, upper + padding)
    ticks = [value for value in ticks if min(xs) <= value <= max(xs)]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FixedFormatter([str(batch) for batch in ticks]))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.set_xlabel(xlabel, labelpad=3)
    ax.set_ylabel('Accuracy (%)', labelpad=3)
    ax.grid(which='major', axis='both', color='#dddddd', linewidth=0.45)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color('#777777')
    legend = ax.legend(loc='upper right', frameon=True, fancybox=False, fontsize=7.5,
                       handlelength=1.8, borderpad=0.35, labelspacing=0.3,
                       framealpha=0.92, edgecolor='#cccccc')
    legend.get_frame().set_linewidth(0.45)
    fig.subplots_adjust(left=0.16, right=0.975, bottom=0.19, top=0.975)
    fig.savefig(output.with_suffix('.png'), dpi=600, facecolor='white')
    fig.savefig(output.with_suffix('.pdf'), facecolor='white')
    snapshot['ylim'] = list(ax.get_ylim())
    snapshot['colors'] = dict(Random=blue, MIPA=color)
    plt.close(fig)
    output.with_suffix('.json').write_text(json.dumps(snapshot, indent=2) + '\n')
    print(json.dumps(snapshot, indent=2))
    print('Saved:', output.with_suffix('.png'))
    print('Saved:', output.with_suffix('.pdf'))


if __name__ == '__main__':
    main()
    temperature()
