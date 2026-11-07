"""Replot both MIPA figures: python plot_mipa_ablations.py (on a compute node).

Requires matplotlib. Reads the saved CSVs and overwrites their PDF/PNG figures.
Edit the settings below, or the plotting calls, to customize the figures.
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

FIGURES = Path(__file__).resolve().parent / 'CVPR_tables/ablations/figures'
FIGSIZE = (3.5, 2.8)  # Inches; identical for both figures.
DPI = 600
MIPA_COLOR = '#007F86'
RANDOM_COLOR = '#C47760'
MARGINS = dict(left=0.16, right=0.97, bottom=0.18, top=0.985)
LEGEND = dict(loc='upper right', frameon=True, fancybox=False,
              framealpha=0.97, fontsize=9, facecolor='white',
              edgecolor='#C5CED3', borderpad=0.25, handlelength=1.6,
              handletextpad=0.6, labelspacing=0.15, borderaxespad=0.25)
STYLE = {
    'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.labelsize': 11.5, 'axes.linewidth': 0.75,
    'axes.labelcolor': '#24343B', 'text.color': '#24343B',
    'xtick.color': '#46555D', 'ytick.color': '#46555D',
    'xtick.labelsize': 9.5, 'ytick.labelsize': 10,
    'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'mathtext.fontset': 'dejavusans', 'pdf.fonttype': 42, 'ps.fonttype': 42,
}
PLOTS = [
    ('mipa_temperature_cifar10_p0p2_lbl1', 'temperature', r'Assignment temperature $\tau$'),
    ('mipa_batch_size_cifar10_p1_lbl1', 'batch', 'Distillation batch size'),
]


def plot(stem, variable, xlabel):
    with (FIGURES / (stem + '.csv')).open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    baseline = next(row for row in rows if row['method'] == 'Random')
    mipa = [row for row in rows if row['method'] == 'MIPA']
    xs = [float(row[variable]) for row in mipa]
    ys = [float(row['mean_percent']) for row in mipa]
    stds = [float(row['std_percent']) for row in mipa]
    random_mean = float(baseline['mean_percent'])
    random_std = float(baseline['std_percent'])
    assert xs and all(x > 0 for x in xs), 'Logarithmic x-axis needs positive values'
    assert all(std >= 0 for std in [random_std, *stds]), 'Standard deviations must be nonnegative'

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_xscale('log', base=2)
    ax.fill_between([min(xs), max(xs)], random_mean - random_std,
                    random_mean + random_std, color=RANDOM_COLOR,
                    alpha=0.12, linewidth=0, zorder=1)
    random_line, = ax.plot([min(xs), max(xs)], [random_mean] * 2,
                           color=RANDOM_COLOR, linewidth=1.5,
                           linestyle=(0, (4, 2.5)), label='Random', zorder=2)
    mipa_line = ax.errorbar(xs, ys, yerr=stds, color=MIPA_COLOR, linewidth=1.8,
                           marker='o', markersize=4.8, markerfacecolor='white',
                           markeredgewidth=1.2, elinewidth=1.1,
                           capsize=3, capthick=1.1, label='MIPA', zorder=3)
    ax.set_xlim(min(xs) / 1.12, max(xs) * 1.12)
    lower = min([random_mean - random_std] + [y - s for y, s in zip(ys, stds)])
    upper = max([random_mean + random_std] + [y + s for y, s in zip(ys, stds)])
    padding = (upper - lower) * 0.04
    ax.set_ylim(lower - padding, upper + padding)
    ax.set_xticks(xs, [row[variable] for row in mipa])
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.minorticks_off()
    ax.set_xlabel(xlabel, labelpad=5)
    ax.set_ylabel('Accuracy (%)', labelpad=5)
    ax.grid(axis='y', color='#E1E6E9', linewidth=0.55)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color('#A3AEB5')
    ax.legend(handles=[mipa_line, random_line], **LEGEND)
    fig.subplots_adjust(**MARGINS)
    # Fixed canvas preserves matching dimensions; tight cropping can change them.
    for extension in ('pdf', 'png'):
        output = FIGURES / (stem + '.' + extension)
        fig.savefig(output, dpi=DPI, facecolor='white')
        print('Saved:', output)
    plt.close(fig)


if __name__ == '__main__':
    plt.rcParams.update(STYLE)
    for args in PLOTS:
        plot(*args)
