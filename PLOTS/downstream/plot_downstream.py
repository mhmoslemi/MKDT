"""Export twelve native-size CVPR panels from the verified downstream results."""
# %% Imports and settings shared with plot_downstream.ipynb
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

HERE = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
if not (HERE/'figures/downstream_results.csv').is_file():
    HERE = HERE/'PLOTS/downstream'
DATA = HERE/'figures/downstream_results.csv'
OUT = HERE/'figures/individual'
FIGSIZE = (1.65, 1.35)  # Native width in the four-column CVPR figure; do not crop tightly.
FONT_FAMILY = 'DejaVu Sans'  # Bundled with Matplotlib.
TICK_SIZE = 8
TITLE_SIZE = 9
GROUP_SIZE = 8
TARGETS = {'CIFAR10': ['CIFAR100', 'TinyImageNet', 'Aircraft', 'Dogs'],
           'CIFAR100': ['CIFAR10', 'TinyImageNet', 'Aircraft', 'Dogs'],
           'TinyImageNet': ['CIFAR10', 'CIFAR100', 'Aircraft', 'Dogs']}
NAMES = {'CIFAR10': 'CIFAR-10', 'CIFAR100': 'CIFAR-100',
         'TinyImageNet': 'Tiny ImageNet', 'Aircraft': 'Aircraft', 'Dogs': 'Dogs'}
METHODS = ['Random', 'K-means', 'MKDT', 'MIPA']
COLORS = ['#4D82AC', '#568F7D', '#8E78A0', '#C45E3C']
MARKERS = ['o', 's', '^', 'D']

# %% Load the same two-decimal values verified against 1,269 completed probe logs

def load_results():
    with DATA.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row['labeled_percent'] = int(row['labeled_percent'])
        for key in ('subset_percent', 'plot_mean', 'plot_std'):
            row[key] = float(row[key])
            assert math.isfinite(row[key])
        assert row['plot_mean'] > 0 and row['plot_std'] >= 0
    keys = {(r['source'], r['target'], r['method'], r['subset_percent'], r['labeled_percent']) for r in rows}
    assert len(rows) == len(keys) == 216
    return rows

# %% One downstream panel, with independently scaled left/right label budgets

def draw(source, target, rows):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    width, height = [72*x for x in FIGSIZE]
    fig.subplots_adjust(left=14.5/width, right=1-14.5/width,
                        bottom=16/height, top=1-22/height)
    right_axis = ax.twinx()
    cells = [r for r in rows if r['source'] == source and r['target'] == target]
    assert len(cells) == 18
    lookup = {(r['method'], r['subset_percent'], r['labeled_percent']): r for r in cells}
    centers, offsets = [0, 1, 2.25, 3.25], [-.30, -.10, .10, .30]
    candidates = [v*10**e for e in range(-1, 3) for v in (1, 1.5, 2, 3, 5, 8)] + [40, 60]
    for j, (labels, axis) in enumerate(((5, ax), (1, right_axis))):
        budget = [r for r in cells if r['labeled_percent'] == labels]
        full = lookup['Full data', 100, labels]
        assert full['plot_mean'] >= max(r['plot_mean'] for r in budget)
        low = min(r['plot_mean']-r['plot_std'] for r in budget)
        high = max(r['plot_mean']+r['plot_std'] for r in budget)
        assert low > 0
        margin = .05*math.log(high/low)
        lower, upper = low*math.exp(-margin), high*math.exp(margin)
        axis.set_yscale('log')
        axis.set_ylim(lower, upper)
        ticks = sorted(v for v in candidates if lower < v < upper)
        if len(ticks) < 3:
            ticks = sorted(ticks+[v for v in (2.5, 4, 6, 7, 25, 70) if lower < v < upper])
        if len(ticks) > 4:
            ticks = [ticks[0], ticks[len(ticks)//2], ticks[-1]]
        axis.yaxis.set_major_locator(FixedLocator(ticks))
        axis.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:g}'))
        axis.yaxis.set_minor_locator(NullLocator())
        axis.tick_params(axis='y', direction='out', length=1.5, width=.5, pad=.6, labelsize=TICK_SIZE)
        axis.grid(False)
        left, right = centers[2*j]-.39, centers[2*j+1]+.39
        mid = (left+right)/2
        axis.hlines(ticks, left, right, color='0.91', linewidth=.35, zorder=0)
        axis.hlines(full['plot_mean'], left, right, color='0.35', linestyle=(0, (3, 2)), linewidth=.55)
        axis.errorbar(mid, full['plot_mean'], yerr=full['plot_std'], fmt='o', color='0.35',
                      markerfacecolor='white', markersize=2.6, markeredgewidth=.6,
                      elinewidth=.6, capsize=1.1, capthick=.6, zorder=4)
        ax.text(mid, 1.015, f'{labels}%', transform=ax.get_xaxis_transform(),
                ha='center', va='bottom', fontsize=GROUP_SIZE, color='0.3')
        for k, subset in enumerate((.2, 1)):
            for m, (method, color, marker) in enumerate(zip(METHODS, COLORS, MARKERS)):
                value = lookup[method, subset, labels]
                axis.errorbar(centers[2*j+k]+offsets[m], value['plot_mean'], yerr=value['plot_std'],
                              fmt=marker, color=color, markerfacecolor=color, markersize=3,
                              markeredgewidth=.45, elinewidth=.6, capsize=1.1, capthick=.6, zorder=5)
        assert all(lower < r['plot_mean']-r['plot_std'] <= r['plot_mean']+r['plot_std'] < upper for r in budget)
    ax.set_xlim(-.43, 3.68)
    ax.set_xticks(centers, ['0.2', '1']*2)
    ax.tick_params(axis='x', direction='out', length=1.8, width=.5, pad=1, labelsize=TICK_SIZE)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_linewidth(.5)
    ax.spines[['left', 'bottom']].set_color('0.35')
    right_axis.spines[['top', 'left', 'bottom']].set_visible(False)
    right_axis.spines['right'].set_linewidth(.5)
    right_axis.spines['right'].set_color('0.35')
    ax.set_title(NAMES[target], fontsize=TITLE_SIZE, pad=12)
    assert not ax.get_ylabel() and not right_axis.get_ylabel() and not fig.legends
    fig.canvas.draw()
    renderer, bounds = fig.canvas.get_renderer(), fig.bbox
    for text in fig.findobj(matplotlib.text.Text):
        if text.get_visible() and text.get_text():
            box = text.get_window_extent(renderer)
            assert box.x0 >= -1 and box.y0 >= -1 and box.x1 <= bounds.x1+1 and box.y1 <= bounds.y1+1, text.get_text()
    return fig

# %% Export exactly twelve equal-size PDF panels; filenames identify both datasets

def main():
    rows = load_results()
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': FONT_FAMILY, 'font.size': TICK_SIZE,
                         'text.color': 'black', 'axes.labelcolor': 'black',
                         'xtick.color': 'black', 'ytick.color': 'black',
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    paths = []
    for source, targets in TARGETS.items():
        for target in targets:
            figure = draw(source, target, rows)
            path = OUT/f'{source.lower()}_pretrain_{target.lower()}_downstream.pdf'
            figure.savefig(path, facecolor='white', metadata={
                'Title': NAMES[source]+' pretraining; '+NAMES[target]+' downstream',
                'CreationDate': None, 'ModDate': None})
            plt.close(figure)
            paths.append(path)
            print(path.name)
    assert len(paths) == 12
    print(f'Saved 12 PDFs at {FIGSIZE[0]:.2f} x {FIGSIZE[1]:.2f} inches; ticks {TICK_SIZE} pt, titles {TITLE_SIZE} pt.')
    return paths


if __name__ == '__main__':
    main()
