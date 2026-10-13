"""Lightweight static/integration checks for the MKDT images-only benchmark."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'benchmarks'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import benchmark_mkdt_images
import report


def main():
    assert benchmark_mkdt_images.MODEL == 'ConvNet'
    assert benchmark_mkdt_images.RUNS == 10
    scores = {}
    for size in report.SIZES:
        for labels in report.LABELS:
            for loss in report.LOSSES:
                scores[('CIFAR10', size, labels, loss)] = '\\dsacc{1.23}{0.45}'
                for target in report.DOWNSTREAM:
                    scores[(target, size, labels, loss)] = '\\dsacc{1.23}{0.45}'

    main_template = (ROOT / 'CVPR_tables/cifar10_main.tex').read_text()
    downstream_template = (ROOT / 'CVPR_tables/cifar10_downstream.tex').read_text()
    main_table, main_count = report.fill_main(main_template, scores)
    downstream_table, downstream_count = report.fill_downstream(downstream_template, scores)
    assert main_count == 6
    assert downstream_count == 6
    assert main_table.count('& & MKDT & \\dsacc{1.23}{0.45} & \\dsacc{1.23}{0.45}') == 6
    assert downstream_table.split('% TABLE 2: VGG11', 1)[0].count(
        '& & MKDT & \\dsacc{1.23}{0.45}') == 6
    assert downstream_table.split('% TABLE 2: VGG11', 1)[1] == downstream_template.split(
        '% TABLE 2: VGG11', 1)[1]
    print('MKDT images-only benchmark validation passed.')


if __name__ == '__main__':
    main()
