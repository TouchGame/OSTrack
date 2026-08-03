"""Compare AO across GOT-10k eval runs (by tag) on their common sequences.

Usage:
    python debug/cmp_got10k.py origin_tag other_tag [other_tag2 ...]
"""
import csv
import os
import sys

import numpy as np

BASE = os.path.join(os.path.dirname(__file__), '..', 'output', 'batch_eval')


def load(tag):
    """Positional parse: col0=sequence, col2=ao. Robust against hand-edited
    CSVs that append extra comparison columns with duplicate 'ao' headers
    (DictReader would silently keep the LAST duplicate)."""
    path = os.path.join(BASE, tag, 'metrics.csv')
    out = {}
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # header
        for row in reader:
            if len(row) >= 3 and row[0] and row[2]:
                try:
                    out[row[0]] = float(row[2])
                except ValueError:
                    pass
    return out


def main():
    tags = sys.argv[1:]
    if len(tags) < 2:
        sys.exit('need at least two run tags')
    runs = {t: load(t) for t in tags}
    common = sorted(set.intersection(*(set(r) for r in runs.values())))
    base = tags[0]

    header = f'{"seq":<22}' + ''.join(f'{t[-12:]:>14}' for t in tags)
    print(header)
    print('-' * len(header))
    for s in sorted(common, key=lambda x: runs[tags[-1]][x] - runs[base][x]):
        cells = ''.join(f'{runs[t][s]:>14.3f}' for t in tags)
        print(f'{s:<22}{cells}')
    print('-' * len(header))
    means = ''.join(f'{np.mean([runs[t][s] for s in common]):>14.4f}' for t in tags)
    print(f'{"MEAN":<22}{means}')
    for t in tags[1:]:
        diff = [runs[t][s] - runs[base][s] for s in common]
        print(f'{t} vs {base}: mean {np.mean(diff):+.4f}, '
              f'worse>0.01: {sum(1 for x in diff if x < -0.01)}, '
              f'better>0.01: {sum(1 for x in diff if x > 0.01)}')


if __name__ == '__main__':
    main()
