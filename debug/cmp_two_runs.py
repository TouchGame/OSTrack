"""Frame-level comparison of two run_got10k box outputs.

Usage: python cmp_two_runs.py <boxes_dir_A> <boxes_dir_B>
For every sequence present in BOTH dirs: per-frame IoU, exact-match rate,
max center/size deviation. Verdict: identical / numerically-equal / diverged.
"""
import os
import sys

import numpy as np


def iou(a, b):
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def main():
    dir_a, dir_b = sys.argv[1], sys.argv[2]
    seqs = sorted(set(os.listdir(dir_a)) & set(os.listdir(dir_b)))
    seqs = [s for s in seqs if s.endswith('.txt')]
    if not seqs:
        sys.exit('no common sequences')

    all_ious, worst = [], []
    n_exact_seq = 0
    print(f'{"Sequence":<24} {"frames":>6} {"meanIoU":>8} {"minIoU":>8} {"maxAbsDiff":>10}')
    for s in seqs:
        A = np.loadtxt(os.path.join(dir_a, s), delimiter=',')
        B = np.loadtxt(os.path.join(dir_b, s), delimiter=',')
        n = min(len(A), len(B))
        A, B = np.atleast_2d(A)[:n], np.atleast_2d(B)[:n]
        ious = np.array([iou(A[i], B[i]) for i in range(n)])
        max_diff = float(np.abs(A - B).max())
        exact = bool(np.array_equal(A, B))
        n_exact_seq += exact
        all_ious.append(ious)
        worst.append((float(ious.min()), s))
        flag = ' EXACT' if exact else ''
        print(f'{s[:-4]:<24} {n:>6} {ious.mean():>8.4f} {ious.min():>8.4f} {max_diff:>10.2f}{flag}')

    pooled = np.concatenate(all_ious)
    print('-' * 62)
    print(f'Sequences: {len(seqs)}, bit-exact: {n_exact_seq}')
    print(f'Pooled mean IoU: {pooled.mean():.4f}, median: {np.median(pooled):.4f}')
    print(f'Frames with IoU<0.9: {int((pooled < 0.9).sum())} / {len(pooled)}'
          f'  (<0.5: {int((pooled < 0.5).sum())})')
    worst.sort()
    print('Worst 5 sequences by min IoU:', [(f'{v:.3f}', s[:-4]) for v, s in worst[:5]])


if __name__ == '__main__':
    main()
