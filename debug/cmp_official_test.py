"""Compare our GOT-10k test boxes against the official OSTrack raw results.

Official raw results: vitb_384_mae_ce_32x4_ep300/got10k.zip  (ep300 weights)
Ours               : output/batch_eval/<tag>/submission/<seq>/<seq>_001.txt

Per-sequence mean IoU tells whether our inference pipeline reproduces the
official tracker (IoU ~1 => same trajectory; low IoU => divergence).
"""
import io
import sys
import zipfile

import numpy as np

ROOT = 'E:/gitProjects/OSTrack'
OFFICIAL_ZIP = f'{ROOT}/vitb_384_mae_ce_32x4_ep300/got10k.zip'
TAG = sys.argv[1] if len(sys.argv) > 1 else 'got10k_test_origin'


def iou(a, b):
    ax1, ay1, aw, ah = a.T
    bx1, by1, bw, bh = b.T
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix = np.maximum(0, np.minimum(ax2, bx2) - np.maximum(ax1, bx1))
    iy = np.maximum(0, np.minimum(ay2, by2) - np.maximum(ay1, by1))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / np.maximum(union, 1e-9)


def main():
    z = zipfile.ZipFile(OFFICIAL_ZIP)
    # locate the submission-layout dir (run_got10k.py names it 'submission',
    # but it may have been renamed before upload, e.g. 'osOrigin')
    import glob
    import os
    cands = [d for d in glob.glob(f'{ROOT}/output/batch_eval/{TAG}/*')
             if os.path.isdir(f'{d}/GOT-10k_Test_000001')]
    if not cands:
        raise SystemExit(f'no submission-layout dir under {TAG}')
    sub = cands[0].replace('\\', '/')
    print('using boxes from:', sub)
    rows = []
    for i in range(1, 181):
        seq = f'GOT-10k_Test_{i:06d}'
        official = np.loadtxt(io.BytesIO(z.read(f'got10k/{seq}.txt')))
        ours = np.loadtxt(f'{sub}/{seq}/{seq}_001.txt', delimiter=',')
        n = min(len(official), len(ours))
        m = iou(official[:n], ours[:n]).mean()
        rows.append((seq, m, len(official), len(ours)))

    rows.sort(key=lambda r: r[1])
    print(f'{"seq":<24}{"meanIoU":>9}  frames(off/ours)')
    for seq, m, lo, lu in rows[:20]:
        print(f'{seq:<24}{m:9.3f}  {lo}/{lu}')
    ious = np.array([r[1] for r in rows])
    print('-' * 50)
    print(f'sequences: {len(rows)}, mean IoU: {ious.mean():.4f}, median: {np.median(ious):.4f}')
    for t in (0.9, 0.7, 0.5, 0.3):
        print(f'  seqs with meanIoU < {t}: {(ious < t).sum()}')


if __name__ == '__main__':
    main()
