"""Probe check: RGB-fixed origin run vs official raw boxes on 4 divergent seqs."""
import io
import zipfile

import numpy as np

ROOT = 'E:/gitProjects/OSTrack'
z = zipfile.ZipFile(f'{ROOT}/vitb_384_mae_ce_32x4_ep300/got10k.zip')

for sid in ['000015', '000031', '000042', '000093']:
    seq = f'GOT-10k_Test_{sid}'
    off = np.loadtxt(io.BytesIO(z.read(f'got10k/{seq}.txt')))
    ours = np.loadtxt(f'{ROOT}/output/batch_eval/rgbfix_probe/submission/{seq}/{seq}_001.txt',
                      delimiter=',')
    n = min(len(off), len(ours))
    a, b = off[:n], ours[:n]
    ix = np.maximum(0, np.minimum(a[:, 0] + a[:, 2], b[:, 0] + b[:, 2]) - np.maximum(a[:, 0], b[:, 0]))
    iy = np.maximum(0, np.minimum(a[:, 1] + a[:, 3], b[:, 1] + b[:, 3]) - np.maximum(a[:, 1], b[:, 1]))
    inter = ix * iy
    iou = inter / (a[:, 2] * a[:, 3] + b[:, 2] * b[:, 3] - inter + 1e-9)
    print(f'{seq}  meanIoU={iou.mean():.4f}  min={iou.min():.3f}  f2={iou[1]:.3f}')
