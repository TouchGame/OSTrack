"""Frame-level divergence check: our origin run vs official raw boxes."""
import io
import zipfile

import numpy as np

ROOT = 'E:/gitProjects/OSTrack'
z = zipfile.ZipFile(f'{ROOT}/vitb_384_mae_ce_32x4_ep300/got10k.zip')


def iou1(a, b):
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-9)


for sid in ['000042', '000015', '000031', '000164', '000107', '000093']:
    seq = f'GOT-10k_Test_{sid}'
    off = np.loadtxt(io.BytesIO(z.read(f'got10k/{seq}.txt')))
    ours = np.loadtxt(f'{ROOT}/output/batch_eval/got10k_test_origin/osOrigin/{seq}/{seq}_001.txt',
                      delimiter=',')
    ious = [iou1(off[i], ours[i]) for i in range(min(len(off), len(ours)))]
    fb = next((i for i, v in enumerate(ious) if v < 0.9), -1)
    ft = next((i for i, v in enumerate(ious) if v < 0.5), -1)
    print(f'{seq}  f2={ious[1]:.3f} f3={ious[2]:.3f} f5={ious[4]:.3f} f10={ious[9]:.3f}'
          f' | first<0.9: f{fb + 1}, first<0.5: f{ft + 1}')
