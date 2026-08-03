"""Inspect the ep100 checkpoint: size, md5, epoch, arch fingerprint."""
import hashlib
import os

import torch

PATH = ('E:/gitProjects/OSTrack/output/checkpoints/train/ostrack/'
        'vitb_384_mae_ce_32x4_got10k_ep100/OSTrack_ep0100.pth.tar')

size = os.path.getsize(PATH)
print(f'file size : {size} bytes ({size / 1024 / 1024:.1f} MB)')

md5 = hashlib.md5()
with open(PATH, 'rb') as f:
    for chunk in iter(lambda: f.read(1 << 20), b''):
        md5.update(chunk)
print(f'md5       : {md5.hexdigest()}')

ckpt = torch.load(PATH, map_location='cpu')
print(f'top keys  : {list(ckpt.keys())}')
print(f'epoch     : {ckpt.get("epoch")}')
net = ckpt.get('net', ckpt)
print(f'net params: {len(net)} tensors')
# a few weight fingerprints to compare against the ep300 checkpoint
for k in ['backbone.pos_embed_z', 'backbone.blocks.0.attn.qkv.weight',
          'box_head.conv1_ctr.0.weight']:
    if k in net:
        t = net[k]
        print(f'{k}: shape={tuple(t.shape)}, mean={t.float().mean():.6f}, std={t.float().std():.6f}')

# embedded training info, if present
for key in ('settings', 'constructor', 'actor_type', 'net_type', 'net_info'):
    if key in ckpt and ckpt[key] is not None:
        print(f'{key}: {str(ckpt[key])[:300]}')
