"""Run the tracker on the DiDi distractor dataset and compute metrics.

The dataset aggregates sequences from multiple sources (GOT-10k, LaSOT, UTB180,
VOT2022-ST, VOT2022-LT, VOT2020-ST, VOT2020-LT), each with a different
ground-truth format:

  - GOT-10k  : comma-separated x,y,w,h floats  (e.g. 401.0000,276.0000,101.0000,68.0000)
  - LaSOT     : comma-separated x,y,w,h ints    (e.g. 347,135,19,18)
  - UTB180    : comma-separated x,y,w,h floats  (e.g. 1534.34,405.24,138.45,107.76)
  - VOT2022-LT: comma-separated x,y,w,h floats  (e.g. 654.0,388.0,33.0,29.0)
  - VOT2020-LT : comma-separated x,y,w,h floats
  - VOT2022-ST : polygon ground-truth (first token is mXXX metadata, the rest are
                 alternating x,y polygon-vertex coordinates; axis-aligned bbox is
                 extracted as min/max of all x and y values)
  - VOT2020-ST : same polygon format as VOT2022-ST

Metrics (all frames, no filtering):
  Success AUC : mean IoU over 21 thresholds [0:0.05:1]
  Precision   : fraction of frames with center error <= 20 px
  Norm Prec  : AUC over 51 normalized-center-error thresholds [0:0.01:0.5]

Usage:
    python tracking/run_dididata.py --seq_file E:/gitProjects/data/DiDiData/list_mini.txt \\
        --data_dir E:/gitProjects/data/DiDiData --tag didi_mech
"""
import os
import sys
import csv
import time
import argparse

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

import cv2 as cv
import numpy as np

from lib.test.evaluation import Tracker


# ── Ground-truth loading ────────────────────────────────────────────────────────

def _is_polygon_gt(first_line: str) -> bool:
    """Detect polygon ground-truth (VOT-ST format) by the mXXX metadata prefix."""
    return first_line.strip().startswith('m')


def load_groundtruth(gt_file: str):
    """Auto-detect format and load ground-truth boxes as a list of [x,y,w,h] lists.

    Polygon formats (VOT-ST): first token is mXXX metadata, the remaining numeric
    tokens are alternating x,y polygon-vertex coordinates; the axis-aligned bbox
    is extracted as (x_min, y_min, x_max-x_min, y_max-y_min).
    """
    if not os.path.isfile(gt_file):
        return None
    with open(gt_file, 'r') as f:
        lines = f.readlines()

    boxes = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split(',')
        if _is_polygon_gt(line):
            # Polygon: skip the mXXX metadata token at index 0
            numeric = [float(p) for p in parts[1:] if p.replace('.', '').replace('-', '').isdigit()]
            xs = numeric[0::2]
            ys = numeric[1::2]
            x_min, y_min = min(xs), min(ys)
            x_max, y_max = max(xs), max(ys)
            boxes.append([x_min, y_min, x_max - x_min, y_max - y_min])
        else:
            # Standard x,y,w,h (float or int)
            # Skip lines with fewer than 4 values (occlusion markers like "0")
            if len(parts) < 4:
                continue
            vals = [float(p) for p in parts[:4]]
            # Skip fully-occluded frames (all zeros)
            if all(v == 0.0 for v in vals):
                continue
            boxes.append(vals)

    return boxes if boxes else None


# ── IOU / tracking metrics ──────────────────────────────────────────────────

def compute_iou(a, b):
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 1e-6 else 0.0


def evaluate_sequence(boxes, gt):
    """LaSOT-style OPE metrics: Success AUC / Precision@20 / Norm-Precision / mIoU."""
    n = min(len(boxes), len(gt))
    ious, cerrs, nerrs = [], [], []
    for i in range(n):
        b, g = boxes[i], gt[i]
        if g[2] <= 0 or g[3] <= 0:
            continue
        ious.append(compute_iou(b, g))
        bcx = b[0] + b[2] / 2.0
        bcy = b[1] + b[3] / 2.0
        gcx = g[0] + g[2] / 2.0
        gcy = g[1] + g[3] / 2.0
        cerrs.append(np.hypot(bcx - gcx, bcy - gcy))
        nerrs.append(np.hypot((bcx - gcx) / max(g[2], 1e-6), (bcy - gcy) / max(g[3], 1e-6)))
    if not ious:
        return None
    ious = np.array(ious)
    cerrs = np.array(cerrs)
    nerrs = np.array(nerrs)
    iou_thr = np.linspace(0, 1, 21)
    nrm_thr = np.linspace(0, 0.5, 51)
    return {
        'auc': float(np.mean([np.mean(ious > t) for t in iou_thr])),
        'prec20': float(np.mean(cerrs <= 20)),
        'nprec': float(np.mean([np.mean(nerrs <= t) for t in nrm_thr])),
        'miou': float(np.mean(ious)),
        'n_eval': len(ious),
    }


# ── Per-sequence tracking ────────────────────────────────────────────────────

def run_sequence(tracker, seq_path):
    """Track one sequence. Returns (boxes, gt, frame_times)."""
    gt = load_groundtruth(os.path.join(seq_path, 'groundtruth.txt'))
    if gt is None:
        print(f'  !! no groundtruth.txt in {seq_path}')
        return None

    # Images are in the color/ sub-folder for all source datasets
    img_dir = os.path.join(seq_path, 'color')
    if not os.path.isdir(img_dir):
        img_dir = os.path.join(seq_path, 'img')

    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    images = sorted(
        os.path.join(img_dir, f) for f in os.listdir(img_dir)
        if f.lower().endswith(valid_exts))
    if not images:
        print(f'  !! no images in {img_dir}')
        return None

    # Tracker expects RGB (BGR2RGB conversion is applied in track())
    frame = cv.cvtColor(cv.imread(images[0]), cv.COLOR_BGR2RGB)
    t0 = time.time()
    tracker.initialize(frame, {'init_bbox': list(gt[0])})
    frame_times = [time.time() - t0]

    boxes = [list(gt[0])]
    for img_path in images[1:]:
        frame = cv.cvtColor(cv.imread(img_path), cv.COLOR_BGR2RGB)
        t0 = time.time()
        out = tracker.track(frame)
        frame_times.append(time.time() - t0)
        boxes.append([float(s) for s in out['target_bbox']])
    return boxes, gt, frame_times


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='DiDi distractor dataset tracking + evaluation.')
    parser.add_argument('--tracker_name', type=str, default='ostrack')
    parser.add_argument('--tracker_param', type=str,
                        default='vitb_384_mae_ce_32x4_ep300')
    parser.add_argument('--data_dir', type=str,
                        default='E:/gitProjects/data/DiDiData',
                        help='Root of the DiDiData dataset.')
    parser.add_argument('--seq_file', type=str, default=None,
                        help='Text file listing sequence names (one per line).')
    parser.add_argument('--seqs', type=str, default=None,
                        help='Comma-separated sequence names to run (overrides seq_file).')
    parser.add_argument('--limit', type=int, default=None,
                        help='Only run the first N sequences.')
    parser.add_argument('--tag', type=str, default=None,
                        help='Run tag for the output folder.')
    args = parser.parse_args()

    # Load sequence list
    if args.seqs:
        seq_names = [s.strip() for s in args.seqs.split(',') if s.strip()]
    elif args.seq_file:
        with open(args.seq_file, 'r') as f:
            seq_names = [l.strip() for l in f if l.strip()]
    else:
        raise ValueError('Must specify either --seq_file or --seqs')

    seq_paths = {}
    for name in seq_names:
        path = os.path.join(args.data_dir, name)
        if os.path.isdir(path):
            seq_paths[name] = path
        else:
            print(f'WARNING: sequence directory not found: {path}')

    seq_list = [(n, seq_paths[n]) for n in seq_names if n in seq_paths]
    if args.limit:
        seq_list = seq_list[:args.limit]

    print(f'Sequences: {len(seq_list)}, output: output/batch_eval/{args.tag}')

    tracker_obj = Tracker(args.tracker_name, args.tracker_param, 'video')
    params = tracker_obj.get_parameters()
    params.debug = int(os.environ.get('OSTRACK_DEBUG', '0'))
    tracker = tracker_obj.create_tracker(params)

    tag = args.tag or f'dididata_{time.strftime("%m%d_%H%M%S")}'
    out_dir = os.path.join(prj_path, 'output', 'batch_eval', tag)
    boxes_dir = os.path.join(out_dir, 'boxes')
    os.makedirs(boxes_dir, exist_ok=True)

    print(f'\n{"Sequence":<30} {"Frames":>6} {"FPS":>7} {"AUC":>7} {"P@20":>7} {"NPrec":>7} {"mIoU":>7}')
    print('-' * 80)

    rows = []
    total_frames, total_time = 0, 0.0
    for seq_name, seq_path in seq_list:
        result = run_sequence(tracker, seq_path)
        if result is None:
            rows.append({'sequence': seq_name, 'frames': 0,
                         'auc': '', 'prec20': '', 'nprec': '', 'miou': ''})
            continue
        boxes, gt, frame_times = result
        n_frames = len(boxes)
        elapsed = sum(frame_times)
        total_frames += n_frames
        total_time += elapsed
        fps = n_frames / elapsed if elapsed > 0 else 0.0

        int_fmt = any(seq_name.startswith(p) for p in ('GOT-10k', 'LaSOT', 'VOT2022-ST', 'VOT2020-ST'))
        np.savetxt(os.path.join(boxes_dir, f'{seq_name}.txt'),
                   np.array(boxes).astype(int) if int_fmt else np.array(boxes),
                   delimiter=',', fmt='%d' if int_fmt else '%.2f')

        m = evaluate_sequence(boxes, gt)
        if m:
            print(f'{seq_name:<30} {n_frames:>6} {fps:>7.1f} '
                  f'{m["auc"]:>7.4f} {m["prec20"]:>7.4f} {m["nprec"]:>7.4f} {m["miou"]:>7.4f}')
            rows.append({'sequence': seq_name, 'frames': n_frames,
                         'auc': round(m['auc'], 4), 'prec20': round(m['prec20'], 4),
                         'nprec': round(m['nprec'], 4), 'miou': round(m['miou'], 4)})
        else:
            print(f'{seq_name:<30} {n_frames:>6} {fps:>7.1f} {"-":>7} {"-":>7} {"-":>7} {"-":>7}')
            rows.append({'sequence': seq_name, 'frames': n_frames,
                         'auc': '', 'prec20': '', 'nprec': '', 'miou': ''})

    # ── save metrics.csv ──
    csv_path = os.path.join(out_dir, 'metrics.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['sequence', 'frames', 'auc', 'prec20', 'nprec', 'miou'])
        writer.writeheader()
        writer.writerows(rows)

    # ── averages + summary ──
    lines = [
        f'Run tag       : {tag}',
        f'Dataset       : DiDiData ({os.path.basename(args.seq_file or args.seqs or "custom")})',
        f'Sequences     : {len([r for r in rows if r["frames"] > 0])}/{len(rows)} run',
        f'Total frames  : {total_frames}, avg FPS: {total_frames / total_time:.1f}' if total_time > 0 else '',
    ]
    ok = [r for r in rows if r['auc'] != '']
    if ok:
        all_auc = [r['auc'] for r in ok]
        all_p20 = [r['prec20'] for r in ok]
        all_np = [r['nprec'] for r in ok]
        all_miou = [r['miou'] for r in ok]
        lines.append(f'Macro (per-seq mean): AUC={np.mean(all_auc):.4f}, '
                     f'P@20={np.mean(all_p20):.4f}, '
                     f'NP={np.mean(all_np):.4f}, mIoU={np.mean(all_miou):.4f}')
    summary = '\n'.join(l for l in lines if l)
    with open(os.path.join(out_dir, 'summary.txt'), 'w', encoding='utf-8') as f:
        f.write(summary + '\n')

    print('-' * 80)
    print(summary)
    print(f'\nSaved: {csv_path}')


if __name__ == '__main__':
    main()
