"""Batch-run the tracker on LaSOT test sequences and compute OPE metrics locally.

Loads the network ONCE, then for every sequence: initialize from the first
ground-truth box and track all frames (no GUI, no debug dumps).

LaSOT layout (this machine, note the doubled category level):
    <root>/<category>/<category>/<seq>/img/*.jpg
    <root>/<category>/<category>/<seq>/groundtruth.txt   (x,y,w,h per frame)

Metrics (official LaSOT OPE protocol, all frames evaluated):
    Success AUC : mean success rate over 21 IoU thresholds [0:0.05:1]
    Precision   : fraction of frames with center error <= 20 px
    Norm. Prec. : AUC over 51 thresholds [0:0.01:0.5] of the center error
                  normalized by the ground-truth box size

Outputs under output/batch_eval/<tag>/:
    boxes/<seq>.txt     tracked boxes (x,y,w,h, one per frame, comma-sep)
    metrics.csv         per-sequence metrics
    summary.txt         averages and run configuration

Usage:
    python tracking/run_lasot.py --seq_file testing_set_quick.txt --tag lasot_quick_origin
    python tracking/run_lasot.py --seqs basketball-1 --tag quick_smoke
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


def parse_box_line(line):
    parts = line.replace(',', ' ').split()
    return [float(p) for p in parts[:4]]


def load_groundtruth(path):
    if not os.path.isfile(path):
        return None
    boxes = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                boxes.append(parse_box_line(line))
    return boxes if boxes else None


def seq_dir_of(root, name):
    """basketball-1 -> <root>/basketball/basketball/basketball-1"""
    cat = name.split('-')[0]
    return os.path.join(root, cat, cat, name)


def compute_iou(a, b):
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    ix = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    iy = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def run_sequence(tracker, seq_path):
    """Track one sequence. Returns (boxes, gt, frame_times) or None."""
    gt = load_groundtruth(os.path.join(seq_path, 'groundtruth.txt'))
    if gt is None:
        print(f'  !! no groundtruth.txt in {seq_path}')
        return None

    img_dir = os.path.join(seq_path, 'img')
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    images = sorted(os.path.join(img_dir, f) for f in os.listdir(img_dir)
                    if f.lower().endswith(valid_exts))
    if len(images) == 0:
        print(f'  !! no images in {img_dir}')
        return None

    # NOTE: tracker expects RGB (official pytracking _read_image does
    # cv.imread + BGR2RGB); feeding raw BGR silently costs ~6pp AO
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


def evaluate(boxes, gt):
    """LaSOT OPE: per-frame IoU / center error / normalized center error."""
    n = min(len(boxes), len(gt))
    ious, cerrs, nerrs = [], [], []
    for i in range(n):
        b, g = boxes[i], gt[i]
        if g[2] <= 0 or g[3] <= 0:      # degenerate GT (full occlusion label)
            continue
        ious.append(compute_iou(b, g))
        bcx, bcy = b[0] + b[2] / 2.0, b[1] + b[3] / 2.0
        gcx, gcy = g[0] + g[2] / 2.0, g[1] + g[3] / 2.0
        cerrs.append(np.hypot(bcx - gcx, bcy - gcy))
        nerrs.append(np.hypot((bcx - gcx) / g[2], (bcy - gcy) / g[3]))
    if not ious:
        return None
    ious, cerrs, nerrs = np.array(ious), np.array(cerrs), np.array(nerrs)
    iou_thr = np.linspace(0, 1, 21)
    nrm_thr = np.linspace(0, 0.5, 51)
    return {'auc': float(np.mean([np.mean(ious > t) for t in iou_thr])),
            'prec20': float(np.mean(cerrs <= 20)),
            'nprec': float(np.mean([np.mean(nerrs <= t) for t in nrm_thr])),
            'miou': float(np.mean(ious)),
            'ious': ious, 'cerrs': cerrs, 'nerrs': nerrs}


def main():
    parser = argparse.ArgumentParser(description='LaSOT batch tracking + evaluation.')
    parser.add_argument('--tracker_name', type=str, default='ostrack')
    parser.add_argument('--tracker_param', type=str, default='vitb_384_mae_ce_32x4_ep300')
    parser.add_argument('--data_dir', type=str,
                        default='E:/gitProjects/data/lasot_train',
                        help='LaSOT root (doubled category level).')
    parser.add_argument('--seq_file', type=str, default='testing_set_quick.txt',
                        help='Sequence list file (relative to data_dir or absolute).')
    parser.add_argument('--seqs', type=str, default=None,
                        help='Comma-separated sequence names (overrides seq_file).')
    parser.add_argument('--limit', type=int, default=None,
                        help='Only run the first N sequences (quick test).')
    parser.add_argument('--tag', type=str, default=None,
                        help='Run tag for the output folder.')
    args = parser.parse_args()

    if args.seqs:
        names = [s.strip() for s in args.seqs.split(',') if s.strip()]
    else:
        list_path = args.seq_file if os.path.isabs(args.seq_file) \
            else os.path.join(args.data_dir, args.seq_file)
        if not os.path.isfile(list_path):
            sys.exit(f'ERROR: sequence list not found: {list_path}')
        with open(list_path, 'r') as f:
            names = [l.strip() for l in f if l.strip()]
    if args.limit:
        names = names[:args.limit]

    sequences = []
    for n in names:
        p = seq_dir_of(args.data_dir, n)
        if os.path.isdir(p):
            sequences.append((n, p))
        else:
            print(f'WARNING: sequence dir not found, skipped: {p}')

    tracker_obj = Tracker(args.tracker_name, args.tracker_param, 'video')
    params = tracker_obj.get_parameters()
    params.debug = 0
    tracker = tracker_obj.create_tracker(params)

    tag = args.tag or f'lasot_{time.strftime("%m%d_%H%M%S")}'
    out_dir = os.path.join(prj_path, 'output', 'batch_eval', tag)
    boxes_dir = os.path.join(out_dir, 'boxes')
    os.makedirs(boxes_dir, exist_ok=True)

    print(f'Run tag: {tag}  (LaSOT, {len(sequences)} sequences)')
    print(f'Output: {out_dir}\n')
    print(f'{"Sequence":<18} {"Frames":>6} {"FPS":>7} {"AUC":>7} {"P@20":>7} {"Pnorm":>7} {"mIoU":>7}')
    print('-' * 66)

    rows = []
    all_ious, all_cerrs, all_nerrs = [], [], []
    total_frames, total_time = 0, 0.0
    for seq_name, seq_path in sequences:
        result = run_sequence(tracker, seq_path)
        if result is None:
            rows.append({'sequence': seq_name, 'frames': 0,
                         'auc': '', 'prec20': '', 'nprec': '', 'miou': ''})
            continue
        boxes, gt, frame_times = result
        n_frames, elapsed = len(boxes), sum(frame_times)
        total_frames += n_frames
        total_time += elapsed
        fps = n_frames / elapsed if elapsed > 0 else 0.0

        np.savetxt(os.path.join(boxes_dir, f'{seq_name}.txt'),
                   np.array(boxes).astype(int), delimiter=',', fmt='%d')

        m = evaluate(boxes, gt)
        if m:
            all_ious.append(m['ious'])
            all_cerrs.append(m['cerrs'])
            all_nerrs.append(m['nerrs'])
            print(f'{seq_name:<18} {n_frames:>6} {fps:>7.1f} '
                  f'{m["auc"]:>7.3f} {m["prec20"]:>7.3f} {m["nprec"]:>7.3f} {m["miou"]:>7.3f}')
            rows.append({'sequence': seq_name, 'frames': n_frames,
                         'auc': round(m['auc'], 4), 'prec20': round(m['prec20'], 4),
                         'nprec': round(m['nprec'], 4), 'miou': round(m['miou'], 4)})
        else:
            print(f'{seq_name:<18} {n_frames:>6} {fps:>7.1f} {"-":>7} {"-":>7} {"-":>7} {"-":>7}')
            rows.append({'sequence': seq_name, 'frames': n_frames,
                         'auc': '', 'prec20': '', 'nprec': '', 'miou': ''})

    # ── save metrics.csv ──
    csv_path = os.path.join(out_dir, 'metrics.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['sequence', 'frames', 'auc',
                                               'prec20', 'nprec', 'miou'])
        writer.writeheader()
        writer.writerows(rows)

    # ── averages + summary ──
    lines = [
        f'Run tag       : {tag}',
        f'Split         : LaSOT ({os.path.basename(args.seq_file) if not args.seqs else "custom"})',
        f'Sequences     : {len([r for r in rows if r["frames"] > 0])}/{len(rows)} run',
        f'Total frames  : {total_frames}, avg FPS: {total_frames / total_time:.1f}' if total_time > 0 else '',
        f'ORIGIN={os.environ.get("OSTRACK_ORIGIN", "0")} '
        f'SIZEGUARD=off(hardcoded) S2_HANN=on(hardcoded)',
    ]
    ok = [r for r in rows if r['auc'] != '']
    if ok and all_ious:
        ious = np.concatenate(all_ious)
        cerrs = np.concatenate(all_cerrs)
        nerrs = np.concatenate(all_nerrs)
        iou_thr = np.linspace(0, 1, 21)
        nrm_thr = np.linspace(0, 0.5, 51)
        lines.append(f'Frame-pooled : AUC={np.mean([np.mean(ious > t) for t in iou_thr]):.4f}, '
                     f'P@20={np.mean(cerrs <= 20):.4f}, '
                     f'Pnorm={np.mean([np.mean(nerrs <= t) for t in nrm_thr]):.4f}, '
                     f'mIoU={np.mean(ious):.4f}')
        lines.append(f'Macro (per-seq mean): AUC={np.mean([r["auc"] for r in ok]):.4f}, '
                     f'P@20={np.mean([r["prec20"] for r in ok]):.4f}, '
                     f'Pnorm={np.mean([r["nprec"] for r in ok]):.4f}, '
                     f'mIoU={np.mean([r["miou"] for r in ok]):.4f}')

    summary = '\n'.join(l for l in lines if l)
    with open(os.path.join(out_dir, 'summary.txt'), 'w', encoding='utf-8') as f:
        f.write(summary + '\n')

    print('-' * 66)
    print(summary)
    print(f'\nSaved: {csv_path}')


if __name__ == '__main__':
    main()
