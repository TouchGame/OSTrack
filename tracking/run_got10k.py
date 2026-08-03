"""Batch-run the tracker on GOT-10k val/test and save metrics or a submission.

Loads the network ONCE, then for every sequence: initialize from the first
ground-truth box and track all frames (no GUI, no debug dumps).

val  split: computes the official GOT-10k metrics locally —
            AO (mean IoU), SR@0.50 and SR@0.75, following the got10k-toolkit
            protocol: frames whose cover label is 0 (target fully invisible)
            are excluded; the first frame counts with IoU 1. Overall AO/SR is
            frame-pooled across sequences (official style), the per-sequence
            macro average is reported alongside.
test split: no local metrics (only the first frame is annotated). Writes the
            official submission layout instead, ready to zip and upload:
                submission/<seq>/<seq>_001.txt   (x,y,w,h per frame, comma-sep)
                submission/<seq>/<seq>_time.txt  (per-frame time in seconds)

Outputs under output/batch_eval/<tag>/:
    boxes/<seq>.txt         tracked boxes (x,y,w,h, one per frame, comma-sep)
    metrics.csv             per-sequence metrics            (val only)
    summary.txt             averages and run configuration  (val only)
    submission/             official upload layout          (test only)

Usage:
    python tracking/run_got10k.py --split val --tag got10k_val_dtv2
    python tracking/run_got10k.py --split val --seqs GOT-10k_Val_000001 --tag quick
    python tracking/run_got10k.py --split test --tag got10k_test_submit
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


def load_label(path):
    """One integer per line (absence.label / cover.label). None if missing."""
    if not os.path.isfile(path):
        return None
    with open(path, 'r') as f:
        vals = [int(float(l.strip())) for l in f if l.strip()]
    return np.array(vals) if vals else None


def list_sequences(split_dir):
    list_file = os.path.join(split_dir, 'list.txt')
    if os.path.isfile(list_file):
        with open(list_file, 'r') as f:
            names = [l.strip() for l in f if l.strip()]
    else:
        names = sorted(n for n in os.listdir(split_dir)
                       if os.path.isdir(os.path.join(split_dir, n)))
    return [(n, os.path.join(split_dir, n)) for n in names
            if os.path.isdir(os.path.join(split_dir, n))]


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

    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    images = sorted(os.path.join(seq_path, f) for f in os.listdir(seq_path)
                    if f.lower().endswith(valid_exts))
    if len(images) == 0:
        print(f'  !! no images in {seq_path}')
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


def evaluate(boxes, gt, cover, absence):
    """got10k-toolkit protocol: IoU per frame, invisible frames excluded."""
    n = min(len(boxes), len(gt))
    ious = np.array([compute_iou(boxes[i], gt[i]) for i in range(n)])
    valid = np.ones(n, dtype=bool)
    if cover is not None and len(cover) >= n:
        valid &= cover[:n] > 0
    elif absence is not None and len(absence) >= n:
        valid &= absence[:n] == 0
    ious = ious[valid]
    if len(ious) == 0:
        return None
    return {'ao': float(np.mean(ious)),
            'sr50': float(np.mean(ious > 0.5)),
            'sr75': float(np.mean(ious > 0.75)),
            'ious': ious}


def main():
    parser = argparse.ArgumentParser(description='GOT-10k batch tracking + evaluation.')
    parser.add_argument('--tracker_name', type=str, default='ostrack')
    parser.add_argument('--tracker_param', type=str, default='vitb_384_mae_ce_32x4_ep300')
    parser.add_argument('--data_dir', type=str,
                        default='E:/gitProjects/OSTrack/data/got10k',
                        help='GOT-10k root containing val/ and/or test/.')
    parser.add_argument('--split', type=str, default='val', choices=['val', 'test'])
    parser.add_argument('--seqs', type=str, default=None,
                        help='Comma-separated sequence names (default: all).')
    parser.add_argument('--limit', type=int, default=None,
                        help='Only run the first N sequences (quick test).')
    parser.add_argument('--tag', type=str, default=None,
                        help='Run tag for the output folder.')
    args = parser.parse_args()

    split_dir = os.path.join(args.data_dir, args.split)
    if not os.path.isdir(split_dir):
        sys.exit(f'ERROR: split directory not found: {split_dir}')

    tracker_obj = Tracker(args.tracker_name, args.tracker_param, 'video')
    params = tracker_obj.get_parameters()
    params.debug = int(os.environ.get('OSTRACK_DEBUG', '0'))
    tracker = tracker_obj.create_tracker(params)

    tag = args.tag or f'got10k_{args.split}_{time.strftime("%m%d_%H%M%S")}'
    out_dir = os.path.join(prj_path, 'output', 'batch_eval', tag)
    boxes_dir = os.path.join(out_dir, 'boxes')
    os.makedirs(boxes_dir, exist_ok=True)
    sub_dir = os.path.join(out_dir, 'submission') if args.split == 'test' else None

    sequences = list_sequences(split_dir)
    if args.seqs:
        wanted = {s.strip() for s in args.seqs.split(',')}
        sequences = [s for s in sequences if s[0] in wanted]
        missing = wanted - {s[0] for s in sequences}
        if missing:
            print(f'WARNING: sequences not found: {sorted(missing)}')
    if args.limit:
        sequences = sequences[:args.limit]

    print(f'Run tag: {tag}  (split={args.split})')
    print(f'Sequences: {len(sequences)}, output: {out_dir}\n')
    print(f'{"Sequence":<22} {"Frames":>6} {"FPS":>7} {"AO":>7} {"SR50":>7} {"SR75":>7}')
    print('-' * 62)

    rows = []
    all_ious = []
    total_frames, total_time = 0, 0.0
    for seq_name, seq_path in sequences:
        result = run_sequence(tracker, seq_path)
        if result is None:
            rows.append({'sequence': seq_name, 'frames': 0,
                         'ao': '', 'sr50': '', 'sr75': ''})
            continue
        boxes, gt, frame_times = result
        n_frames, elapsed = len(boxes), sum(frame_times)
        total_frames += n_frames
        total_time += elapsed
        fps = n_frames / elapsed if elapsed > 0 else 0.0

        np.savetxt(os.path.join(boxes_dir, f'{seq_name}.txt'),
                   np.array(boxes), delimiter=',', fmt='%.4f')
        if sub_dir:
            seq_sub = os.path.join(sub_dir, seq_name)
            os.makedirs(seq_sub, exist_ok=True)
            np.savetxt(os.path.join(seq_sub, f'{seq_name}_001.txt'),
                       np.array(boxes), delimiter=',', fmt='%.4f')
            np.savetxt(os.path.join(seq_sub, f'{seq_name}_time.txt'),
                       np.array(frame_times), fmt='%.6f')

        m = None
        if args.split == 'val':
            cover = load_label(os.path.join(seq_path, 'cover.label'))
            absence = load_label(os.path.join(seq_path, 'absence.label'))
            m = evaluate(boxes, gt, cover, absence)
        if m:
            all_ious.append(m['ious'])
            print(f'{seq_name:<22} {n_frames:>6} {fps:>7.1f} '
                  f'{m["ao"]:>7.3f} {m["sr50"]:>7.3f} {m["sr75"]:>7.3f}')
            rows.append({'sequence': seq_name, 'frames': n_frames,
                         'ao': round(m['ao'], 4), 'sr50': round(m['sr50'], 4),
                         'sr75': round(m['sr75'], 4)})
        else:
            print(f'{seq_name:<22} {n_frames:>6} {fps:>7.1f} {"-":>7} {"-":>7} {"-":>7}')
            rows.append({'sequence': seq_name, 'frames': n_frames,
                         'ao': '', 'sr50': '', 'sr75': ''})

    # ── save metrics.csv ──
    csv_path = os.path.join(out_dir, 'metrics.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['sequence', 'frames', 'ao', 'sr50', 'sr75'])
        writer.writeheader()
        writer.writerows(rows)

    # ── averages + summary ──
    lines = [
        f'Run tag       : {tag}',
        f'Split         : {args.split}',
        f'Sequences     : {len([r for r in rows if r["frames"] > 0])}/{len(rows)} run',
        f'Total frames  : {total_frames}, avg FPS: {total_frames / total_time:.1f}' if total_time > 0 else '',
    ]
    ok = [r for r in rows if r['ao'] != '']
    if ok and all_ious:
        pooled = np.concatenate(all_ious)
        lines.append(f'Official (frame-pooled): AO={np.mean(pooled):.4f}, '
                     f'SR50={np.mean(pooled > 0.5):.4f}, SR75={np.mean(pooled > 0.75):.4f}')
        lines.append(f'Macro (per-seq mean)   : AO={np.mean([r["ao"] for r in ok]):.4f}, '
                     f'SR50={np.mean([r["sr50"] for r in ok]):.4f}, '
                     f'SR75={np.mean([r["sr75"] for r in ok]):.4f}')
    if sub_dir:
        lines.append(f'Submission dir: {sub_dir}')
        lines.append('Zip the folders INSIDE submission/ (not the submission folder itself) '
                     'and upload to the GOT-10k server.')

    summary = '\n'.join(l for l in lines if l)
    with open(os.path.join(out_dir, 'summary.txt'), 'w', encoding='utf-8') as f:
        f.write(summary + '\n')

    print('-' * 62)
    print(summary)
    print(f'\nSaved: {csv_path}')


if __name__ == '__main__':
    main()
