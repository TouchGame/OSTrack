"""Batch-run the tracker over all dataset sequences and save metrics.

Loads the network ONCE, then for every sequence: initialize from the first
ground-truth box, track all frames (no GUI, no debug dumps), compute
Success (mean IoU) and CLE (mean center error, px) — same definitions as
tracking/eval_tracker.py so numbers stay comparable — and compare against
each sequence's track_results/origin_<mod>.txt baseline when present.

Outputs under output/batch_eval/<tag>/:
    boxes/<seq>_<mod>.txt   tracked boxes (x y w h, one per frame)
    metrics.csv             per-sequence metrics + origin comparison
    summary.txt             averages and run configuration

Usage:
    python tracking/run_all_sequences.py                       # all sequences, modality v
    python tracking/run_all_sequences.py --seqs BlackCar,Jogging
    python tracking/run_all_sequences.py --limit 3             # quick test
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
from eval_tracker import compute_iou, compute_center_error, load_groundtruth, load_tracking_result


def list_sequences(data_dir, modality):
    seqs = []
    for name in sorted(os.listdir(data_dir)):
        path = os.path.join(data_dir, name)
        if os.path.isdir(path) and os.path.isdir(os.path.join(path, modality)):
            seqs.append((name, path))
    return seqs


def run_sequence(tracker, seq_path, modality):
    """Track one sequence. Returns (boxes, num_frames, elapsed_seconds) or None."""
    folder = os.path.join(seq_path, modality)
    gt_file = os.path.join(seq_path, f'groundTruth_{modality}.txt')
    gt = load_groundtruth(gt_file)
    if gt is None or len(gt) == 0:
        print(f'  !! no ground truth: {gt_file}')
        return None

    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
    images = sorted(os.path.join(folder, f) for f in os.listdir(folder)
                    if f.lower().endswith(valid_exts))
    if len(images) == 0:
        print(f'  !! no images in {folder}')
        return None

    init_box = gt[0]
    frame = cv.imread(images[0])
    t0 = time.time()
    tracker.initialize(frame, {'init_bbox': init_box})

    boxes = [list(init_box)]
    for img_path in images[1:]:
        frame = cv.imread(img_path)
        if frame is None:
            continue
        out = tracker.track(frame)
        boxes.append([int(s) for s in out['target_bbox']])
    elapsed = time.time() - t0
    return boxes, gt, len(images), elapsed


def evaluate(boxes, gt):
    """Same protocol as eval_tracker.py: mean IoU + mean center error."""
    n = min(len(boxes), len(gt))
    if n == 0:
        return None
    ious = [compute_iou(b, g) for b, g in zip(boxes[:n], gt[:n])]
    errs = [compute_center_error(b, g) for b, g in zip(boxes[:n], gt[:n])]
    return {'success': float(np.mean(ious)), 'cle': float(np.mean(errs)), 'n': n}


def main():
    parser = argparse.ArgumentParser(description='Batch tracking + evaluation over all sequences.')
    parser.add_argument('--tracker_name', type=str, default='ostrack')
    parser.add_argument('--tracker_param', type=str, default='vitb_384_mae_ce_32x4_ep300')
    parser.add_argument('--data_dir', type=str,
                        default='E:/gitProjects/OSTrack/data/Multi_Modal_RGBT_dataset_CSR')
    parser.add_argument('--modality', type=str, default='v', choices=['v', 'i'])
    parser.add_argument('--seqs', type=str, default=None,
                        help='Comma-separated sequence names (default: all).')
    parser.add_argument('--limit', type=int, default=None,
                        help='Only run the first N sequences (quick test).')
    parser.add_argument('--tag', type=str, default=None,
                        help='Run tag for the output folder (default: auto from config flags).')
    args = parser.parse_args()

    tracker_obj = Tracker(args.tracker_name, args.tracker_param, 'video')
    params = tracker_obj.get_parameters()
    params.debug = 0
    tracker = tracker_obj.create_tracker(params)

    uni = getattr(params.cfg.MODEL.BACKBONE, 'UNIDIRECTIONAL', False)
    ref_pool = getattr(params.cfg.MODEL.BACKBONE, 'REF_POOL', False)
    tag = args.tag or f'uni{int(uni)}_refpool{int(ref_pool)}_{args.modality}_{time.strftime("%m%d_%H%M%S")}'

    out_dir = os.path.join(prj_path, 'output', 'batch_eval', tag)
    boxes_dir = os.path.join(out_dir, 'boxes')
    os.makedirs(boxes_dir, exist_ok=True)

    sequences = list_sequences(args.data_dir, args.modality)
    if args.seqs:
        wanted = {s.strip() for s in args.seqs.split(',')}
        sequences = [s for s in sequences if s[0] in wanted]
        missing = wanted - {s[0] for s in sequences}
        if missing:
            print(f'WARNING: sequences not found: {sorted(missing)}')
    if args.limit:
        sequences = sequences[:args.limit]

    print(f'Run tag: {tag}  (UNI={uni}, REF_POOL={ref_pool}, modality={args.modality})')
    print(f'Sequences: {len(sequences)}, output: {out_dir}\n')
    print(f'{"Sequence":<16} {"Frames":>6} {"FPS":>7} {"Succ":>7} {"CLE":>7} {"oSucc":>7} {"oCLE":>7}')
    print('-' * 64)

    rows = []
    total_frames, total_time = 0, 0.0
    for seq_name, seq_path in sequences:
        result = run_sequence(tracker, seq_path, args.modality)
        if result is None:
            rows.append({'sequence': seq_name, 'frames': 0, 'success': '', 'cle': '',
                         'origin_success': '', 'origin_cle': ''})
            continue
        boxes, gt, n_frames, elapsed = result
        m = evaluate(boxes, gt)

        # per-sequence origin baseline, if the old workflow produced one
        origin_file = os.path.join(seq_path, 'track_results', f'origin_{args.modality}.txt')
        om = None
        origin_boxes = load_tracking_result(origin_file)
        if origin_boxes:
            om = evaluate(origin_boxes, gt)

        np.savetxt(os.path.join(boxes_dir, f'{seq_name}_{args.modality}.txt'),
                   np.array(boxes, dtype=int), delimiter='\t', fmt='%d')

        fps = n_frames / elapsed if elapsed > 0 else 0.0
        total_frames += n_frames
        total_time += elapsed
        o_succ = f'{om["success"]:.3f}' if om else 'N/A'
        o_cle = f'{om["cle"]:.2f}' if om else 'N/A'
        print(f'{seq_name:<16} {n_frames:>6} {fps:>7.1f} {m["success"]:>7.3f} {m["cle"]:>7.2f} '
              f'{o_succ:>7} {o_cle:>7}')
        if len(boxes) != len(gt):
            print(f'  WARNING: {seq_name}: {len(boxes)} tracked vs {len(gt)} gt frames')

        rows.append({'sequence': seq_name, 'frames': n_frames,
                     'success': round(m['success'], 4), 'cle': round(m['cle'], 3),
                     'origin_success': round(om['success'], 4) if om else '',
                     'origin_cle': round(om['cle'], 3) if om else ''})

    # ── save metrics.csv ──
    csv_path = os.path.join(out_dir, 'metrics.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['sequence', 'frames', 'success', 'cle',
                                               'origin_success', 'origin_cle'])
        writer.writeheader()
        writer.writerows(rows)

    # ── averages + summary ──
    ok = [r for r in rows if r['success'] != '']
    with_origin = [r for r in ok if r['origin_success'] != '']
    lines = [
        f'Run tag       : {tag}',
        f'Config        : UNIDIRECTIONAL={uni}, REF_POOL={ref_pool}, modality={args.modality}',
        f'Sequences     : {len(ok)}/{len(rows)} evaluated',
        f'Total frames  : {total_frames}, avg FPS: {total_frames / total_time:.1f}' if total_time > 0 else '',
    ]
    if ok:
        avg_s = np.mean([r['success'] for r in ok])
        avg_c = np.mean([r['cle'] for r in ok])
        lines.append(f'Average       : Success={avg_s:.4f}, CLE={avg_c:.3f}px')
    if with_origin:
        avg_s = np.mean([r['success'] for r in with_origin])
        avg_c = np.mean([r['cle'] for r in with_origin])
        avg_os = np.mean([r['origin_success'] for r in with_origin])
        avg_oc = np.mean([r['origin_cle'] for r in with_origin])
        lines.append(f'vs Origin ({len(with_origin)} seqs): '
                     f'Success {avg_os:.4f} -> {avg_s:.4f} ({(avg_s - avg_os) / avg_os * 100:+.2f}%), '
                     f'CLE {avg_oc:.3f} -> {avg_c:.3f}px ({(avg_c - avg_oc) / avg_oc * 100:+.2f}%, lower is better)')

    summary = '\n'.join(l for l in lines if l)
    with open(os.path.join(out_dir, 'summary.txt'), 'w', encoding='utf-8') as f:
        f.write(summary + '\n')

    print('-' * 64)
    print(summary)
    print(f'\nSaved: {csv_path}')


if __name__ == '__main__':
    main()
