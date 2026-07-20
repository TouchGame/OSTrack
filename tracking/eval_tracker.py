"""Evaluate tracking results against ground truth."""
import os
import numpy as np
from pathlib import Path


def load_groundtruth(gt_file):
    """Load ground truth from file. Format: x1 y1 x2 y2 per line, convert to x y w h."""
    if not os.path.isfile(gt_file):
        return None
    with open(gt_file, 'r') as f:
        lines = f.readlines()
    boxes = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = list(map(float, line.split()))
        if len(parts) >= 4:
            x1, y1, x2, y2 = parts[0], parts[1], parts[2], parts[3]
            # Convert [x1, y1, x2, y2] to [x, y, w, h]
            x, y = int(x1), int(y1)
            w, h = int(x2 - x1), int(y2 - y1)
            boxes.append([x, y, w, h])
    return boxes


def load_tracking_result(result_file):
    """Load tracking result. Format: x y w h per line."""
    if not os.path.isfile(result_file):
        return None
    with open(result_file, 'r') as f:
        lines = f.readlines()
    boxes = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = list(map(float, line.split()))
        if len(parts) >= 4:
            boxes.append([parts[0], parts[1], parts[2], parts[3]])
    return boxes


def compute_iou(box1, box2):
    """Compute IoU between two boxes [x, y, w, h]."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Convert to [x1, y1, x2, y2]
    box1_x1, box1_y1, box1_x2, box1_y2 = x1, y1, x1 + w1, y1 + h1
    box2_x1, box2_y1, box2_x2, box2_y2 = x2, y2, x2 + w2, y2 + h2

    # Intersection
    inter_x1 = max(box1_x1, box2_x1)
    inter_y1 = max(box1_y1, box2_y1)
    inter_x2 = min(box1_x2, box2_x2)
    inter_y2 = min(box1_y2, box2_y2)

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def compute_center_error(box1, box2):
    """Compute center location error between two boxes [x, y, w, h]."""
    cx1, cy1 = box1[0] + box1[2] / 2, box1[1] + box1[3] / 2
    cx2, cy2 = box2[0] + box2[2] / 2, box2[1] + box2[3] / 2
    return np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)


def evaluate_sequence(tracked_file, gt_file):
    """Evaluate a single sequence."""
    tracked = load_tracking_result(tracked_file)
    gt = load_groundtruth(gt_file)

    if tracked is None or gt is None:
        print(f"Missing files: tracked={tracked_file}, gt={gt_file}")
        return None

    n = min(len(tracked), len(gt))
    if n == 0:
        return None

    tracked = tracked[:n]
    gt = gt[:n]

    # Compute Success (IoU)
    ious = [compute_iou(t, g) for t, g in zip(tracked, gt)]
    success = np.mean(ious)

    # Compute Precision (center error)
    errors = [compute_center_error(t, g) for t, g in zip(tracked, gt)]
    precision = np.mean(errors)

    return {
        'success': success,
        'precision': precision,
        'ious': ious,
        'errors': errors,
        'num_frames': n
    }


def evaluate_dataset(data_dir, result_subdir='track_results'):
    """Evaluate all sequences in a dataset, compare with origin results."""
    result_dir = os.path.join(data_dir, result_subdir)

    # Check if data_dir is a single sequence directory (contains groundTruth_v.txt directly)
    single_gt_file = os.path.join(data_dir, 'groundTruth_v.txt')
    single_result_file = os.path.join(data_dir, result_subdir, 'v.txt')
    single_origin_file = os.path.join(data_dir, result_subdir, 'origin_v.txt')

    if os.path.exists(single_gt_file) and os.path.exists(single_result_file):
        # Single sequence mode
        seq_name = os.path.basename(data_dir)
        result = evaluate_sequence(single_result_file, single_gt_file)
        origin_result = evaluate_sequence(single_origin_file, single_gt_file) if os.path.exists(single_origin_file) else None

        print(f"\n{'='*60}")
        print(f"Sequence: {seq_name}")
        print(f"{'='*60}")

        if origin_result:
            print(f"  Origin  -> Success: {origin_result['success']:.3f}, Precision: {origin_result['precision']:.2f}")
        if result:
            print(f"  Current -> Success: {result['success']:.3f}, Precision: {result['precision']:.2f}")
        if result and origin_result:
            succ_imp = (result['success'] - origin_result['success']) / origin_result['success'] * 100
            prec_imp = (result['precision'] - origin_result['precision']) / origin_result['precision'] * 100
            print(f"  Improve -> Success: {succ_imp:+.2f}%, Precision: {prec_imp:+.2f}%")
        return

    if not os.path.exists(result_dir):
        print(f"Result directory not found: {result_dir}")
        return

    sequences = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    sequences = [s for s in sequences if not s.startswith('.') and not s.startswith('__')]

    all_success = []
    all_precision = []
    all_origin_success = []
    all_origin_precision = []

    print(f"\n{'='*60}")
    print(f"{'Sequence':<20} {'Origin':<20} {'Current':<20}")
    print(f"{'='*60}")
    print(f"{'Sequence':<20} {'Succ':<10} {'Prec':<10} {'Succ':<10} {'Prec':<10}")
    print("-" * 60)

    for seq in sorted(sequences):
        seq_dir = os.path.join(data_dir, seq)
        result_file = os.path.join(result_dir, f'{seq}.txt')
        origin_file = os.path.join(result_dir, 'origin_v.txt')
        gt_file = os.path.join(seq_dir, 'groundTruth_v.txt')

        if not os.path.exists(result_file):
            print(f"{seq:<20} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10}")
            continue

        result = evaluate_sequence(result_file, gt_file)
        origin_result = evaluate_sequence(origin_file, gt_file) if os.path.exists(origin_file) else None

        if result is None:
            print(f"{seq:<20} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10}")
            continue

        if origin_result:
            print(f"{seq:<20} {origin_result['success']:<10.3f} {origin_result['precision']:<10.2f} {result['success']:<10.3f} {result['precision']:<10.2f}")
            all_success.append(result['success'])
            all_precision.append(result['precision'])
            all_origin_success.append(origin_result['success'])
            all_origin_precision.append(origin_result['precision'])
        else:
            print(f"{seq:<20} {'N/A':<10} {'N/A':<10} {result['success']:<10.3f} {result['precision']:<10.2f}")
            all_success.append(result['success'])
            all_precision.append(result['precision'])

    print("-" * 60)

    if all_origin_success:
        avg_curr_succ = np.mean(all_success)
        avg_orig_succ = np.mean(all_origin_success)
        avg_curr_prec = np.mean(all_precision)
        avg_orig_prec = np.mean(all_origin_precision)
        print(f"{'Average':<20} {avg_orig_succ:<10.3f} {avg_orig_prec:<10.2f} {avg_curr_succ:<10.3f} {avg_curr_prec:<10.2f}")
        succ_imp = (avg_curr_succ - avg_orig_succ) / avg_orig_succ * 100
        prec_imp = (avg_curr_prec - avg_orig_prec) / avg_orig_prec * 100
        print(f"{'Improve':<20} {'':>10} {'':>10} {succ_imp:>+10.2f}% {prec_imp:>+10.2f}%")
    elif all_success:
        avg_curr_succ = np.mean(all_success)
        avg_curr_prec = np.mean(all_precision)
        print(f"{'Average':<20} {'N/A':<10} {'N/A':<10} {avg_curr_succ:<10.3f} {avg_curr_prec:<10.2f}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate tracking results')
    parser.add_argument('--data_dir', type=str,
                        default='E:/gitProjects/OSTrack/data/Multi_Modal_RGBT_dataset_CSR',
                        help='Path to dataset directory')
    parser.add_argument('--result_dir', type=str,
                        default='track_results',
                        help='Subdirectory containing tracking results')
    args = parser.parse_args()

    evaluate_dataset(args.data_dir, args.result_dir)
