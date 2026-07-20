import os
import numpy as np

def compute_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    box1_x1, box1_y1, box1_x2, box1_y2 = x1, y1, x1 + w1, y1 + h1
    box2_x1, box2_y1, box2_x2, box2_y2 = x2, y2, x2 + w2, y2 + h2
    inter_x1 = max(box1_x1, box2_x1)
    inter_y1 = max(box1_y1, box2_y1)
    inter_x2 = min(box1_x2, box2_x2)
    inter_y2 = min(box2_y2, box2_y2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

def compute_center_error(box1, box2):
    cx1, cy1 = box1[0] + box1[2] / 2, box1[1] + box1[3] / 2
    cx2, cy2 = box2[0] + box2[2] / 2, box2[1] + box2[3] / 2
    return np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)

# BlackSwan1 paths
gt_file = 'E:/gitProjects/OSTrack/data/Multi_Modal_RGBT_dataset_CSR/BlackSwan1/groundTruth_v.txt'
result_file = 'E:/gitProjects/OSTrack/data/Multi_Modal_RGBT_dataset_CSR/BlackSwan1/track_results/v.txt'

# Load ground truth [x1,y1,x2,y2] -> [x,y,w,h]
with open(gt_file, 'r') as f:
    gt_lines = f.readlines()
gt = []
for line in gt_lines:
    line = line.strip()
    if not line:
        continue
    parts = list(map(float, line.split()))
    if len(parts) >= 4:
        x1, y1, x2, y2 = parts
        x, y = int(x1), int(y1)
        w, h = int(x2 - x1), int(y2 - y1)
        gt.append([x, y, w, h])

# Load tracking result [x,y,w,h]
with open(result_file, 'r') as f:
    result_lines = f.readlines()
tracked = []
for line in result_lines:
    line = line.strip()
    if not line:
        continue
    parts = list(map(float, line.split()))
    if len(parts) >= 4:
        tracked.append([int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])])

n = min(len(tracked), len(gt))
print(f'Sequence: BlackSwan1')
print(f'Ground truth frames: {len(gt)}, Tracked frames: {len(tracked)}, Evaluated: {n}')

ious = [compute_iou(t, g) for t, g in zip(tracked[:n], gt[:n])]
errors = [compute_center_error(t, g) for t, g in zip(tracked[:n], gt[:n])]

print(f'Success (IoU): {np.mean(ious):.3f}')
print(f'Precision (center error): {np.mean(errors):.2f} pixels')
print(f'IoU > 0.5: {sum(1 for i in ious if i > 0.5)}/{n} ({100*sum(1 for i in ious if i > 0.5)/n:.1f}%)')
