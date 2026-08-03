import os
import sys
import argparse

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

from lib.test.evaluation import Tracker
import cv2 as cv
import numpy as np
import time


def _get_target_patch_heatmap(state, score_map, prev_state, resize_factor, search_size=256, stride=16):
    """Compute the heatmap value at the target's patch location.

    Args:
        state: predicted bbox in original image [x, y, w, h]
        score_map: heatmap tensor, shape [1, 1, feat_sz, feat_sz]
        prev_state: previous frame's bbox [x, y, w, h] (used as crop center)
        resize_factor: factor from crop -> resized search region
        search_size: search region size (default 256)
        stride: backbone stride (default 16)

    Returns:
        heat_val: the heatmap value at target's patch
        patch_info: dict with intermediate info (tcx, tcy, px, py)
    """
    # Target center in original image
    tcx_orig = state[0] + state[2] / 2.0
    tcy_orig = state[1] + state[3] / 2.0

    # Crop center in original image (= previous frame center)
    prev_cx = prev_state[0] + prev_state[2] / 2.0
    prev_cy = prev_state[1] + prev_state[3] / 2.0

    # Target center in the resized [0, search_size] crop coords
    # crop was centered at prev_center, resized by resize_factor
    crop_center_in_resized = search_size / 2.0
    tcx_in_resized = crop_center_in_resized + (tcx_orig - prev_cx) * resize_factor
    tcy_in_resized = crop_center_in_resized + (tcy_orig - prev_cy) * resize_factor

    # Normalize to [0, 1]
    tcx_norm = np.clip(tcx_in_resized, 0, search_size) / search_size
    tcy_norm = np.clip(tcy_in_resized, 0, search_size) / search_size

    feat_sz = search_size // stride
    # Patch index in heatmap
    px = int(np.clip(tcx_norm * feat_sz, 0, feat_sz - 1))
    py = int(np.clip(tcy_norm * feat_sz, 0, feat_sz - 1))

    # The box center comes from offset regression and often lands in a cell
    # adjacent to the argmax of a sharp peak; reading that single cell shows
    # the peak's skirt (misleadingly low). Report the peak the box sits on:
    # max over the 3x3 neighbourhood, and snap px/py to that cell.
    score_np = score_map.squeeze().cpu().numpy()
    y0, y1 = max(py - 1, 0), min(py + 2, feat_sz)
    x0, x1 = max(px - 1, 0), min(px + 2, feat_sz)
    win = score_np[y0:y1, x0:x1]
    dy, dx = np.unravel_index(win.argmax(), win.shape)
    py, px = y0 + int(dy), x0 + int(dx)
    heat_val = score_np[py, px]

    patch_info = {
        'tcx': tcx_in_resized,
        'tcy': tcy_in_resized,
        'px': px,
        'py': py,
    }
    return float(heat_val), patch_info


def draw_heatmap_comparison(search_crop, raw_np, resp_np, state, prev_state, resize_factor,
                            stride=16, search_size=256, hann_window=None, gt_box=None):
    """Draw search crop with bbox, peak location, and side-by-side heatmaps.

    Layout:
    Top-left: search crop with bboxes
    Top-right: raw heatmap
    Bottom-left: Hann window heatmap
    Bottom-right: boosted heatmap (resp)
    """
    feat_sz = search_size // stride

    # Target center in search crop coords
    tcx_orig = state[0] + state[2] / 2.0
    tcy_orig = state[1] + state[3] / 2.0
    prev_cx = prev_state[0] + prev_state[2] / 2.0
    prev_cy = prev_state[1] + prev_state[3] / 2.0
    crop_center_in_resized = search_size / 2.0
    tcx_in_resized = crop_center_in_resized + (tcx_orig - prev_cx) * resize_factor
    tcy_in_resized = crop_center_in_resized + (tcy_orig - prev_cy) * resize_factor

    # Target center pixel in search crop
    tcx_pixel = int(np.clip(tcx_in_resized, 0, search_size - 1))
    tcy_pixel = int(np.clip(tcy_in_resized, 0, search_size - 1))

    # Peak locations
    raw_peak_idx = raw_np.argmax()
    raw_peak_py, raw_peak_px = raw_peak_idx // feat_sz, raw_peak_idx % feat_sz
    raw_peak_x = int(np.clip((raw_peak_px + 0.5) * stride, 0, search_size - 1))
    raw_peak_y = int(np.clip((raw_peak_py + 0.5) * stride, 0, search_size - 1))

    resp_peak_idx = resp_np.argmax()
    resp_peak_py, resp_peak_px = resp_peak_idx // feat_sz, resp_peak_idx % feat_sz
    resp_peak_x = int(np.clip((resp_peak_px + 0.5) * stride, 0, search_size - 1))
    resp_peak_y = int(np.clip((resp_peak_py + 0.5) * stride, 0, search_size - 1))

    # Draw on search crop (same size as search_size)
    disp = search_crop.copy()
    # Draw predicted bbox: state is in ORIGINAL image coords, so map it into
    # the resized crop (center via tcx/tcy_in_resized, size scaled by resize_factor)
    bw = state[2] * resize_factor
    bh = state[3] * resize_factor
    x1 = int(np.clip(tcx_in_resized - bw / 2.0, 0, search_size - 1))
    y1 = int(np.clip(tcy_in_resized - bh / 2.0, 0, search_size - 1))
    x2 = int(np.clip(tcx_in_resized + bw / 2.0, 0, search_size - 1))
    y2 = int(np.clip(tcy_in_resized + bh / 2.0, 0, search_size - 1))
    cv.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # Blue rectangle = ground-truth box, mapped into the crop with the same
    # geometry as the predicted box. Drawn plainly; occluded/interpolated
    # ghost frames are not special-cased.
    if gt_box is not None:
        gcx = gt_box[0] + gt_box[2] / 2.0
        gcy = gt_box[1] + gt_box[3] / 2.0
        gcx_in_resized = crop_center_in_resized + (gcx - prev_cx) * resize_factor
        gcy_in_resized = crop_center_in_resized + (gcy - prev_cy) * resize_factor
        gbw = gt_box[2] * resize_factor
        gbh = gt_box[3] * resize_factor
        gx1 = int(np.clip(gcx_in_resized - gbw / 2.0, 0, search_size - 1))
        gy1 = int(np.clip(gcy_in_resized - gbh / 2.0, 0, search_size - 1))
        gx2 = int(np.clip(gcx_in_resized + gbw / 2.0, 0, search_size - 1))
        gy2 = int(np.clip(gcy_in_resized + gbh / 2.0, 0, search_size - 1))
        cv.rectangle(disp, (gx1, gy1), (gx2, gy2), (255, 0, 0), 2)
    # Green dot = target center
    cv.circle(disp, (tcx_pixel, tcy_pixel), 6, (0, 255, 0), -1)
    # Red dot = raw heatmap peak
    cv.circle(disp, (raw_peak_x, raw_peak_y), 8, (0, 0, 255), -1)
    # Yellow dot = peak (same if raw==resp)
    cv.circle(disp, (resp_peak_x, resp_peak_y), 8, (0, 255, 255), -1)

    # Resize heatmaps to search_size for visualization
    raw_hm = (raw_np / raw_np.max() * 255).astype(np.uint8)
    raw_hm_color = cv.applyColorMap(raw_hm, cv.COLORMAP_JET)
    raw_hm_color = cv.resize(raw_hm_color, (search_size, search_size))

    resp_hm = (resp_np / resp_np.max() * 255).astype(np.uint8)
    resp_hm_color = cv.applyColorMap(resp_hm, cv.COLORMAP_JET)
    resp_hm_color = cv.resize(resp_hm_color, (search_size, search_size))

    # Hann-modulated heatmap (bottom-left): raw * Hann
    if hann_window is not None:
        hann_np = hann_window.squeeze()
        if hasattr(hann_np, 'cpu'):
            hann_np = hann_np.cpu().numpy()
        # Hann-modulated = raw score map multiplied by Hann window (before boosting)
        hann_modulated = raw_np * hann_np
        hann_mod_max = hann_modulated.max() if hann_modulated.max() > 0 else 1.0
        hann_hm = (hann_modulated / hann_mod_max * 255).astype(np.uint8)
        hann_hm_color = cv.applyColorMap(hann_hm, cv.COLORMAP_JET)
        hann_hm_color = cv.resize(hann_hm_color, (search_size, search_size))
    else:
        hann_hm_color = np.zeros((search_size, search_size, 3), np.uint8)
        cv.putText(hann_hm_color, 'No Hann', (5, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (255, 255, 255), 1)

    # Draw crosshairs at peak and target center on heatmaps
    for hm, px, py, _ in [(raw_hm_color, raw_peak_px, raw_peak_py, 'raw'),
                            (resp_hm_color, resp_peak_px, resp_peak_py, 'resp')]:
        cx = int((px + 0.5) * stride)
        cy = int((py + 0.5) * stride)
        cv.drawMarker(hm, (cx, cy), (0, 255, 255), cv.MARKER_CROSS, 20, 2)
        cv.drawMarker(hm, (tcx_pixel, tcy_pixel), (0, 255, 0), cv.MARKER_CROSS, 20, 2)

    # Add labels
    cv.putText(disp, f'Green=target({tcx_pixel},{tcy_pixel}) Red=peak({raw_peak_x},{raw_peak_y})',
               (5, 15), cv.FONT_HERSHEY_COMPLEX_SMALL, 0.6, (255, 255, 255), 1)
    cv.putText(raw_hm_color, f'raw peak={raw_np.max():.3f} at ({raw_peak_px},{raw_peak_py})',
               (5, 15), cv.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (255, 255, 255), 1)
    cv.putText(resp_hm_color, f'resp peak={resp_np.max():.3f} at ({resp_peak_px},{resp_peak_py})',
               (5, 15), cv.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (255, 255, 255), 1)
    cv.putText(hann_hm_color, 'Hann window', (5, 15), cv.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (255, 255, 255), 1)

    # Side by side: search crop | raw heatmap | resp heatmap
    # Top: search crop | raw heatmap
    # Bottom: Hann window | resp heatmap
    top = np.hstack([disp, raw_hm_color])
    bottom = np.hstack([hann_hm_color, resp_hm_color])
    result = np.vstack([top, bottom])
    return result


def list_datasets(data_dir):
    """List all available sequences in the dataset directory.

    Supports three layouts:
      - GTOT-style : <seq>/v/*.png + groundTruth_v.txt          (layout='gtot')
      - GOT-10k    : <seq>/*.jpg  + groundtruth.txt             (layout='got10k')
      - LaSOT      : <cat>/<cat>/<seq>/img/*.jpg + groundtruth.txt (layout='lasot',
                     detected via testing_set.txt at the root; sequences are
                     resolved lazily by name, not enumerated)
    """
    sequences = []
    for name in sorted(os.listdir(data_dir)):
        path = os.path.join(data_dir, name)
        if not os.path.isdir(path):
            continue
        has_v = os.path.isdir(os.path.join(path, 'v'))
        has_i = os.path.isdir(os.path.join(path, 'i'))
        if has_v or has_i:
            sequences.append({'name': name, 'path': path, 'layout': 'gtot',
                              'has_v': has_v, 'has_i': has_i})
        elif os.path.isfile(os.path.join(path, 'groundtruth.txt')):
            sequences.append({'name': name, 'path': path, 'layout': 'got10k',
                              'has_v': True, 'has_i': False})
    return sequences


def resolve_lasot_seq(data_dir, seq_name):
    """basketball-1 -> <root>/basketball/basketball/basketball-1 (or None)."""
    cat = seq_name.split('-')[0]
    path = os.path.join(data_dir, cat, cat, seq_name)
    if os.path.isdir(os.path.join(path, 'img')):
        return {'name': seq_name, 'path': path, 'layout': 'lasot',
                'has_v': True, 'has_i': False}
    return None


def run_image_folder(tracker_name, tracker_param, folder_path, optional_box=None, debug=None, save_results=False,
                     debug_dir=None, gt_boxes=None):
    """Run the tracker on an image folder (sequence)."""

    valid_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    images = []
    for f in os.listdir(folder_path):
        ext = os.path.splitext(f)[1].lower()
        if ext in valid_exts:
            images.append(os.path.join(folder_path, f))

    images = sorted(images)

    if len(images) == 0:
        print(f"No images found in {folder_path}")
        return None, None

    print(f"Found {len(images)} images in {folder_path}")

    tracker_obj = Tracker(tracker_name, tracker_param, "video")
    params = tracker_obj.get_parameters()
    params.debug = debug if debug is not None else 0
    tracker = tracker_obj.create_tracker(params)

    # Save search crops for debugging
    if debug_dir is None:
        debug_dir = os.path.join(os.path.dirname(folder_path), 'inputimage')
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)

    output_boxes = []
    track_times = []  # pure tracker.track() time, excludes I/O and drawing

    display_name = 'Display: ' + tracker_name
    cv.namedWindow(display_name, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
    cv.resizeWindow(display_name, 960, 720)

    frame = cv.imread(images[0])
    cv.imshow(display_name, frame)

    def _build_init_info(box):
        return {'init_bbox': box}

    # tracker expects RGB (official pytracking _read_image does BGR2RGB);
    # keep the BGR copy only for display
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    if optional_box is not None:
        assert isinstance(optional_box, (list, tuple))
        assert len(optional_box) == 4
        tracker.initialize(frame_rgb, _build_init_info(optional_box))
        output_boxes.append(optional_box)
        print(f"Using provided box: {optional_box}")
    else:
        print("Select target ROI and press ENTER")
        x, y, w, h = cv.selectROI(display_name, frame.copy(), fromCenter=False)
        init_state = [x, y, w, h]
        tracker.initialize(frame_rgb, _build_init_info(init_state))
        output_boxes.append(init_state)
        print(f"Selected box: {init_state}")

    for i, img_path in enumerate(images[1:], 1):
        frame = cv.imread(img_path)
        if frame is None:
            continue

        t0 = time.time()
        out = tracker.track(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
        track_times.append(time.time() - t0)
        state = [int(s) for s in out['target_bbox']]
        output_boxes.append(state)

        # Save search crop image for debugging (crop is RGB now -> back to BGR)
        search_crop = out.get('search_crop')
        if search_crop is not None and isinstance(search_crop, np.ndarray):
            search_crop = cv.cvtColor(search_crop, cv.COLOR_RGB2BGR)
            # File names use tracker.frame_id so they line up with the
            # tracker's internal [Stage1]/[Stage2] debug prints.
            img_name = f"frame_{tracker.frame_id:04d}.jpg"
            cv.imwrite(os.path.join(debug_dir, img_name), search_crop)

        # ── 热力图可视化：保存热力图叠加图 ──
        response = out.get('response')
        score_map = out.get('score_map')
        if response is not None:
            resp_np = response.squeeze().cpu().numpy()
            raw_np = score_map.squeeze().cpu().numpy()
            prev_state = out.get('prev_state', state)
            resize_factor = out.get('resize_factor', 1.0)
            search_sz = search_crop.shape[0]

            gt_box = gt_boxes[i] if (gt_boxes is not None and i < len(gt_boxes)) else None
            heatmap_overlay = draw_heatmap_comparison(
                search_crop, raw_np, resp_np,
                state, prev_state, resize_factor,
                stride=16, search_size=search_sz,
                hann_window=tracker.output_window,
                gt_box=gt_box
            )
            cv.imwrite(os.path.join(debug_dir, f"heat_{tracker.frame_id:04d}.jpg"), heatmap_overlay)

            _, patch_info = _get_target_patch_heatmap(
                state, score_map,
                prev_state=prev_state,
                resize_factor=resize_factor,
                search_size=search_sz, stride=16
            )

            target_val_raw = raw_np[patch_info['py'], patch_info['px']]
            target_val_resp = resp_np[patch_info['py'], patch_info['px']]

            raw_max_idx = raw_np.argmax()
            raw_max_py, raw_max_px = raw_max_idx // raw_np.shape[1], raw_max_idx % raw_np.shape[1]
            resp_max_idx = resp_np.argmax()
            resp_max_py, resp_max_px = resp_max_idx // resp_np.shape[1], resp_max_idx % resp_np.shape[1]

            raw_max_x = (raw_max_px + 0.5) * 16
            raw_max_y = (raw_max_py + 0.5) * 16
            resp_max_x = (resp_max_px + 0.5) * 16
            resp_max_y = (resp_max_py + 0.5) * 16

            print(f"Frame {tracker.frame_id}: target=({patch_info['tcx']:.0f},{patch_info['tcy']:.0f}), "
                  f"patch=({patch_info['px']},{patch_info['py']}), "
                  f"raw_max=({raw_max_x:.0f},{raw_max_y:.0f}) val={raw_np.max():.3f} "
                  f"resp_max=({resp_max_x:.0f},{resp_max_y:.0f}) val={resp_np.max():.3f} "
                  f"target_raw={target_val_raw:.4f} target_resp={target_val_resp:.4f}")
        # ── ─────────────────────────────────

        frame_disp = frame.copy()
        cv.rectangle(frame_disp, (state[0], state[1]),
                     (state[2] + state[0], state[3] + state[1]),
                     (0, 255, 0), 5)

        cv.putText(frame_disp, f'Frame {tracker.frame_id}/{len(images) - 1}', (20, 30),
                   cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 1)
        cv.putText(frame_disp, 'Press r to reset', (20, 55),
                   cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 1)
        cv.putText(frame_disp, 'Press q to quit', (20, 80),
                   cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 1)

        cv.imshow(display_name, frame_disp)
        key = cv.waitKey(1)

        if key == ord('q'):
            break
        elif key == ord('r'):
            frame_disp = frame.copy()
            cv.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30),
                       cv.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (0, 0, 0), 1)
            cv.imshow(display_name, frame_disp)
            x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
            init_state = [x, y, w, h]
            tracker.initialize(cv.cvtColor(frame, cv.COLOR_BGR2RGB), _build_init_info(init_state))
            output_boxes.append(init_state)

    cv.destroyAllWindows()

    if save_results:
        result_dir = os.path.join(os.path.dirname(folder_path), 'track_results')
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        from pathlib import Path
        video_name = Path(folder_path).stem
        bbox_file = os.path.join(result_dir, f'{video_name}.txt')
        tracked_bb = np.array(output_boxes).astype(int)
        np.savetxt(bbox_file, tracked_bb, delimiter='\t', fmt='%d')
        print(f"Results saved to {bbox_file}")

    print(f"Tracking completed. Total frames: {len(output_boxes)}")
    fps = len(track_times) / sum(track_times) if track_times else None
    return output_boxes, fps


def compute_iou(b, g):
    """IoU of two [x, y, w, h] boxes."""
    ix1, iy1 = max(b[0], g[0]), max(b[1], g[1])
    ix2, iy2 = min(b[0] + b[2], g[0] + g[2]), min(b[1] + b[3], g[1] + g[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = b[2] * b[3] + g[2] * g[3] - inter
    return inter / union if union > 0 else 0.0


def evaluate_sequence(boxes, gt, cover=None, absence=None, is_got10k=False):
    """GOT-10k official protocol when label files are available (cover/absence filtering);
    otherwise falls back to the LaSOT OPE protocol (degenerate-GT skip only).

    GOT-10k: invisible frames (cover.label==0 or absence.label==1) are EXCLUDED
    from the mean, matching got10k-toolkit. SR50/SR75 are computed over valid frames only.
    """
    n = min(len(boxes), len(gt))
    ious = np.array([compute_iou(boxes[i], gt[i]) for i in range(n)])

    if is_got10k and (cover is not None or absence is not None):
        # Official got10k-toolkit protocol: exclude invisible frames
        valid = np.ones(n, dtype=bool)
        if cover is not None and len(cover) >= n:
            valid &= cover[:n] > 0
        elif absence is not None and len(absence) >= n:
            valid &= absence[:n] == 0
        ious_valid = ious[valid]
        if len(ious_valid) == 0:
            return None
        return {'ao': float(np.mean(ious_valid)),
                'sr50': float(np.mean(ious_valid > 0.5)),
                'sr75': float(np.mean(ious_valid > 0.75)),
                'ious': ious_valid,
                'n_eval': int(len(ious_valid)), 'n_total': int(n)}
    else:
        # LaSOT OPE protocol: skip degenerate GT rows only
        cerrs, nerrs = [], []
        valid_idx = []
        for i in range(n):
            g = gt[i]
            if g[2] <= 0 or g[3] <= 0:
                continue
            valid_idx.append(i)
            b = boxes[i]
            bcx, bcy = b[0] + b[2] / 2.0, b[1] + b[3] / 2.0
            gcx, gcy = g[0] + g[2] / 2.0, g[1] + g[3] / 2.0
            cerrs.append(np.hypot(bcx - gcx, bcy - gcy))
            nerrs.append(np.hypot((bcx - gcx) / g[2], (bcy - gcy) / g[3]))
        if not valid_idx:
            return None
        ious_valid = ious[valid_idx]
        cerrs, nerrs = np.array(cerrs), np.array(nerrs)
        iou_thr = np.linspace(0, 1, 21)
        nrm_thr = np.linspace(0, 0.5, 51)
        return {'auc': float(np.mean([np.mean(ious_valid > t) for t in iou_thr])),
                'prec20': float(np.mean(cerrs <= 20)),
                'nprec': float(np.mean([np.mean(nerrs <= t) for t in nrm_thr])),
                'miou': float(np.mean(ious_valid)),
                'n_eval': int(len(ious_valid)), 'n_total': int(n)}


def print_sequence_metrics(seq_name, boxes, gt, cover=None, absence=None, fps=None):
    """Print the per-sequence metric block after tracking finishes.

    Detects GOT-10k label availability and switches protocol:
      - GOT-10k (cover/absence available): prints AO / SR50 / SR75 (official got10k-toolkit)
      - Otherwise                       : prints LaSOT AUC / P@20 / NormPrec / mIoU
    """
    print('\n=== Sequence metrics:', seq_name, '===')
    if gt is None or len(gt) == 0:
        print('No ground truth available, metrics skipped.')
        return
    if len(boxes) != len(gt):
        print(f'Note: {len(boxes)} predicted vs {len(gt)} GT frames '
              f'(early quit / reset?), evaluating first {min(len(boxes), len(gt))}.')
    is_got10k = (cover is not None or absence is not None)
    m = evaluate_sequence(boxes, gt, cover, absence, is_got10k)
    if m is None:
        print('No valid GT frames to evaluate.')
        return
    print(f'Frames evaluated : {m["n_eval"]}/{m["n_total"]}')
    if is_got10k:
        print(f'AO (mIoU)        : {m["ao"]:.4f}')
        print(f'SR@0.50          : {m["sr50"]:.4f}')
        print(f'SR@0.75          : {m["sr75"]:.4f}')
    else:
        print(f'Success AUC      : {m["auc"]:.4f}')
        print(f'Precision@20px   : {m["prec20"]:.4f}')
        print(f'Norm-Precision   : {m["nprec"]:.4f}')
        print(f'mIoU             : {m["miou"]:.4f}')
    if fps is not None:
        print(f'FPS              : {fps:.1f}')


def load_groundtruth(gt_file):
    """Load ground truth from file, auto-detecting the format:
      - comma-separated  x,y,w,h    (GOT-10k groundtruth.txt) -> used as is
      - space-separated  x1 y1 x2 y2 (GTOT groundTruth_*.txt) -> converted to x y w h
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
        if ',' in line:
            parts = list(map(float, line.split(',')))
            if len(parts) >= 4:
                boxes.append([int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])])
        else:
            parts = list(map(float, line.split()))
            if len(parts) >= 4:
                x1, y1, x2, y2 = parts[0], parts[1], parts[2], parts[3]
                # Convert [x1, y1, x2, y2] to [x, y, w, h]
                boxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
    return boxes


def load_label(path):
    """One integer per line (absence.label / cover.label). None if missing.
    Matches got10k-toolkit official filtering protocol.
    """
    if not os.path.isfile(path):
        return None
    with open(path, 'r') as f:
        vals = [int(float(l.strip())) for l in f if l.strip()]
    return np.array(vals) if vals else None


def main():
    parser = argparse.ArgumentParser(description='Run the tracker on a dataset sequence.')
    parser.add_argument('tracker_name', type=str, help='Name of tracking method.')
    parser.add_argument('tracker_param', type=str, help='Name of parameter file.')
    parser.add_argument('--data_dir', type=str,
                        default='E:/gitProjects/OSTrack/data/got10k/val',
                        help='Path to dataset directory (GOT-10k val or GTOT-style).')
    parser.add_argument('--modality', type=str, default='v',
                        choices=['v', 'i', 'all'],
                        help='Modality: v=visible, i=infrared, all=both.')
    parser.add_argument('--seq', type=str, default=None,
                        help='Sequence name to run (if not specified, interactive selection).')
    parser.add_argument('--optional_box', type=float, default=None, nargs="+",
                        help='optional_box with format x y w h. If not provided, uses groundTruth file.')
    parser.add_argument('--debug', type=int, default=0, help='Debug level.')
    parser.add_argument('--save_results', dest='save_results', action='store_true',
                        help='Save bounding boxes')
    parser.set_defaults(save_results=False)

    args = parser.parse_args()

    # LaSOT root: sequences are resolved lazily by name (too many to list)
    is_lasot_root = os.path.isfile(os.path.join(args.data_dir, 'testing_set.txt'))
    if is_lasot_root:
        if args.seq is None:
            print("LaSOT root detected: please pass --seq <name>, e.g. --seq basketball-1")
            return
        seq_info = resolve_lasot_seq(args.data_dir, args.seq)
        if seq_info is None:
            print(f"LaSOT sequence '{args.seq}' not found under {args.data_dir}")
            return
        sequences = [seq_info]
    else:
        # List all sequences
        sequences = list_datasets(args.data_dir)

    if len(sequences) == 0:
        print(f"No sequences found in {args.data_dir}")
        return

    # Select sequence
    if args.seq is not None:
        seq_info = next((s for s in sequences if s['name'] == args.seq), None)
        if seq_info is None:
            print(f"Sequence '{args.seq}' not found. Available sequences:")
            for s in sequences:
                print(f"  {s['name']}")
            return
    else:
        print("\n=== Available Sequences ===")
        for i, s in enumerate(sequences):
            mods = []
            if s['has_v']:
                mods.append('v(visible)')
            if s['has_i']:
                mods.append('i(infrared)')
            print(f"{i+1}. {s['name']} [{', '.join(mods)}]")
        print()

        choice = input("Select sequence number (or q to quit): ").strip()
        if choice.lower() == 'q':
            return
        try:
            idx = int(choice) - 1
            seq_info = sequences[idx]
        except:
            print("Invalid selection.")
            return

    # Select modality (GOT-10k / LaSOT layouts have a single RGB stream)
    if seq_info.get('layout') in ('got10k', 'lasot'):
        modalities = ['rgb']
    elif args.modality == 'all':
        modalities = []
        if seq_info['has_v']:
            modalities.append('v')
        if seq_info['has_i']:
            modalities.append('i')
    else:
        modalities = [args.modality]

    print(f"\nRunning on sequence: {seq_info['name']}")
    print(f"Modalities: {modalities}")

    for mod in modalities:
        if seq_info.get('layout') == 'got10k':
            # GOT-10k: frames live directly in the sequence dir
            folder_path = seq_info['path']
            gt_file = os.path.join(seq_info['path'], 'groundtruth.txt')
            debug_dir = os.path.join(seq_info['path'], 'inputimage')
        elif seq_info.get('layout') == 'lasot':
            # LaSOT: frames live in <seq>/img
            folder_path = os.path.join(seq_info['path'], 'img')
            gt_file = os.path.join(seq_info['path'], 'groundtruth.txt')
            debug_dir = os.path.join(seq_info['path'], 'inputimage')
        else:
            folder_path = os.path.join(seq_info['path'], mod)
            gt_file = os.path.join(seq_info['path'], f'groundTruth_{mod}.txt')
            debug_dir = None  # keep the historical <seq>/inputimage location

        print(f"\n--- Tracking modality: {mod} ({folder_path}) ---")

        if not os.path.isdir(folder_path):
            print(f"Modality folder not found: {folder_path}")
            continue

        # Get init box from ground truth or provided box
        init_box = args.optional_box
        if init_box is None:
            gt_boxes = load_groundtruth(gt_file)
            if gt_boxes is not None and len(gt_boxes) > 0:
                init_box = gt_boxes[0]
                print(f"Using ground truth init box: {init_box}")
            else:
                print(f"No ground truth file found: {gt_file}, please select manually")
                init_box = None
        else:
            print(f"Using provided init box: {init_box}")

        gt_boxes_vis = load_groundtruth(gt_file)
        output_boxes, fps = run_image_folder(
            tracker_name=args.tracker_name,
            tracker_param=args.tracker_param,
            folder_path=folder_path,
            optional_box=init_box,
            debug=args.debug,
            save_results=args.save_results,
            debug_dir=debug_dir,
            gt_boxes=gt_boxes_vis
        )

        # Per-sequence metrics against the full ground-truth file
        # Load GOT-10k label files for official protocol when available
        if output_boxes is not None:
            gt_boxes = load_groundtruth(gt_file)
            cover = None
            absence = None
            if seq_info.get('layout') == 'got10k':
                seq_dir = seq_info['path']
                cover = load_label(os.path.join(seq_dir, 'cover.label'))
                absence = load_label(os.path.join(seq_dir, 'absence.label'))
            print_sequence_metrics(seq_info['name'], output_boxes, gt_boxes,
                                  cover=cover, absence=absence, fps=fps)

        # Auto-save boxes in CSV format for GOT-10k sequences (for comparison with run_got10k.py)
        if output_boxes is not None and seq_info.get('layout') == 'got10k':
            boxes_out_dir = os.path.join(prj_path, 'output', 'batch_eval', 'demo_boxes')
            os.makedirs(boxes_out_dir, exist_ok=True)
            boxes_file = os.path.join(boxes_out_dir, f'{seq_info["name"]}.txt')
            np.savetxt(boxes_file, np.array(output_boxes), delimiter=',', fmt='%.4f')
            print(f'[DEBUG] Boxes saved to {boxes_file}')

        if output_boxes is not None and args.save_results:
            seq_name = seq_info['name']
            result_dir = os.path.join(args.data_dir, 'track_results', seq_name)
            os.makedirs(result_dir, exist_ok=True)
            bbox_file = os.path.join(result_dir, f'{seq_name}.txt')
            tracked_bb = np.array(output_boxes).astype(int)
            np.savetxt(bbox_file, tracked_bb, delimiter='\t', fmt='%d')
            print(f"Saved to {bbox_file}")


if __name__ == '__main__':
    main()