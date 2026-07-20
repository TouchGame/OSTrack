import os
import sys
import argparse

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

from lib.test.evaluation import Tracker
import cv2 as cv
import numpy as np


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

    # Extract heatmap value at that patch
    score_np = score_map.squeeze().cpu().numpy()
    heat_val = score_np[py, px]

    patch_info = {
        'tcx': tcx_in_resized,
        'tcy': tcy_in_resized,
        'px': px,
        'py': py,
    }
    return float(heat_val), patch_info


def draw_heatmap_comparison(search_crop, raw_np, resp_np, state, prev_state, resize_factor,
                            stride=16, search_size=256, hann_window=None):
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
    # Draw predicted bbox
    x1 = int(np.clip(state[0], 0, search_size - 1))
    y1 = int(np.clip(state[1], 0, search_size - 1))
    x2 = int(np.clip(state[0] + state[2], 0, search_size - 1))
    y2 = int(np.clip(state[1] + state[3], 0, search_size - 1))
    cv.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
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
    """List all available sequences in the dataset directory."""
    sequences = []
    for name in os.listdir(data_dir):
        path = os.path.join(data_dir, name)
        if os.path.isdir(path):
            # Check if it has 'v' or 'i' subfolders
            has_v = os.path.isdir(os.path.join(path, 'v'))
            has_i = os.path.isdir(os.path.join(path, 'i'))
            if has_v or has_i:
                sequences.append({'name': name, 'path': path, 'has_v': has_v, 'has_i': has_i})
    return sequences


def run_image_folder(tracker_name, tracker_param, folder_path, optional_box=None, debug=None, save_results=False):
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
        return None

    print(f"Found {len(images)} images in {folder_path}")

    tracker_obj = Tracker(tracker_name, tracker_param, "video")
    params = tracker_obj.get_parameters()
    params.debug = debug if debug is not None else 0
    tracker = tracker_obj.create_tracker(params)

    # Save search crops for debugging
    debug_dir = os.path.join(os.path.dirname(folder_path), 'inputimage')
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)

    output_boxes = []
    score_maps = []  # store score_map for each frame

    display_name = 'Display: ' + tracker_name
    cv.namedWindow(display_name, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
    cv.resizeWindow(display_name, 960, 720)

    frame = cv.imread(images[0])
    cv.imshow(display_name, frame)

    def _build_init_info(box):
        return {'init_bbox': box}

    if optional_box is not None:
        assert isinstance(optional_box, (list, tuple))
        assert len(optional_box) == 4
        tracker.initialize(frame, _build_init_info(optional_box))
        output_boxes.append(optional_box)
        print(f"Using provided box: {optional_box}")
    else:
        print("Select target ROI and press ENTER")
        x, y, w, h = cv.selectROI(display_name, frame.copy(), fromCenter=False)
        init_state = [x, y, w, h]
        tracker.initialize(frame, _build_init_info(init_state))
        output_boxes.append(init_state)
        print(f"Selected box: {init_state}")

    for i, img_path in enumerate(images[1:], 1):
        frame = cv.imread(img_path)
        if frame is None:
            continue

        out = tracker.track(frame)
        state = [int(s) for s in out['target_bbox']]
        output_boxes.append(state)

        # Save search crop image for debugging
        search_crop = out.get('search_crop')
        if search_crop is not None and isinstance(search_crop, np.ndarray):
            img_name = f"frame_{i+1:04d}.jpg"
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

            heatmap_overlay = draw_heatmap_comparison(
                search_crop, raw_np, resp_np,
                state, prev_state, resize_factor,
                stride=16, search_size=search_sz,
                hann_window=tracker.output_window
            )
            cv.imwrite(os.path.join(debug_dir, f"heat_{i+1:04d}_fid{tracker.frame_id}.jpg"), heatmap_overlay)

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

            print(f"Frame {i+1}: target=({patch_info['tcx']:.0f},{patch_info['tcy']:.0f}), "
                  f"patch=({patch_info['px']},{patch_info['py']}), "
                  f"raw_max=({raw_max_x:.0f},{raw_max_y:.0f}) val={raw_np.max():.3f} "
                  f"resp_max=({resp_max_x:.0f},{resp_max_y:.0f}) val={resp_np.max():.3f} "
                  f"target_raw={target_val_raw:.4f} target_resp={target_val_resp:.4f}")
        # ── ─────────────────────────────────

        frame_disp = frame.copy()
        cv.rectangle(frame_disp, (state[0], state[1]),
                     (state[2] + state[0], state[3] + state[1]),
                     (0, 255, 0), 5)

        cv.putText(frame_disp, f'Frame {i+1}/{len(images)}', (20, 30),
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
            tracker.initialize(frame, _build_init_info(init_state))
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
    return output_boxes


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


def main():
    parser = argparse.ArgumentParser(description='Run the tracker on a dataset sequence.')
    parser.add_argument('tracker_name', type=str, help='Name of tracking method.')
    parser.add_argument('tracker_param', type=str, help='Name of parameter file.')
    parser.add_argument('--data_dir', type=str,
                        default='E:/gitProjects/OSTrack/data/Multi_Modal_RGBT_dataset_CSR',
                        help='Path to dataset directory.')
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

    # Select modality
    if args.modality == 'all':
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
        folder_path = os.path.join(seq_info['path'], mod)
        gt_file = os.path.join(seq_info['path'], f'groundTruth_{mod}.txt')

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

        output_boxes = run_image_folder(
            tracker_name=args.tracker_name,
            tracker_param=args.tracker_param,
            folder_path=folder_path,
            optional_box=init_box,
            debug=args.debug,
            save_results=args.save_results
        )

        if output_boxes is not None and args.save_results:
            result_dir = os.path.join(args.data_dir, 'track_results', seq_name)
            os.makedirs(result_dir, exist_ok=True)
            bbox_file = os.path.join(result_dir, f'{seq_name}.txt')
            tracked_bb = np.array(output_boxes).astype(int)
            np.savetxt(bbox_file, tracked_bb, delimiter='\t', fmt='%d')
            print(f"Saved to {bbox_file}")


if __name__ == '__main__':
    main()