import os
import sys
import argparse

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

from lib.test.evaluation import Tracker
import cv2 as cv
import numpy as np


def run_image_folder(tracker_name, tracker_param, folder_path, optional_box=None, debug=None, save_results=False):
    """Run the tracker on an image folder (sequence)."""

    # Get all image files in the folder
    valid_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    images = []
    for f in os.listdir(folder_path):
        ext = os.path.splitext(f)[1].lower()
        if ext in valid_exts:
            images.append(os.path.join(folder_path, f))

    images = sorted(images)

    if len(images) == 0:
        print(f"No images found in {folder_path}")
        return

    print(f"Found {len(images)} images in {folder_path}")

    tracker_obj = Tracker(tracker_name, tracker_param, "video")

    # Get params and create tracker (mimic run_video logic)
    params = tracker_obj.get_parameters()
    params.debug = debug if debug is not None else 0
    tracker = tracker_obj.create_tracker(params)

    output_boxes = []

    display_name = 'Display: ' + tracker_name
    cv.namedWindow(display_name, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
    cv.resizeWindow(display_name, 960, 720)

    # Read first frame
    frame = cv.imread(images[0])
    cv.imshow(display_name, frame)

    def _build_init_info(box):
        return {'init_bbox': box}

    # Initialize with user selection or provided box
    if optional_box is not None:
        assert isinstance(optional_box, (list, tuple))
        assert len(optional_box) == 4, "valid box's format is [x,y,w,h]"
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

    # Track all images
    for i, img_path in enumerate(images[1:], 1):
        frame = cv.imread(img_path)
        if frame is None:
            print(f"Failed to read {img_path}")
            continue

        out = tracker.track(frame)
        state = [int(s) for s in out['target_bbox']]
        output_boxes.append(state)

        # Draw box
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

    # Save results
    if save_results:
        from pathlib import Path
        result_dir = os.path.join(os.path.dirname(folder_path), 'track_results')
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        video_name = Path(folder_path).stem
        bbox_file = os.path.join(result_dir, f'{video_name}.txt')
        tracked_bb = np.array(output_boxes).astype(int)
        np.savetxt(bbox_file, tracked_bb, delimiter='\t', fmt='%d')
        print(f"Results saved to {bbox_file}")

    print(f"Tracking completed. Total frames: {len(output_boxes)}")


def main():
    parser = argparse.ArgumentParser(description='Run the tracker on an image folder.')
    parser.add_argument('tracker_name', type=str, help='Name of tracking method.')
    parser.add_argument('tracker_param', type=str, help='Name of parameter file.')
    parser.add_argument('folder_path', type=str, help='Path to image folder.')
    parser.add_argument('--optional_box', type=float, default=None, nargs="+",
                        help='optional_box with format x y w h.')
    parser.add_argument('--debug', type=int, default=0, help='Debug level.')
    parser.add_argument('--save_results', dest='save_results', action='store_true',
                        help='Save bounding boxes')
    parser.set_defaults(save_results=False)

    args = parser.parse_args()

    run_image_folder(args.tracker_name, args.tracker_param, args.folder_path,
                     args.optional_box, args.debug, args.save_results)


if __name__ == '__main__':
    main()