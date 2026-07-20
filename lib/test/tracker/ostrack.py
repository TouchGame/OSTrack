import math
import numpy as np
from scipy.ndimage import label as scipy_label

from lib.models.ostrack import build_ostrack
from lib.test.tracker.basetracker import BaseTracker
import torch

from lib.test.tracker.vis_utils import gen_visualization
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
# for debug
import cv2
import os

from lib.test.tracker.data_utils import Preprocessor
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond


class OSTrack(BaseTracker):
    def __init__(self, params, dataset_name):
        super(OSTrack, self).__init__(params)
        network = build_ostrack(params.cfg, training=False)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None

        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        # motion constrain
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()

        # for debug
        self.debug = params.debug
        self.use_visdom = params.debug
        self.frame_id = 0
        # Always enable file-based visualization (save images with Hann window in bottom-left)
        self.save_dir = "debug"
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        if self.debug and self.use_visdom:
            # self.add_hook()
            self._init_visdom(None, 1)
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        self.z_dict1 = {}

        # for multi-candidate decision
        self.use_multicandidate = True

        # for prev frame target feature
        self.prev_search_crop = None
        self.prev_state = None

        # for occlusion prediction
        self.position_history = []  # [(dx, dy), ...] in feature map coordinates

        # for box size history (recent 3 frames)
        self.box_size_history = []  # [(w, h), ...]

        # for consecutive prediction limit
        self.consecutive_predictions = 0  # count of consecutive frames using prediction
        self.max_consecutive_predictions = 8  # max allowed consecutive predictions

    def initialize(self, image, info: dict):
        # forward the template once
        z_patch_arr, resize_factor, z_amask_arr = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)
        self.z_patch_arr = z_patch_arr
        template = self.preprocessor.process(z_patch_arr, z_amask_arr)
        with torch.no_grad():
            self.z_dict1 = template

        self.box_mask_z = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            template_bbox = self.transform_bbox_to_crop(info['init_bbox'], resize_factor,
                                                        template.tensors.device).squeeze(1)
            self.box_mask_z = generate_mask_cond(self.cfg, 1, template.tensors.device, template_bbox)

        # save states
        self.state = info['init_bbox']
        self.frame_id = 0
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def _topk_pool2d(self, score_map: torch.Tensor, k: int = 3):
        """Extract top-K peak positions from score_map.

        Args:
            score_map: [B, 1, H, W] raw score heatmap
            k: number of top peaks to extract

        Returns:
            topk_scores: [B, K] scores of top-K peaks
            topk_positions: [B, K, 2] positions (y, x) of top-K peaks
        """
        B, C, H, W = score_map.shape
        score_flat = score_map.view(B, -1)  # [B, H*W]
        topk_scores, topk_indices = torch.topk(score_flat, k=k, dim=1)  # [B, K]
        topk_y = topk_indices // W  # row indices
        topk_x = topk_indices % W   # col indices
        topk_positions = torch.stack([topk_y, topk_x], dim=2)  # [B, K, 2]
        return topk_scores, topk_positions

    def _compute_cosine_similarity(self, search_tokens: torch.Tensor, candidate_positions: torch.Tensor,
                                   prev_feat_normalized: torch.Tensor):
        """Compute cosine similarity between candidate token features and previous frame features.

        Args:
            search_tokens: [B, L_x, C] search region tokens from backbone
            candidate_positions: [B, K, 2] positions (y, x) in heatmap coordinates
            prev_feat_normalized: [B, C] normalized previous frame target features

        Returns:
            similarities: [B, K] cosine similarity scores
        """
        B, L_x, C = search_tokens.shape
        K = candidate_positions.shape[1]
        feat_sz = self.feat_sz  # e.g., 16 for 256 search or 24 for 384

        # Map (y, x) to token index in search region
        y = candidate_positions[..., 0]  # [B, K]
        x = candidate_positions[..., 1]  # [B, K]
        token_indices = y * feat_sz + x  # [B, K], flat token indices within search tokens

        # Gather features at candidate positions: [B, K, C]
        indices_expanded = token_indices.unsqueeze(-1).expand(B, K, C)  # [B, K, C]
        candidate_features = search_tokens.gather(dim=1, index=indices_expanded.to(torch.int64))  # [B, K, C]

        # Debug
        print(f"  [Sim Debug] token_indices={token_indices}, candidate_features norm={candidate_features.norm():.4f}")

        # Normalize candidate features
        candidate_features_norm = torch.nn.functional.normalize(candidate_features, p=2, dim=-1)  # [B, K, C]

        # Cosine similarity: [B, K, C] @ [B, C, 1] -> [B, K]
        similarities = (candidate_features_norm * prev_feat_normalized.unsqueeze(1)).sum(dim=-1)

        print(f"  [Sim Debug] candidate_features_norm norm={candidate_features_norm.norm():.4f}, similarities={similarities}")

        return similarities

    def _findLocalPeaks(self, score_map, min_distance=3):
        """Find local peaks in heatmap using NMS-like approach.

        Args:
            score_map: [B, 1, H, W]
            min_distance: minimum distance between peaks

        Returns:
            peaks: list of [K, 2] (y, x) peak positions
            peak_scores: list of [K] peak scores
        """
        B = score_map.shape[0]
        H, W = score_map.shape[2], score_map.shape[3]
        score_np = score_map.squeeze(1).cpu().numpy()  # [B, H, W]

        all_peaks = []
        all_peak_scores = []

        for b in range(B):
            heat = score_np[b]  # [H, W]
            peaks = []
            peak_vals = []

            # Simple greedy NMS-like peak finding
            threshold_fraction = 0.3
            min_val = heat.max() * threshold_fraction
            flat_indices = np.argsort(heat.flatten())[::-1]  # sorted descending

            visited = set()
            for flat_idx in flat_indices:
                if flat_idx in visited:
                    continue
                y, x = flat_idx // W, flat_idx % W
                val = heat[y, x]
                if val < min_val:
                    break
                peaks.append([y, x])
                peak_vals.append(val)
                # Mark neighbors as visited
                for dy in range(-min_distance, min_distance + 1):
                    for dx in range(-min_distance, min_distance + 1):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < H and 0 <= nx < W:
                            visited.add(ny * W + nx)

            all_peaks.append(peaks)
            all_peak_scores.append(peak_vals)

        return all_peaks, all_peak_scores

    def _predict_displacement(self):
        """Predict displacement using recent position history.

        Returns:
            (dx, dy): predicted displacement in pixel coordinates
        """
        if len(self.position_history) < 2:
            print(f"  [Predict] Not enough history, returning 0")
            return 0.0, 0.0

        # Use recent N frames average displacement
        n = min(5, len(self.position_history))
        recent = self.position_history[-n:]
        avg_dx = sum(d[0] for d in recent) / n
        avg_dy = sum(d[1] for d in recent) / n
        print(f"  [Predict] history={self.position_history[-n:]}, avg_dx={avg_dx:.2f}, avg_dy={avg_dy:.2f}")
        return avg_dx, avg_dy

    def _regionGrowing(self, score_map, peaks, peak_scores, threshold_fraction=0.3):
        """Priority-queue based region growing with bbox size constraint.

        Args:
            score_map: [B, 1, H, W]
            peaks: list of [K, 2] peak positions
            peak_scores: list of [K] peak scores
            threshold_fraction: grow until score drops below threshold

        Returns:
            centers: [B, K, 2] region centers (the peak positions)
            scores: [B, K] region scores (peak values)
            masks: list of [B, K, H, W] region masks
        """
        import heapq

        B = score_map.shape[0]
        H, W = score_map.shape[2], score_map.shape[3]
        score_np = score_map.squeeze(1).cpu().numpy()  # [B, H, W]

        all_centers = []
        all_scores = []
        all_masks = []

        for b in range(B):
            peaks_b = peaks[b]
            scores_b = peak_scores[b]
            K = len(peaks_b)

            # Maximum bbox size: 1.2x of previous frame's target size
            # Use feature map size * 0.3 as default if no prev_info
            max_bbox_h = max(3, int(H * 0.3))
            max_bbox_w = max(3, int(W * 0.3))

            # visited_or_owner: 0=unclaimed, 1..K=owner region id
            visited_or_owner = np.zeros((H, W), dtype=np.int32)
            # Priority queues: (neg_score, y, x) - higher score = lower neg = higher priority
            queues = [[] for _ in range(K)]
            # Track bbox for each region
            bboxes = [None] * K
            # Threshold override for boundary cases
            threshold_override = [scores_b[pi] * threshold_fraction for pi in range(K)]

            # 4-connectivity neighbors
            neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]

            # Initialize all peaks
            for pi, (py, px) in enumerate(peaks_b):
                visited_or_owner[py, px] = pi + 1
                heapq.heappush(queues[pi], (-scores_b[pi], py, px))
                bboxes[pi] = {'min_y': py, 'max_y': py, 'min_x': px, 'max_x': px}

            # Expansion with priority queue and bbox constraint
            while any(len(q) > 0 for q in queues):
                for pi in range(K):
                    if len(queues[pi]) == 0:
                        continue
                    neg_score, y, x = heapq.heappop(queues[pi])
                    curr_threshold = threshold_override[pi]

                    for dy, dx in neighbors:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < H and 0 <= nx < W:
                            owner = visited_or_owner[ny, nx]
                            if owner == 0:
                                # Unclaimed cell
                                if score_np[b, ny, nx] >= curr_threshold:
                                    # Check bbox constraint
                                    bbox = bboxes[pi]
                                    new_min_y = min(bbox['min_y'], ny)
                                    new_max_y = max(bbox['max_y'], ny)
                                    new_min_x = min(bbox['min_x'], nx)
                                    new_max_x = max(bbox['max_x'], nx)
                                    new_h = new_max_y - new_min_y + 1
                                    new_w = new_max_x - new_min_x + 1
                                    if new_h <= max_bbox_h and new_w <= max_bbox_w:
                                        visited_or_owner[ny, nx] = pi + 1
                                        heapq.heappush(queues[pi], (-score_np[b, ny, nx], ny, nx))
                                        bboxes[pi] = {'min_y': new_min_y, 'max_y': new_max_y,
                                                      'min_x': new_min_x, 'max_x': new_max_x}
                            elif owner != pi + 1:
                                # Touched another region - raise threshold if lower peak
                                pj = owner - 1
                                if scores_b[pi] < scores_b[pj]:
                                    threshold_override[pi] = max(threshold_override[pi], scores_b[pj] * threshold_fraction)

            # Build masks from visited_or_owner
            centers_b = []
            scores_out_b = []
            masks_b = []

            for pi in range(K):
                py, px = peaks_b[pi]
                mask = (visited_or_owner == (pi + 1))
                centers_b.append([py, px])
                scores_out_b.append(scores_b[pi])
                masks_b.append(mask)

            # Debug: print region mask ranges
            for ri, (center, mask) in enumerate(zip(centers_b, masks_b)):
                mask_y, mask_x = np.where(mask)
                if len(mask_y) > 0:
                    print(f"    [RegionGrowing] Region {ri}: center=({center[0]},{center[1]}), mask y=[{mask_y.min()},{mask_y.max()}], x=[{mask_x.min()},{mask_x.max()}]")

            all_centers.append(centers_b)
            all_scores.append(scores_out_b)
            all_masks.append(masks_b)

        return all_centers, all_scores, all_masks

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        # Save previous state for heatmap calculation
        prev_state = self.state if self.frame_id > 1 else self.state
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)

        with torch.no_grad():
            x_dict = search
            # merge the template and the search
            # run the transformer
            out_dict = self.network.forward(
                template=self.z_dict1.tensors, search=x_dict.tensors, ce_template_mask=self.box_mask_z)

        # add hann windows
        pred_score_map = out_dict['score_map']
        response = self.output_window * pred_score_map

        # ============================================================
        # Stage 1: Interference Warning
        # ============================================================
        raw_max_idx = pred_score_map.flatten(1).argmax(dim=1)  # [B], flat index
        resp_max_idx = response.flatten(1).argmax(dim=1)       # [B], flat index

        # Check raw_max position change (significant jump = possible interference)
        raw_y = int(raw_max_idx[0].item() // self.feat_sz)
        raw_x = int(raw_max_idx[0].item() % self.feat_sz)
        raw_jump = 0.0
        if hasattr(self, '_prev_raw_max') and self._prev_raw_max is not None:
            raw_jump = math.sqrt((raw_y - self._prev_raw_max[0])**2 + (raw_x - self._prev_raw_max[1])**2)
        print(f"  [Debug] raw_y={raw_y}, raw_x={raw_x}, raw_jump={raw_jump:.2f}, prev_jump={getattr(self, '_prev_raw_jump', 0.0):.2f}")
        self._prev_raw_max = (raw_y, raw_x)
        self._prev_raw_jump = raw_jump

        interference_warning = (raw_max_idx != resp_max_idx).any().item()
        feat_center = self.feat_sz // 2  # 12 for 24x24
        # Also trigger warning if raw peak is far from expected center
        dist_to_center = math.sqrt((raw_y - feat_center)**2 + (raw_x - feat_center)**2)
        if not interference_warning and dist_to_center > 3.0:
            # Raw peak is far from center (>3 pixels in feature map), possible interference
            interference_warning = True
            print(f'  [Warning] raw peak ({raw_y},{raw_x}) far from center {feat_center}, dist={dist_to_center:.2f}')
        # Also trigger if raw_max == resp_max but jumped far from previous raw_max
        if not interference_warning:
            prev_raw_y = getattr(self, '_prev_raw_y', raw_y)
            prev_raw_x = getattr(self, '_prev_raw_x', raw_x)
            prev_movement = getattr(self, '_prev_raw_movement', 0.0)
            dist = math.sqrt((raw_y - prev_raw_y)**2 + (raw_x - prev_raw_x)**2)
            print(f"  [Warning Check] raw=({raw_y},{raw_x}), prev_raw=({prev_raw_y:.2f},{prev_raw_x:.2f}), dist={dist:.2f}, prev_mv={prev_movement:.2f}")
            if dist > 2 * prev_movement:
                interference_warning = True
                print(f'  [Warning] dist={dist:.2f} > 2*prev_movement={2*prev_movement:.2f}, possible interference')

        self._interference_warning = interference_warning

        resp_y = int(resp_max_idx[0].item() // self.feat_sz)
        resp_x = int(resp_max_idx[0].item() % self.feat_sz)
        print(f'Frame {self.frame_id}: raw_idx=({raw_y},{raw_x}), resp_idx=({resp_y},{resp_x}), warning={interference_warning}')

        if interference_warning and self.use_multicandidate:
            print(f'[Stage1] Interference warning triggered at frame {self.frame_id}')

        # ============================================================
        # Stage 2: Multi-Candidate Decision (only when warning triggers)
        # ============================================================
        if interference_warning and self.use_multicandidate and self.prev_search_crop is not None:
            # Find connected regions using local peaks + region growing
            max_score = pred_score_map.max().item()
            peaks, peak_scores = self._findLocalPeaks(pred_score_map, min_distance=3)
            centers, region_scores_list, masks = self._regionGrowing(
                pred_score_map, peaks, peak_scores, threshold_fraction=0.3)
            print(f"  [Stage2] max_score={max_score:.4f}, num_peaks={len(peaks[0])}, num_regions={len(centers[0])}")

            # Occlusion detection: check if any region is near expected position (center of search region)
            # Expected position in feature map coordinates is center (12, 12) for 24x24
            feat_center_y, feat_center_x = self.feat_sz // 2, self.feat_sz // 2
            occlusion_threshold = 3.0  # pixels in feature map
            near_expected = False
            for center_y, center_x in centers[0]:
                dist = math.sqrt((center_y - feat_center_y)**2 + (center_x - feat_center_x)**2)
                if dist < occlusion_threshold:
                    near_expected = True
                    break

            # Skip boosting if only one region found (no real interference)
            skip_boosting = len(centers[0]) == 1
            if skip_boosting:
                print(f"  [Stage2] Only one region found, skipping boosting")

            # Compute prev_frame target features for similarity
            lens_z = self.network.backbone.pos_embed_z.shape[1]
            lens_x = self.network.backbone.pos_embed_x.shape[1]

            # Crop previous frame's target from prev_search_crop
            # prev_search_crop is 384x384, target at center (192,192)
            # Extract 128x128 centered on target
            search_sz = self.params.search_size  # 384
            template_sz = self.params.template_size  # 128
            center = search_sz // 2  # 192
            half = template_sz // 2  # 64
            prev_target_crop_np = self.prev_search_crop[center-half:center+half, center-half:center+half]
            # preprocessor expects HWC format
            prev_target_amask = np.ones((template_sz, template_sz), dtype=bool)
            prev_target_tensor = self.preprocessor.process(prev_target_crop_np, prev_target_amask)

            # Also prepare prev_search_crop as x (search region) for backbone
            prev_search_tensor = self.preprocessor.process(self.prev_search_crop, prev_target_amask)

            # Get backbone features: z=template(128x128), x=search(384x384)
            with torch.no_grad():
                prev_feat_out, _ = self.network.backbone(
                    z=prev_target_tensor.tensors, x=prev_search_tensor.tensors)
                prev_feat = prev_feat_out[:, :lens_z, :]
                prev_feat_normalized = torch.nn.functional.normalize(prev_feat.mean(dim=1), p=2, dim=-1)

            # Get current frame search tokens
            backbone_feat = out_dict['backbone_feat']  # [B, Lz+Lx, C]
            search_tokens = backbone_feat[:, lens_z:lens_z + lens_x, :]  # [B, Lx, C]

            # Debug: check prev_feat_normalized and search_tokens
            print(f"  [Debug] prev_feat_normalized shape={prev_feat_normalized.shape}, mean={prev_feat_normalized.mean():.4f}, norm={prev_feat_normalized.norm():.4f}")
            print(f"  [Debug] search_tokens shape={search_tokens.shape}, mean={search_tokens.mean():.4f}")

            # Compute similarity for each region
            combined_list = []
            raw_sims_list = []  # Save raw similarities for debug printing

            # Use feature map center as reference since each frame is cropped around target
            feat_center = self.feat_sz // 2
            sigma = 2.0  # fixed sigma

            for b in range(pred_score_map.shape[0]):
                centers_b = centers[b]
                scores_b = torch.tensor(region_scores_list[b], dtype=torch.float32, device=pred_score_map.device)
                masks_b = masks[b]

                if len(centers_b) == 0:
                    combined_list.append(torch.tensor([0.0]))
                    continue

                # Compute similarity for each region
                positions_tensor = torch.tensor(centers_b, dtype=torch.long, device=pred_score_map.device).unsqueeze(0)  # [1, K, 2]
                sims = self._compute_cosine_similarity(search_tokens[b:b+1], positions_tensor, prev_feat_normalized[b:b+1])  # [1, K]
                raw_sims_list.append(sims.squeeze(0).cpu().numpy())  # Save raw similarity for printing

                # Compute position consistency score (Gaussian distance penalty)
                position_scores = []
                for cy, cx in centers_b:
                    dist = math.sqrt((cy - feat_center)**2 + (cx - feat_center)**2)
                    pos_score = math.exp(-dist**2 / (2 * sigma**2))
                    position_scores.append(pos_score)
                position_scores_tensor = torch.tensor(position_scores, dtype=torch.float32, device=pred_score_map.device)
                print(f"  [Debug] center={feat_center}, position_scores={position_scores}")

                # Combined: score × similarity × position_score
                combined_b = scores_b.unsqueeze(0) * sims * position_scores_tensor.unsqueeze(0)  # [1, K]
                print(f"  [Debug] scores_b={scores_b[:2]}, position_scores={position_scores[:2]}, combined_b={combined_b[:2]}")
                combined_list.append(combined_b.squeeze(0))

            # Find best region - handle each batch element
            best_idx_per_batch = []
            best_combined_val = []
            for b in range(pred_score_map.shape[0]):
                if len(combined_list[b].shape) == 0 or combined_list[b].numel() == 0:
                    best_idx_per_batch.append(0)
                    best_combined_val.append(0.0)
                    continue
                combined_b = combined_list[b]
                if combined_b.dim() > 1:
                    combined_b = combined_b.squeeze()
                bi = combined_b.argmax(dim=0).item()
                best_idx_per_batch.append(bi)
                best_combined_val.append(float(combined_b[bi].item()))

            print(f"  [Stage2] regions={len(centers[0])}, centers={centers[0]}, "
                  f"scores={[f'{s:.3f}' for s in region_scores_list[0]]}, "
                  f"combined={[f'{c:.3f}' for c in best_combined_val]}, best_idx={best_idx_per_batch[0]}")

            # Also print similarity per region for each batch
            for b in range(pred_score_map.shape[0]):
                regions_b = centers[b]
                raw_sims_b = raw_sims_list[b] if b < len(raw_sims_list) else []
                print(f"    Batch {b}: {len(regions_b)} regions")
                for ri, (cy, cx) in enumerate(regions_b):
                    score_val = region_scores_list[b][ri]
                    raw_sim = raw_sims_b[ri] if ri < len(raw_sims_b) else 0.0
                    combined_val = combined_list[b][ri].item() if ri < len(combined_list[b]) else 0.0
                    print(f"      Region {ri}: center=({cy},{cx}), score={score_val:.3f}, sim={raw_sim:.3f}, combined={combined_val:.3f}")

            # Occlusion detection: check if all combined scores are very low
            max_combined = 0.0
            for b in range(pred_score_map.shape[0]):
                if len(combined_list[b]) > 0 and combined_list[b].numel() > 0:
                    max_combined = max(max_combined, combined_list[b].max().item())
            low_confidence_threshold = 0.1

            # Check consecutive prediction limit before entering prediction mode
            # Allow up to max_consecutive_predictions, skip when exceeded
            if self.consecutive_predictions >= self.max_consecutive_predictions:
                print(f"  [Occlusion] Consecutive predictions limit reached ({self.consecutive_predictions}), skipping prediction")
                skip_prediction = True
            elif max_combined < low_confidence_threshold:
                print(f"  [Occlusion] All regions have low combined score ({max_combined:.3f} < {low_confidence_threshold}), predicting position")
                skip_prediction = False
                self.consecutive_predictions += 1
            else:
                skip_prediction = True

            if not skip_prediction:
                response = self.output_window * pred_score_map
                vis_score_map = pred_score_map[0, 0].cpu().numpy()
                # Predict displacement from history (now in pixel coordinates)
                pred_dx, pred_dy = self._predict_displacement()
                print(f"  [Occlusion Debug] pred_dx={pred_dx:.2f}, pred_dy={pred_dy:.2f}")
                x1, y1, w, h = self.state
                print(f"  [Occlusion Debug] x1={x1:.2f}, y1={y1:.2f}, w={w:.2f}, h={h:.2f}")
                # Apply pixel displacement directly
                pred_x = x1 + pred_dx
                pred_y = y1 + pred_dy
                print(f"  [Occlusion Debug] After pred: pred_x={pred_x:.2f}, pred_y={pred_y:.2f}")
                self.state = [pred_x, pred_y, w, h]
                image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.rectangle(image_BGR, (int(pred_x),int(pred_y)), (int(pred_x+w),int(pred_y+h)), color=(0,0,255), thickness=2)
                hann_vis = self.output_window[0].cpu().numpy()
                hann_vis = hann_vis.squeeze()
                hann_size = min(100, image_BGR.shape[0] - 10, image_BGR.shape[1] - 10)
                if hann_size > 20:
                    hann_vis_resized = cv2.resize(hann_vis, (hann_size, hann_size))
                    hann_vis_resized = (hann_vis_resized * 255).astype(np.uint8)
                    hann_color = cv2.applyColorMap(hann_vis_resized, cv2.COLORMAP_JET)
                    y1_hann = image_BGR.shape[0] - hann_size
                    image_BGR[y1_hann:y1_hann+hann_size, 0:hann_size] = hann_color
                if hasattr(self, 'save_dir') and self.save_dir:
                    import os
                    save_path = os.path.abspath(os.path.join(self.save_dir, "%04d.jpg" % self.frame_id))
                    cv2.imwrite(save_path, image_BGR)
                # Don't clear history during occlusion - just keep it for next prediction
                self._prev_state = list(self.state)
                self.prev_search_crop = x_patch_arr.copy()
                if self.save_all_boxes:
                    self.save_all_boxes.append(self.state)
                return {"target_bbox": self.state,
                        "vis_score_map": vis_score_map,
                        "response": response.squeeze(0).cpu(),
                        "score_map": pred_score_map[0, 0].cpu(),
                        "prev_state": prev_state,
                        "resize_factor": resize_factor,
                        "search_crop": x_patch_arr}

            # Build boosted score_map using best region mask
            # Only keep the selected region, zero out others
            if skip_boosting:
                # Single region - use normal response (Hann modulated)
                response = self.output_window * pred_score_map
                vis_score_map = pred_score_map[0, 0].cpu().numpy()
                print(f"  [Stage2] Using normal response (no boost)")
                pred_boxes = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'])
                pred_boxes = pred_boxes.view(-1, 4)
                pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()
                self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)
            else:
                boosted_score_map = torch.zeros_like(pred_score_map)
                boost_factor = 3.0
                min_region_area = 9
                for b in range(pred_score_map.shape[0]):
                    bi = best_idx_per_batch[b]
                    print(f"  [Draw Debug] batch={b}, best_idx={bi}, num_masks={len(masks[b])}")
                    # Debug: show all region mask ranges
                    for ri, mask in enumerate(masks[b]):
                        mask_tensor = torch.from_numpy(mask.astype(np.float32))
                        mask_y, mask_x = torch.where(mask_tensor > 0)
                        if len(mask_y) > 0:
                            print(f"    Region {ri} mask: min_y={mask_y.min()}, max_y={mask_y.max()}, min_x={mask_x.min()}, max_x={mask_x.max()}")
                    if bi < len(masks[b]):
                        best_center = centers[b][bi]
                        best_mask = torch.from_numpy(masks[b][bi].astype(np.float32)).to(pred_score_map.device)
                        region_area = best_mask.sum().item()
                        print(f"    Using region {bi}: center={best_center}, area={region_area}")
                        print(f"    best_mask shape={best_mask.shape}, mask at (11,14)={best_mask[11, 14].item():.2f}, mask at (15,8)={best_mask[15, 8].item():.2f}")
                        if region_area >= min_region_area:
                            # Only selected region has non-zero scores
                            boosted_score_map[b, 0] = best_mask * pred_score_map[b, 0] * boost_factor
                        else:
                            # Disabled fallback: use selected region anyway even if small
                            # print(f"  [Stage2] region area {region_area} < {min_region_area}, fallback to raw max")
                            # max_idx = pred_score_map[b, 0].argmax()
                            # boosted_score_map[b, 0].flatten()[max_idx] = pred_score_map[b, 0].flatten()[max_idx] * boost_factor
                            boosted_score_map[b, 0] = best_mask * pred_score_map[b, 0] * boost_factor
                    else:
                        print(f"    ERROR: best_idx={bi} >= len(masks)={len(masks[b])}")

                # Use box_head to predict from boosted score_map directly (no Hann, mask already isolates target)
                response = boosted_score_map
                vis_score_map = boosted_score_map[0, 0].cpu().numpy()
                # Debug: check boosted_score_map peak position
                boosted_max = boosted_score_map[0, 0].max().item()
                boosted_max_idx = boosted_score_map[0, 0].argmax().item()
                boosted_max_y = boosted_max_idx // self.feat_sz
                boosted_max_x = boosted_max_idx % self.feat_sz
                print(f"  [Boost Debug] boosted_max={boosted_max:.4f} at ({boosted_max_y},{boosted_max_x}), region1 center=(11,14)")
                pred_boxes = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'])
                pred_boxes = pred_boxes.view(-1, 4)
                pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()
                self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        # Print Stage2 decision result
        if interference_warning and self.use_multicandidate and self.prev_search_crop is not None:
            if 'centers' in locals() and len(centers[0]) > 0:
                best_region_center = centers[0][best_idx_per_batch[0]]
                print(f"  [Stage2 Decision] selected={best_idx_per_batch[0]}, center={best_region_center}, final_state={self.state}")
                # Save selected region center and movement for position consistency
                curr_sel_y = float(best_region_center[0])
                curr_sel_x = float(best_region_center[1])
                prev_sel_y = getattr(self, '_selected_center_y', curr_sel_y)
                prev_sel_x = getattr(self, '_selected_center_x', curr_sel_x)
                sel_movement = math.sqrt((curr_sel_y - prev_sel_y)**2 + (curr_sel_x - prev_sel_x)**2)
                self._selected_center_y = curr_sel_y
                self._selected_center_x = curr_sel_x
                self._prev_selected_movement = sel_movement
                print(f"  [Stage2 Decision Debug] state={self.state}, sel_movement={sel_movement:.2f}")
        else:
            # Normal flow: use box_head to predict from response
            pred_boxes = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'])
            pred_boxes = pred_boxes.view(-1, 4)
            pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()
            raw_state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

            # Check if box size is abnormally large compared to recent history
            raw_w, raw_h = raw_state[2], raw_state[3]
            box_size_threshold = 1.5  # 50% larger than average is abnormal
            use_normal_flow = True
            force_normal = False

            if len(self.box_size_history) >= 3:
                avg_w = sum(w for w, _ in self.box_size_history) / len(self.box_size_history)
                avg_h = sum(h for _, h in self.box_size_history) / len(self.box_size_history)
                if raw_w > avg_w * box_size_threshold or raw_h > avg_h * box_size_threshold:
                    print(f"  [Box Size Check] abnormal box detected: {raw_w:.1f}x{raw_h:.1f} vs avg {avg_w:.1f}x{avg_h:.1f}")
                    # Also check consecutive prediction limit
                    if self.consecutive_predictions >= self.max_consecutive_predictions:
                        print(f"  [Box Size Check] consecutive predictions limit reached ({self.consecutive_predictions}), forcing normal flow")
                        force_normal = True
                    else:
                        use_normal_flow = False

            if force_normal:
                # Consecutive prediction limit reached - force region selection
                # Find connected regions and pick the one with highest score
                peaks, peak_scores = self._findLocalPeaks(pred_score_map, min_distance=3)
                centers, region_scores_list, masks = self._regionGrowing(
                    pred_score_map, peaks, peak_scores, threshold_fraction=0.3)
                print(f"  [Force Normal] num_regions={len(centers[0])}")

                # Select region with highest score
                best_region_idx = 0
                best_score = region_scores_list[0][0]
                for i, score in enumerate(region_scores_list[0]):
                    if score > best_score:
                        best_score = score
                        best_region_idx = i
                print(f"  [Force Normal] selected region {best_region_idx}, score={best_score:.3f}, center=({centers[0][best_region_idx]})")

                # Boost the selected region
                boosted_score_map = torch.zeros_like(pred_score_map)
                best_mask = torch.from_numpy(masks[0][best_region_idx].astype(np.float32)).to(pred_score_map.device)
                boosted_score_map[0, 0] = best_mask * pred_score_map[0, 0] * 3.0

                response = boosted_score_map
                vis_score_map = boosted_score_map[0, 0].cpu().numpy()

                pred_boxes = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'])
                pred_boxes = pred_boxes.view(-1, 4)
                pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()
                self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)
                print(f"  [Force Normal] new state={self.state}")

                self.consecutive_predictions = 0
                # Skip box_size_history update to preserve correct history
            elif not use_normal_flow:
                # Box is abnormally large - likely target and干扰物 merged
                # Use position history to predict displacement, like occlusion mode
                print(f"  [Box Size Fix] abnormal box detected, using position prediction")
                self.consecutive_predictions += 1
                print(f"  [Box Size Fix] consecutive_predictions={self.consecutive_predictions}")

                # Get recent box size (average of history)
                if len(self.box_size_history) >= 3:
                    ref_w = sum(w for w, _ in self.box_size_history) / len(self.box_size_history)
                    ref_h = sum(h for _, h in self.box_size_history) / len(self.box_size_history)
                else:
                    ref_w, ref_h = raw_w, raw_h

                # Predict displacement from position history
                pred_dx, pred_dy = self._predict_displacement()
                x1, y1, w, h = self.state
                pred_x = x1 + pred_dx
                pred_y = y1 + pred_dy
                print(f"  [Box Size Fix] After pred: pred_x={pred_x:.2f}, pred_y={pred_y:.2f}, size={ref_w:.1f}x{ref_h:.1f}")

                self.state = [pred_x, pred_y, ref_w, ref_h]
                self.state = clip_box(self.state, H, W, margin=10)
                vis_score_map = pred_score_map[0, 0].cpu().numpy()

                # Update box size history
                self.box_size_history.append((self.state[2], self.state[3]))
                if len(self.box_size_history) > 3:
                    self.box_size_history.pop(0)
            else:
                # Normal flow - reset counter and update history
                self.consecutive_predictions = 0
                self.state = raw_state
                vis_score_map = pred_score_map[0, 0].cpu().numpy()

                # Update box size history
                self.box_size_history.append((self.state[2], self.state[3]))
                if len(self.box_size_history) > 3:
                    self.box_size_history.pop(0)

        # Always draw debug visualizations (file-based, not visdom)
        x1, y1, w, h = self.state
        image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.rectangle(image_BGR, (int(x1),int(y1)), (int(x1+w),int(y1+h)), color=(0,0,255), thickness=2)

        # Draw Hann window in bottom-left corner (adaptive to image size)
        hann_vis = self.output_window[0].cpu().numpy()
        hann_vis = hann_vis.squeeze()
        hann_size = min(100, image_BGR.shape[0] - 10, image_BGR.shape[1] - 10)  # Adaptive size, leave margin
        if hann_size > 20:  # Only draw if enough space
            hann_vis_resized = cv2.resize(hann_vis, (hann_size, hann_size))
            hann_vis_resized = (hann_vis_resized * 255).astype(np.uint8)
            hann_color = cv2.applyColorMap(hann_vis_resized, cv2.COLORMAP_JET)
            y1_hann = image_BGR.shape[0] - hann_size
            x1_hann = 0
            image_BGR[y1_hann:y1_hann+hann_size, x1_hann:x1_hann+hann_size] = hann_color

        # Save to debug folder if save_dir exists
        if hasattr(self, 'save_dir') and self.save_dir:
            import os
            save_path = os.path.abspath(os.path.join(self.save_dir, "%04d.jpg" % self.frame_id))
            cv2.imwrite(save_path, image_BGR)
        # visdom mode (only if debug > 0)
        if hasattr(self, 'debug') and self.debug and hasattr(self, 'use_visdom') and self.use_visdom:
            self.visdom.register((image, info['gt_bbox'].tolist(), self.state), 'Tracking', 1, 'Tracking')
            self.visdom.register(torch.from_numpy(x_patch_arr).permute(2, 0, 1), 'image', 1, 'search_region')
            self.visdom.register(torch.from_numpy(self.z_patch_arr).permute(2, 0, 1), 'image', 1, 'template')
            self.visdom.register(pred_score_map.view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map')
            self.visdom.register((pred_score_map * self.output_window).view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map_hann')
            if 'removed_indexes_s' in out_dict and out_dict['removed_indexes_s']:
                removed_indexes_s = out_dict['removed_indexes_s']
                removed_indexes_s = [removed_indexes_s_i.cpu().numpy() for removed_indexes_s_i in removed_indexes_s]
                masked_search = gen_visualization(x_patch_arr, removed_indexes_s)
                self.visdom.register(torch.from_numpy(masked_search).permute(2, 0, 1), 'image', 1, 'masked_search')
            while self.pause_mode:
                if self.step:
                    self.step = False
                    break

        # Save current search crop and state for next frame's prev_target feature
        self.prev_search_crop = x_patch_arr.copy()
        self.prev_state = list(self.state)

        # Update prev_raw_y, prev_raw_x, prev_raw_movement for next frame's warning check
        # Use selected region center (Stage2) or raw peak (normal) for movement calculation
        if hasattr(self, '_selected_center_y'):
            if skip_boosting:
                # Single region case: use Hann-modulated peak position, not raw peak
                resp_max_idx = response.flatten(1).argmax(dim=1)
                resp_y = resp_max_idx[0].item() // self.feat_sz
                resp_x = resp_max_idx[0].item() % self.feat_sz
                prev_raw_y = getattr(self, '_prev_raw_y', resp_y)
                prev_raw_x = getattr(self, '_prev_raw_x', resp_x)
                curr_movement = math.sqrt((resp_y - prev_raw_y)**2 + (resp_x - prev_raw_x)**2)
                self._prev_raw_y = resp_y
                self._prev_raw_x = resp_x
                self._prev_raw_movement = curr_movement
                print(f"  [Skip Boost] Using resp peak ({resp_y},{resp_x}) for prev_raw update")
            else:
                # Stage2 boosting: use selected region center
                selected_y = self._selected_center_y
                selected_x = self._selected_center_x
                prev_raw_y = getattr(self, '_prev_raw_y', selected_y)
                prev_raw_x = getattr(self, '_prev_raw_x', selected_x)
                self._prev_raw_y = selected_y
                self._prev_raw_x = selected_x
                curr_movement = 0.0  # Placeholder, not used
            # Clear for next frame
            delattr(self, '_selected_center_y')
            delattr(self, '_selected_center_x')
        else:
            # Normal flow: use raw_y/x
            prev_raw_y = getattr(self, '_prev_raw_y', raw_y)
            prev_raw_x = getattr(self, '_prev_raw_x', raw_x)
            curr_movement = math.sqrt((raw_y - prev_raw_y)**2 + (raw_x - prev_raw_x)**2)
            self._prev_raw_y = raw_y
            self._prev_raw_x = raw_x
            self._prev_raw_movement = curr_movement

        # Update position history for occlusion prediction
        # Use pixel coordinates (state center) instead of feature map coordinates
        confidence_threshold = 0.1
        should_update_history = False

        if hasattr(self, '_selected_center_y'):
            # Stage2 selected region: check combined score
            if len(combined_list) > 0 and combined_list[0].numel() > 0:
                if combined_list[0].max().item() > confidence_threshold:
                    should_update_history = True
        else:
            # Normal flow - always update
            should_update_history = True

        if should_update_history:
            # Current state center in pixel coordinates
            curr_cx = self.state[0] + self.state[2] / 2
            curr_cy = self.state[1] + self.state[3] / 2

            # Previous state center
            if hasattr(self, '_prev_state') and self._prev_state is not None:
                prev_cx = self._prev_state[0] + self._prev_state[2] / 2
                prev_cy = self._prev_state[1] + self._prev_state[3] / 2
            else:
                prev_cx = curr_cx
                prev_cy = curr_cy

            dx = curr_cx - prev_cx
            dy = curr_cy - prev_cy

            if len(self.position_history) > 0 or (abs(dx) > 0.5 or abs(dy) > 0.5):
                self.position_history.append((dx, dy))
                if len(self.position_history) > 10:
                    self.position_history.pop(0)
            print(f"  [History Update] dx={dx:.2f}, dy={dy:.2f}, history_len={len(self.position_history)}")
            self._prev_state = list(self.state)
        else:
            print(f"  [History Update] Skipped - low confidence")

        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save,
                    "score_map": pred_score_map,
                    "response": response,
                    "prev_state": prev_state,
                    "resize_factor": resize_factor,
                    "search_crop": x_patch_arr}
        else:
            return {"target_bbox": self.state,
                    "score_map": pred_score_map,
                    "response": response,
                    "prev_state": prev_state,
                    "resize_factor": resize_factor,
                    "search_crop": x_patch_arr}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

    def add_hook(self):
        conv_features, enc_attn_weights, dec_attn_weights = [], [], []

        for i in range(12):
            self.network.backbone.blocks[i].attn.register_forward_hook(
                # lambda self, input, output: enc_attn_weights.append(output[1])
                lambda self, input, output: enc_attn_weights.append(output[1])
            )

        self.enc_attn_weights = enc_attn_weights


def get_tracker_class():
    return OSTrack
