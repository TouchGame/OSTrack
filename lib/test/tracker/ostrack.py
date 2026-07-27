import math
import heapq
import numpy as np

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
from lib.utils.misc import NestedTensor


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

        # ── Unidirectional Attention (LMTrack-style) ─────────────────────────────
        # Configured inside build_ostrack from cfg.MODEL.BACKBONE.UNIDIRECTIONAL,
        # so training and testing share the same construction path.
        if getattr(params.cfg.MODEL.BACKBONE, 'UNIDIRECTIONAL', False):
            print("[OSTrack] Unidirectional attention ENABLED")
        else:
            print("[OSTrack] Unidirectional attention disabled (default)")

        # for debug
        self.debug = params.debug
        self.use_visdom = params.debug
        self.frame_id = 0
        # File-based visualization only in debug mode (imwrite per frame is expensive)
        self.save_dir = "debug"
        if self.debug:
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            if self.use_visdom:
                self._init_visdom(None, 1)
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        self.z_dict1 = {}

        # for multi-candidate decision
        self.use_multicandidate = True

        # for consecutive prediction limit
        self.max_consecutive_predictions = 8  # max allowed consecutive predictions

        # ── TCM: Token Context Memory (LMTrack-style dynamic template) ──────────
        self.ref_pool_enabled = getattr(params.cfg.MODEL.BACKBONE, 'REF_POOL', False)
        self.feat_dim = self.network.backbone.embed_dim
        self._z_side = self.cfg.TEST.TEMPLATE_SIZE // self.cfg.MODEL.BACKBONE.STRIDE  # 12 for 192
        self._z_len = self._z_side * self._z_side                                     # 144
        self._pool_size = 16          # cap on high-value search tokens per frame
        self._ema_alpha = 0.9         # EMA for token pool update (higher = keep more old)
        self._tcm_conf_thresh = 0.3   # min score-map confidence to allow a pool update
                                      # (typical conf is only ~0.4 even on easy frames;
                                      #  0.5 would silently disable TCM entirely)
        self._tcm_rel_drop = 0.0      # pause updates when conf < this frac of running avg
                                      # (0.0 disables the relative gate: it locks out the
                                      #  deformation frames TCM exists to memorize)
        self._tcm_score_ratio = 0.0   # in-box tokens must score >= this frac of the
                                      # in-box peak (0.0 = plain in-box top-k; 0.5
                                      #  proved too strict and starved pool updates)
        self._tcm_blend_alpha = 0.15  # pool influence when blending into the template
        # search-grid → template-grid scale: both crops are target-centered, but
        # sampled with different context factors, so token offsets must be rescaled
        self._s2t_scale = (self.params.template_size / self.params.template_factor) / \
                          (self.params.search_size / self.params.search_factor)

    def _dprint(self, *args, **kwargs):
        """Debug printing, silenced unless params.debug is set."""
        if self.debug:
            print(*args, **kwargs)

    def initialize(self, image, info: dict):
        # ── Template Initialization ─────────────────────────────────────────────
        z_patch_arr, resize_factor, z_amask_arr = sample_target(
            image, info['init_bbox'], self.params.template_factor, output_sz=self.params.template_size)
        self.z_patch_arr = z_patch_arr
        template = self.preprocessor.process(z_patch_arr, z_amask_arr)

        # Store template in IMAGE format; TCM replaces it with token format later
        self.z_dict1 = template

        self.box_mask_z = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            template_bbox = self.transform_bbox_to_crop(info['init_bbox'], resize_factor,
                                                        template.tensors.device).squeeze(1)
            self.box_mask_z = generate_mask_cond(self.cfg, 1, template.tensors.device, template_bbox)

        # ── TCM: cache block-0 INPUT-space template tokens ──────────────────────
        # patch_embed(z) (+ pos_embed_z) is the space the backbone expects when
        # template_is_tokens=True. Never cache final-layer output features here.
        self._template_raw = None     # [1, 144, C] patch embeddings, pos-free anchor
        self._template_tokens = None  # [1, 144, C] anchor + pos_embed_z
        self._token_pool = None       # [1, 144, C] per-slot EMA buffer (pos-free)
        if self.ref_pool_enabled:
            with torch.no_grad():
                z_raw = self.network.backbone.patch_embed(template.tensors)
                self._template_raw = z_raw.clone()
                self._template_tokens = z_raw + self.network.backbone.pos_embed_z
                self._token_pool = z_raw.clone()
            print(f'[Template] cached tokens shape={tuple(self._template_raw.shape)}')
        # TCM gating state (per sequence)
        self._conf_avg = None         # running avg of gate-passing confidences
        self._prev_tcm_area = None    # last predicted box area, for jump detection

        # ── reset per-sequence state ─────────────────────────────────────────────
        self.prev_search_crop = None
        self.prev_state = None
        self._prev_state = None
        self.position_history = []      # [(dx, dy), ...] in pixel coordinates
        self.box_size_history = []      # [(w, h), ...] recent 3 frames
        self.consecutive_predictions = 0
        for attr in ('_prev_raw_y', '_prev_raw_x', '_prev_raw_movement', '_interference_warning'):
            if hasattr(self, attr):
                delattr(self, attr)

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

        # Normalize candidate features. NOTE: candidates at CE-eliminated positions
        # come back as zero vectors (recover_tokens zero-pads); normalize() keeps
        # them at zero, so their similarity is 0 rather than NaN.
        candidate_features_norm = torch.nn.functional.normalize(candidate_features, p=2, dim=-1)  # [B, K, C]

        # Cosine similarity: [B, K, C] * [B, 1, C] -> [B, K]
        similarities = (candidate_features_norm * prev_feat_normalized.unsqueeze(1)).sum(dim=-1)

        self._dprint(f"  [Sim Debug] token_indices={token_indices}, similarities={similarities}")

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
            self._dprint(f"  [Predict] Not enough history, returning 0")
            return 0.0, 0.0

        # Use recent N frames average displacement
        n = min(5, len(self.position_history))
        recent = self.position_history[-n:]
        avg_dx = sum(d[0] for d in recent) / n
        avg_dy = sum(d[1] for d in recent) / n
        self._dprint(f"  [Predict] history={recent}, avg_dx={avg_dx:.2f}, avg_dy={avg_dy:.2f}")
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

            # Maximum bbox size: default to 30% of the feature map
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

            if self.debug:
                for ri, (center, mask) in enumerate(zip(centers_b, masks_b)):
                    mask_y, mask_x = np.where(mask)
                    if len(mask_y) > 0:
                        self._dprint(f"    [RegionGrowing] Region {ri}: center=({center[0]},{center[1]}), "
                                     f"mask y=[{mask_y.min()},{mask_y.max()}], x=[{mask_x.min()},{mask_x.max()}]")

            all_centers.append(centers_b)
            all_scores.append(scores_out_b)
            all_masks.append(masks_b)

        return all_centers, all_scores, all_masks

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

    def _draw_debug_frame(self, image, state, color=(0, 0, 255)):
        """Draw bbox + Hann window inset and save to self.save_dir (debug only)."""
        if not self.debug:
            return
        x1, y1, w, h = state
        image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.rectangle(image_BGR, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color=color, thickness=2)
        hann_vis = self.output_window[0].cpu().numpy().squeeze()
        hann_size = min(100, image_BGR.shape[0] - 10, image_BGR.shape[1] - 10)
        if hann_size > 20:
            hann_vis_resized = cv2.resize(hann_vis, (hann_size, hann_size))
            hann_vis_resized = (hann_vis_resized * 255).astype(np.uint8)
            hann_color = cv2.applyColorMap(hann_vis_resized, cv2.COLORMAP_JET)
            y1_hann = image_BGR.shape[0] - hann_size
            image_BGR[y1_hann:y1_hann + hann_size, 0:hann_size] = hann_color
        save_path = os.path.abspath(os.path.join(self.save_dir, "%04d.jpg" % self.frame_id))
        cv2.imwrite(save_path, image_BGR)

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        prev_state = self.state
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)

        with torch.no_grad():
            x_dict = search
            # Token-format template only exists after TCM produced it ([1, N, C]).
            # Falls back to the image template otherwise (first frame, REF_POOL off,
            # or the TCM update never passed its confidence gate) — never crashes.
            is_tokens = torch.is_tensor(self.z_dict1.tensors) and self.z_dict1.tensors.dim() == 3
            out_dict = self.network.forward(
                template=self.z_dict1.tensors, search=x_dict.tensors,
                ce_template_mask=self.box_mask_z,
                template_is_tokens=is_tokens)

        # add hann windows
        pred_score_map = out_dict['score_map']
        response = self.output_window * pred_score_map

        # ============================================================
        # Stage 1: Interference Warning
        # ============================================================
        raw_max_idx = pred_score_map.flatten(1).argmax(dim=1)  # [B], flat index
        resp_max_idx = response.flatten(1).argmax(dim=1)       # [B], flat index

        raw_y = int(raw_max_idx[0].item() // self.feat_sz)
        raw_x = int(raw_max_idx[0].item() % self.feat_sz)

        interference_warning = (raw_max_idx != resp_max_idx).any().item()
        feat_center = self.feat_sz // 2
        # Also trigger warning if raw peak is far from expected center
        dist_to_center = math.sqrt((raw_y - feat_center) ** 2 + (raw_x - feat_center) ** 2)
        if not interference_warning and dist_to_center > 3.0:
            interference_warning = True
            self._dprint(f'  [Warning] raw peak ({raw_y},{raw_x}) far from center {feat_center}, dist={dist_to_center:.2f}')
        # Also trigger if the raw peak jumped far compared to its recent movement.
        # The absolute floor (2.0) stops near-static targets from triggering on
        # 1-pixel jitter (prev_movement ~ 0 would otherwise flag everything).
        if not interference_warning:
            prev_raw_y = getattr(self, '_prev_raw_y', raw_y)
            prev_raw_x = getattr(self, '_prev_raw_x', raw_x)
            prev_movement = getattr(self, '_prev_raw_movement', 0.0)
            dist = math.sqrt((raw_y - prev_raw_y) ** 2 + (raw_x - prev_raw_x) ** 2)
            if dist > max(2 * prev_movement, 2.0):
                interference_warning = True
                self._dprint(f'  [Warning] dist={dist:.2f} > max(2*prev_mv, 2.0), possible interference')

        self._interference_warning = interference_warning

        resp_y = int(resp_max_idx[0].item() // self.feat_sz)
        resp_x = int(resp_max_idx[0].item() % self.feat_sz)
        self._dprint(f'Frame {self.frame_id}: raw=({raw_y},{raw_x}), resp=({resp_y},{resp_x}), warning={interference_warning}')

        # Stage-2 bookkeeping (locals, so the post-hoc history gate actually works)
        stage2_ran = False
        stage2_confident = True
        skip_boosting = False
        selected_center = None

        # ============================================================
        # Stage 2: Multi-Candidate Decision (only when warning triggers)
        # ============================================================
        if interference_warning and self.use_multicandidate and self.prev_search_crop is not None:
            self._dprint(f'[Stage1] Interference warning triggered at frame {self.frame_id}')
            # Find connected regions using local peaks + region growing
            max_score = pred_score_map.max().item()
            peaks, peak_scores = self._findLocalPeaks(pred_score_map, min_distance=3)
            centers, region_scores_list, masks = self._regionGrowing(
                pred_score_map, peaks, peak_scores, threshold_fraction=0.3)
            self._dprint(f"  [Stage2] max_score={max_score:.4f}, num_regions={len(centers[0])}")

            # Skip boosting if only one region found (no real interference)
            skip_boosting = len(centers[0]) == 1
            if skip_boosting:
                self._dprint(f"  [Stage2] Only one region found, skipping boosting")

            # Compute prev_frame target features for similarity
            lens_z = self.network.backbone.pos_embed_z.shape[1]
            lens_x = self.network.backbone.pos_embed_x.shape[1]

            # Crop previous frame's target from prev_search_crop:
            # the crop is search_size x search_size with the target at its center;
            # extract a template_size x template_size patch around the center
            search_sz = self.params.search_size      # e.g. 384
            template_sz = self.params.template_size  # e.g. 192
            center = search_sz // 2
            half = template_sz // 2
            prev_target_crop_np = self.prev_search_crop[center - half:center + half, center - half:center + half]
            prev_target_amask = np.ones((template_sz, template_sz), dtype=bool)
            prev_target_tensor = self.preprocessor.process(prev_target_crop_np, prev_target_amask)

            # Also prepare prev_search_crop as x (search region) for backbone
            prev_search_amask = np.ones(self.prev_search_crop.shape[:2], dtype=bool)
            prev_search_tensor = self.preprocessor.process(self.prev_search_crop, prev_search_amask)

            # Get backbone features: z=template crop, x=full previous search region
            with torch.no_grad():
                prev_feat_out, _ = self.network.backbone(
                    z=prev_target_tensor.tensors, x=prev_search_tensor.tensors)
                prev_feat = prev_feat_out[:, :lens_z, :]
                # Average only the CENTRAL template tokens (target area). A full-crop
                # mean is dominated by background (target covers ~1/4 of the crop),
                # which flattens similarity differences between candidates.
                t_side = int(math.sqrt(lens_z))
                c0 = t_side // 2 - t_side // 4
                c1 = t_side // 2 + t_side // 4
                center_feat = prev_feat.view(1, t_side, t_side, -1)[:, c0:c1, c0:c1, :]
                center_feat = center_feat.reshape(1, -1, prev_feat.shape[-1]).mean(dim=1)  # [1, C]
                prev_feat_normalized = torch.nn.functional.normalize(center_feat, p=2, dim=-1)

            # Get current frame search tokens
            backbone_feat = out_dict['backbone_feat']  # [B, Lz+Lx, C]
            search_tokens = backbone_feat[:, -lens_x:, :]  # [B, Lx, C]

            # Compute similarity for each region
            combined_list = []
            raw_sims_list = []

            # Use feature map center as reference since each frame is cropped around target
            sigma = 2.0  # fixed sigma

            for b in range(pred_score_map.shape[0]):
                centers_b = centers[b]
                scores_b = torch.tensor(region_scores_list[b], dtype=torch.float32, device=pred_score_map.device)

                if len(centers_b) == 0:
                    combined_list.append(torch.tensor([0.0]))
                    continue

                # Compute similarity for each region
                positions_tensor = torch.tensor(centers_b, dtype=torch.long, device=pred_score_map.device).unsqueeze(0)  # [1, K, 2]
                sims = self._compute_cosine_similarity(search_tokens[b:b + 1], positions_tensor, prev_feat_normalized[b:b + 1])  # [1, K]
                raw_sims_list.append(sims.squeeze(0).cpu().numpy())

                # Compute position consistency score (Gaussian distance penalty)
                position_scores = []
                for cy, cx in centers_b:
                    dist = math.sqrt((cy - feat_center) ** 2 + (cx - feat_center) ** 2)
                    pos_score = math.exp(-dist ** 2 / (2 * sigma ** 2))
                    position_scores.append(pos_score)
                position_scores_tensor = torch.tensor(position_scores, dtype=torch.float32, device=pred_score_map.device)

                # Combined: score × similarity × position_score
                combined_b = scores_b.unsqueeze(0) * sims * position_scores_tensor.unsqueeze(0)  # [1, K]
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

            if self.debug:
                self._dprint(f"  [Stage2] regions={len(centers[0])}, centers={centers[0]}, "
                             f"scores={[f'{s:.3f}' for s in region_scores_list[0]]}, best_idx={best_idx_per_batch[0]}")
                for b in range(pred_score_map.shape[0]):
                    raw_sims_b = raw_sims_list[b] if b < len(raw_sims_list) else []
                    for ri, (cy, cx) in enumerate(centers[b]):
                        raw_sim = raw_sims_b[ri] if ri < len(raw_sims_b) else 0.0
                        combined_val = combined_list[b][ri].item() if ri < len(combined_list[b]) else 0.0
                        self._dprint(f"      Region {ri}: center=({cy},{cx}), score={region_scores_list[b][ri]:.3f}, "
                                     f"sim={raw_sim:.3f}, combined={combined_val:.3f}")

            # Occlusion detection: check if all combined scores are very low
            max_combined = 0.0
            for b in range(pred_score_map.shape[0]):
                if len(combined_list[b]) > 0 and combined_list[b].numel() > 0:
                    max_combined = max(max_combined, combined_list[b].max().item())
            low_confidence_threshold = 0.1

            # Check consecutive prediction limit before entering prediction mode
            if self.consecutive_predictions >= self.max_consecutive_predictions:
                self._dprint(f"  [Occlusion] Consecutive predictions limit reached ({self.consecutive_predictions}), skipping prediction")
                skip_prediction = True
            elif max_combined < low_confidence_threshold:
                self._dprint(f"  [Occlusion] All regions low combined score ({max_combined:.3f} < {low_confidence_threshold}), predicting position")
                skip_prediction = False
                self.consecutive_predictions += 1
            else:
                skip_prediction = True

            if not skip_prediction:
                # ── Occlusion mode: dead-reckon from position history ──
                response = self.output_window * pred_score_map
                pred_dx, pred_dy = self._predict_displacement()
                x1, y1, w, h = self.state
                pred_x = x1 + pred_dx
                pred_y = y1 + pred_dy
                # Keep the predicted box inside the image (was previously unclipped)
                self.state = clip_box([pred_x, pred_y, w, h], H, W, margin=10)
                self._dprint(f"  [Occlusion] predicted state={self.state}")
                self._draw_debug_frame(image, self.state)
                self._prev_state = list(self.state)
                self.prev_search_crop = x_patch_arr.copy()
                self.prev_state = list(self.state)
                return {"target_bbox": self.state,
                        "response": response.squeeze(0).cpu(),
                        "score_map": pred_score_map[0, 0].cpu(),
                        "prev_state": prev_state,
                        "resize_factor": resize_factor,
                        "search_crop": x_patch_arr}

            # Build boosted score_map using best region mask
            if skip_boosting:
                # Single region - use normal response (Hann modulated)
                response = self.output_window * pred_score_map
                pred_boxes = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'])
                pred_boxes = pred_boxes.view(-1, 4)
                pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()
                self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)
            else:
                boosted_score_map = torch.zeros_like(pred_score_map)
                boost_factor = 3.0
                for b in range(pred_score_map.shape[0]):
                    bi = best_idx_per_batch[b]
                    if bi < len(masks[b]):
                        best_mask = torch.from_numpy(masks[b][bi].astype(np.float32)).to(pred_score_map.device)
                        # Only selected region has non-zero scores
                        boosted_score_map[b, 0] = best_mask * pred_score_map[b, 0] * boost_factor
                    else:
                        self._dprint(f"    ERROR: best_idx={bi} >= len(masks)={len(masks[b])}")

                # Predict from boosted score_map directly (no Hann, mask already isolates target)
                response = boosted_score_map
                pred_boxes = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'])
                pred_boxes = pred_boxes.view(-1, 4)
                pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()
                self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

            # Stage-2 decision bookkeeping
            stage2_ran = True
            if len(centers[0]) > 0:
                best_region_center = centers[0][best_idx_per_batch[0]]
                selected_center = (float(best_region_center[0]), float(best_region_center[1]))
                self._dprint(f"  [Stage2 Decision] selected={best_idx_per_batch[0]}, center={best_region_center}, state={self.state}")
            stage2_confident = (len(combined_list) > 0 and combined_list[0].numel() > 0
                                and combined_list[0].max().item() > low_confidence_threshold)
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
                    self._dprint(f"  [Box Size Check] abnormal box: {raw_w:.1f}x{raw_h:.1f} vs avg {avg_w:.1f}x{avg_h:.1f}")
                    if self.consecutive_predictions >= self.max_consecutive_predictions:
                        self._dprint(f"  [Box Size Check] consecutive predictions limit reached, forcing normal flow")
                        force_normal = True
                    else:
                        use_normal_flow = False

            if force_normal:
                # Consecutive prediction limit reached - force region selection
                peaks, peak_scores = self._findLocalPeaks(pred_score_map, min_distance=3)
                centers, region_scores_list, masks = self._regionGrowing(
                    pred_score_map, peaks, peak_scores, threshold_fraction=0.3)

                # Select region with highest score
                best_region_idx = 0
                best_score = region_scores_list[0][0]
                for i, score in enumerate(region_scores_list[0]):
                    if score > best_score:
                        best_score = score
                        best_region_idx = i
                self._dprint(f"  [Force Normal] selected region {best_region_idx}, score={best_score:.3f}")

                # Boost the selected region
                boosted_score_map = torch.zeros_like(pred_score_map)
                best_mask = torch.from_numpy(masks[0][best_region_idx].astype(np.float32)).to(pred_score_map.device)
                boosted_score_map[0, 0] = best_mask * pred_score_map[0, 0] * 3.0

                response = boosted_score_map

                pred_boxes = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'])
                pred_boxes = pred_boxes.view(-1, 4)
                pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()
                self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

                self.consecutive_predictions = 0
                # Skip box_size_history update to preserve correct history
            elif not use_normal_flow:
                # Box abnormally large - likely target and distractor merged.
                # Use position history to predict displacement, like occlusion mode.
                self.consecutive_predictions += 1
                self._dprint(f"  [Box Size Fix] abnormal box, consecutive_predictions={self.consecutive_predictions}")

                # Get recent box size (average of history)
                if len(self.box_size_history) >= 3:
                    ref_w = sum(w for w, _ in self.box_size_history) / len(self.box_size_history)
                    ref_h = sum(h for _, h in self.box_size_history) / len(self.box_size_history)
                else:
                    ref_w, ref_h = raw_w, raw_h

                # Predict displacement from position history
                pred_dx, pred_dy = self._predict_displacement()
                x1, y1, w, h = self.state
                self.state = clip_box([x1 + pred_dx, y1 + pred_dy, ref_w, ref_h], H, W, margin=10)

                # Update box size history
                self.box_size_history.append((self.state[2], self.state[3]))
                if len(self.box_size_history) > 3:
                    self.box_size_history.pop(0)
            else:
                # Normal flow - reset counter and update history
                self.consecutive_predictions = 0
                self.state = raw_state

                # Update box size history
                self.box_size_history.append((self.state[2], self.state[3]))
                if len(self.box_size_history) > 3:
                    self.box_size_history.pop(0)

        # Debug visualizations (file-based; only when debug is enabled)
        self._draw_debug_frame(image, self.state)
        if self.debug and self.use_visdom:
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

        # ── movement bookkeeping for the next frame's warning check ──
        if stage2_ran:
            if skip_boosting or selected_center is None:
                # Single region: use Hann-modulated peak position, not raw peak
                resp_idx = response.flatten(1).argmax(dim=1)
                sel_y = int(resp_idx[0].item() // self.feat_sz)
                sel_x = int(resp_idx[0].item() % self.feat_sz)
            else:
                sel_y, sel_x = selected_center
            prev_ry = getattr(self, '_prev_raw_y', sel_y)
            prev_rx = getattr(self, '_prev_raw_x', sel_x)
            self._prev_raw_movement = math.sqrt((sel_y - prev_ry) ** 2 + (sel_x - prev_rx) ** 2)
            self._prev_raw_y = sel_y
            self._prev_raw_x = sel_x
        else:
            prev_ry = getattr(self, '_prev_raw_y', raw_y)
            prev_rx = getattr(self, '_prev_raw_x', raw_x)
            self._prev_raw_movement = math.sqrt((raw_y - prev_ry) ** 2 + (raw_x - prev_rx) ** 2)
            self._prev_raw_y = raw_y
            self._prev_raw_x = raw_x

        # ── position history update (skipped when Stage-2 result is low-confidence,
        #     so drifting predictions don't pollute the occlusion motion model) ──
        should_update_history = (not stage2_ran) or stage2_confident
        if should_update_history:
            curr_cx = self.state[0] + self.state[2] / 2
            curr_cy = self.state[1] + self.state[3] / 2
            if self._prev_state is not None:
                prev_cx = self._prev_state[0] + self._prev_state[2] / 2
                prev_cy = self._prev_state[1] + self._prev_state[3] / 2
            else:
                prev_cx, prev_cy = curr_cx, curr_cy
            dx = curr_cx - prev_cx
            dy = curr_cy - prev_cy
            if len(self.position_history) > 0 or (abs(dx) > 0.5 or abs(dy) > 0.5):
                self.position_history.append((dx, dy))
                if len(self.position_history) > 10:
                    self.position_history.pop(0)
            self._prev_state = list(self.state)
        else:
            self._dprint(f"  [History Update] Skipped - low confidence")

        # ============================================================
        # Stage 3: TCM — Token Context Memory
        # Blend high-response search tokens into the template, all in the
        # block-0 INPUT space (patch embeddings), anchored to the first frame.
        # ============================================================
        if self.ref_pool_enabled and self._template_raw is not None:
            conf = pred_score_map.max().item()
            # ── Quality gates: never memorize a frame we are not sure about ──
            # 1) absolute floor + RELATIVE drop vs the running average of past
            #    accepted frames: occlusion onset shows up as a sudden relative
            #    drop long before conf reaches any absolute threshold.
            conf_ok = conf > self._tcm_conf_thresh and \
                (self._conf_avg is None or conf > self._tcm_rel_drop * self._conf_avg)
            # 2) box-area jump guard: a sudden size change means the box latched
            #    onto something else; skip that frame.
            area = self.state[2] * self.state[3]
            area_ok = self._prev_tcm_area is None or \
                0.5 * self._prev_tcm_area < area < 2.0 * self._prev_tcm_area
            self._prev_tcm_area = area
            if conf_ok and area_ok and not interference_warning:
                # running average only tracks accepted frames, so it stays high
                # during occlusion and keeps the gate shut until quality returns
                self._conf_avg = conf if self._conf_avg is None else \
                    0.95 * self._conf_avg + 0.05 * conf
                with torch.no_grad():
                    # Fresh patch-embed of the current search crop: same input space
                    # as the cached template tokens, and immune to CE zero-padding.
                    search_raw = self.network.backbone.patch_embed(x_dict.tensors)  # [1, Lx, C]
                # Restrict candidates to the predicted box on the search grid, so
                # background/distractor tokens outside the target never enter the
                # pool (a global top-k pulls in high-response background around
                # the peak and slowly dilutes the template on easy sequences).
                stride = self.cfg.MODEL.BACKBONE.STRIDE
                half_crop = 0.5 * self.params.search_size
                p_cx = prev_state[0] + 0.5 * prev_state[2]   # crop center (image coords)
                p_cy = prev_state[1] + 0.5 * prev_state[3]
                gx1 = int(math.floor(((self.state[0] - p_cx) * resize_factor + half_crop) / stride))
                gy1 = int(math.floor(((self.state[1] - p_cy) * resize_factor + half_crop) / stride))
                gx2 = int(math.ceil(((self.state[0] + self.state[2] - p_cx) * resize_factor + half_crop) / stride))
                gy2 = int(math.ceil(((self.state[1] + self.state[3] - p_cy) * resize_factor + half_crop) / stride))
                gx1, gy1 = max(0, gx1), max(0, gy1)
                gx2, gy2 = min(self.feat_sz, gx2), min(self.feat_sz, gy2)

                flat_scores = pred_score_map.flatten(1)  # [1, Lx]
                box_mask = torch.zeros(self.feat_sz, self.feat_sz, dtype=torch.bool,
                                       device=flat_scores.device)
                if gx2 > gx1 and gy2 > gy1:
                    box_mask[gy1:gy2, gx1:gx2] = True
                in_box = int(box_mask.sum().item())
                if in_box > 0:
                    masked_scores = flat_scores.masked_fill(~box_mask.flatten().unsqueeze(0), float('-inf'))
                    # Quality filter instead of greedy top-16: keep only tokens
                    # scoring near the in-box peak. On small targets (in-box area
                    # ~20 tokens) a fixed top-16 would sweep in box-corner
                    # background; this keeps just the genuine target tokens.
                    box_peak = masked_scores.max()
                    n_good = int((masked_scores >= self._tcm_score_ratio * box_peak).sum().item())
                    k = min(self._pool_size, n_good)
                    _, topk_pos = masked_scores.topk(k, dim=1)  # [1, K]
                    topk_feat = search_raw.gather(
                        dim=1, index=topk_pos.unsqueeze(-1).expand(-1, k, self.feat_dim))
                else:
                    k = 0

                # Spatially-aligned slot mapping: token offset from the response peak
                # on the search grid, rescaled to the template grid (target-centered).
                t_side = self._z_side
                t_center = t_side // 2
                peak_y = int(resp_max_idx[0].item() // self.feat_sz)
                peak_x = int(resp_max_idx[0].item() % self.feat_sz)
                pool = self._token_pool
                n_updated = 0
                if k > 0:
                    # vectorized slot update (the per-token python loop cost 30+
                    # CPU-GPU syncs per frame)
                    pos = topk_pos[0]                                   # [k]
                    sy = torch.div(pos, self.feat_sz, rounding_mode='floor')
                    sx = pos % self.feat_sz
                    ty = t_center + torch.round((sy - peak_y).float() * self._s2t_scale).long()
                    tx = t_center + torch.round((sx - peak_x).float() * self._s2t_scale).long()
                    valid = (ty >= 0) & (ty < t_side) & (tx >= 0) & (tx < t_side)
                    t_pos = (ty * t_side + tx)[valid]                   # [m]
                    if t_pos.numel() > 0:
                        pool[0, t_pos] = self._ema_alpha * pool[0, t_pos] + \
                            (1 - self._ema_alpha) * topk_feat[0][valid]
                        n_updated = int(t_pos.numel())

                # Blend with the first-frame anchor; slots never touched by the pool
                # still equal the anchor, so blending is the identity there.
                alpha = self._tcm_blend_alpha
                updated_z = (1 - alpha) * self._template_raw + alpha * pool \
                    + self.network.backbone.pos_embed_z
                self.z_dict1 = NestedTensor(updated_z, None)  # token format
                self._dprint(f'  [TCM] frame={self.frame_id}, conf={conf:.3f}, slots_updated={n_updated}')
            else:
                self._dprint(f'  [TCM] frame={self.frame_id}, conf={conf:.3f}, '
                             f'gated (conf_ok={conf_ok}, area_ok={area_ok})')

        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save}

        # Normal return path
        return {
            'target_bbox': self.state,
            'score_map': pred_score_map,
            'response': response,
            'prev_state': prev_state,
            'resize_factor': resize_factor,
            'search_crop': x_patch_arr
        }


def get_tracker_class():
    return OSTrack
