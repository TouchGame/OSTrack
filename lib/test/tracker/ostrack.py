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

        # ── ORIGIN mode: set env OSTRACK_ORIGIN=1 to disable every inference-time
        #    addition (multi-candidate, TCM, distractor bank) and run the vanilla
        #    OSTrack pipeline, for A/B control runs on the same checkpoint ──────
        self.origin_mode = os.environ.get('OSTRACK_ORIGIN', '0') == '1'
        if self.origin_mode:
            self.use_multicandidate = False
            print('[OSTrack] ORIGIN mode: all inference-time mechanisms disabled')

        # for consecutive prediction limit
        self.max_consecutive_predictions = 8  # max allowed consecutive predictions

        # Box-size anomaly guard permanently DISABLED (hardcoded best config):
        # ablation showed it hurts on val (frame-pooled AO 0.8752 -> 0.8554) by
        # suppressing legitimate fast scale change. Kept as a named constant so
        # the guard branch below stays readable.
        self.size_guard_enabled = False

        # Stage-2 boosted score map is multiplied by the Hann window before
        # cal_bbox. Ablation on GOT-10k val showed on/off are BIT-IDENTICAL
        # (180/180 seqs, AO=0.8759 either way): best_mask already isolates a
        # single region, so the Hann taper never moves cal_bbox's argmax. Kept
        # ON (original design, no risk; may still help on larger masks in
        # other datasets). ORIGIN mode turns it off with every other addition.
        self.s2_hann_enabled = not self.origin_mode

        # ── Stage-2 candidate source: generate peaks/regions/boosted map from
        # the Hann-modulated response instead of the raw score map. Rationale:
        # a strong EDGE distractor hijacks the relative peak threshold
        # (0.3 x max) and can drown a weak CENTER target into a single-region
        # skip_boosting frame; Hann compresses that dynamic range so the
        # center candidate survives detection. Side effects: region scores
        # feed the occlusion gate (easier to trigger) and combined already
        # has a Gaussian prior (center bias counted twice).
        # OSTRACK_S2_SRC_HANN=0 restores raw-map candidate generation.
        self.s2_src_hann = (not self.origin_mode and
                            os.environ.get('OSTRACK_S2_SRC_HANN', '1') == '1')

        # ── Stage-2 single-region raw peak: when the interference warning was
        # triggered by a Hann-induced peak shift (raw argmax != Hann argmax) and
        # region growing found only ONE region, predict from the RAW peak read
        # INSIDE that region's mask instead of the Hann-modulated peak. A single
        # region means no distractor to suppress, so Hann's motion prior is pure
        # regression cost here; masking to the region keeps the raw peak from
        # ever landing on an off-center distractor that Hann had suppressed.
        # HARDCODED BEST (default ON): GOT-10k val ep100 frame-pooled
        # AO 0.8734 -> 0.8744, the top config of the four-switch ablation.
        # OSTRACK_S2_SINGLE_RAW=0 disables it for A/B control runs.
        self.s2_single_raw = (not self.origin_mode and
                              os.environ.get('OSTRACK_S2_SINGLE_RAW', '1') == '1')

        # ── Stage-2 candidate-region generation knobs. Values (0.3 / 3 / 0.3)
        # reproduce the original region generation exactly. To experiment,
        # just edit the three numbers below (applied ONLY to the Stage-2
        # candidate path, not the distractor table or force-normal fallback):
        #   _peak_thresh_frac  abs peak floor as frac of heat.max (_findLocalPeaks)
        #   _peak_min_dist     NMS suppression radius in cells    (_findLocalPeaks)
        #   _grow_thresh_frac  region-grow cutoff as frac of peak (_regionGrowing)
        # Probe hypothesis: the truly-high response blob is much smaller than
        # the target box (target area ~1/25 of the search frame -> side ~5
        # cells, center-to-edge ~2.5), so a TIGHTER NMS radius (min_dist=2) can
        # split a merged target+distractor blob -- but full GOT-10k val ep100
        # showed it is net -0.02pp (helps seq 014/126, breaks 071/078), so the
        # original 3 is kept as the default.
        self._peak_thresh_frac = 0.3
        self._peak_min_dist = 3
        self._grow_thresh_frac = 0.3

        # ── Motion-consistency coast gate: in the uncertain band
        #    [low_confidence_threshold, _coast_iou_band) the winner region peak
        #    would normally be decoded and trusted. If its decoded box overlaps
        #    the dead-reckoning prediction by IoU < _coast_iou_thresh, treat it
        #    as a distractor sitting where the target vanished and coast instead
        #    (max_consecutive_predictions still caps the coasting run).
        #    OSTRACK_COAST_IOU=0 disables this gate for A/B control runs.
        self._coast_iou_enabled = os.environ.get('OSTRACK_COAST_IOU', '1') == '1'
        self._coast_iou_band = 0.2
        self._coast_iou_thresh = 0.5
        # Dead-reckoning residual reliability (OSTRACK_RESID, default ON): on
        # each confidently-locked frame record |actual displacement - predicted
        # displacement| / target-size. If the recent mean residual exceeds
        # _coast_resid_thresh the motion model has been spatially inaccurate
        # (camera shake on 034, acceleration lag on 071) -> skip the coast veto
        # and trust the detector's winner instead of coasting on a bad forecast.
        # Part of the default optimal combo; set OSTRACK_RESID=0 to A/B-disable.
        self._coast_resid_enabled = os.environ.get('OSTRACK_RESID', '1') == '1'
        self._coast_resid_thresh = 0.35

        # ── Single-region dual-peak IoU arbitration (OSTRACK_DUALPEAK, default
        #    ON): when Hann shifts the peak on a single region, decode both the
        #    raw and Hann peak boxes and keep whichever better matches the
        #    dead-reckoning prediction. Net-negative ALONE, but with the residual
        #    guard + coast gate the three form the default optimal combo (full-val
        #    net-neutral on AO, strictly >= baseline on hard anchors). Set 0 to A/B.
        self._dualpeak_enabled = os.environ.get('OSTRACK_DUALPEAK', '1') == '1'

        # ── Stage-1 jump check v2: the old grid-based conditions 2 (peak
        #    eccentricity > 3 cells) and 3 (peak jump > 2×prev movement) both
        #    proxy the same thing — per-frame motion — so they are replaced by
        #    ONE spatial-continuity test in IMAGE coordinates: distance from
        #    the raw peak (mapped to image coords) to the previous frame's
        #    center must not exceed ratio × median of recent accepted per-frame
        #    displacements. Floor = 2 grid cells in pixels: that is the
        #    quantization noise of the peak measurement itself, without it a
        #    static target's 1-cell jitter fires every frame (median ≈ 0).
        #    HARDCODED BEST (default OFF): the old two grid-based checks win
        #    on GOT-10k val ep100, so jump-v2 is off in the best config.
        #    OSTRACK_JUMP_V2=1 re-enables the image-coord continuity test. ──
        self.jump_v2_enabled = os.environ.get('OSTRACK_JUMP_V2', '0') == '1'
        self._jump_ratio = 2.0        # allowed multiple of recent median speed
        self._speed_hist = []         # accepted per-frame displacements (image px)
        self._speed_hist_len = 5

        # ── TCM: Token Context Memory (LMTrack-style dynamic template) ──────────
        self.ref_pool_enabled = getattr(params.cfg.MODEL.BACKBONE, 'REF_POOL', False)
        if self.origin_mode:
            self.ref_pool_enabled = False
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
        # ── Distractor bank: Stage-2 rejected candidate regions are confirmed
        #    distractors; remember them (backbone-OUTPUT space, unlike the TCM
        #    pool which lives in patch-embed INPUT space) ──────────────────────
        self._db_size = 8             # ring-buffer capacity
        self._db_rel_score = 0.3      # rejected region must score >= this frac of
                                      # the best region to be worth remembering
        self._db_veto = True          # veto TCM pool writes that look more like a
                                      # remembered distractor than like the target
        self._db_veto_margin = 0.0    # extra margin required to veto (probe: 0.1
                                      # made LightOcc worse -> veto helps, keep 0.0)
        self._db_penalty = False      # apply distractor penalty in Stage-2 scoring
        self._db_min_dist = 5.0       # min grid distance (tokens) from the chosen
                                      # region: partial occlusion splits the TARGET
                                      # into adjacent regions, and banking such a
                                      # fragment poisons the bank (LightOcc probe)
        self._db_max_anchor_sim = 0.6 # don't bank features this similar to the
                                      # first-frame target: a same-class twin
                                      # (Running: other runners) has zero
                                      # discriminative value and penalizing it
                                      # punishes the target itself
        self._tcm_min_tokens = 6      # skip pool writes on degenerate tiny boxes
        self._tcm_interval = 1        # sparse rebuild: run Stage 3 only every N
                                      # frames (1 = every frame). Probed N=3: zero
                                      # speed gain (Stage 3 costs ~1.4% frame time)
                                      # and -0.010 full-run Success (MotorNig -0.49:
                                      # a stale template misses rapid night-scene
                                      # appearance drift) -> keep 1
        self._distractor_bank = []    # normalized [C] features, FIFO
        self._anchor_center_feat = None  # first-frame template center feature [1, C]
        # ── Distractor trajectory table: appearance cannot separate same-class
        #    twins (the Running lesson), but their positions/trajectories can.
        #    Stage-2 rejected regions feed lightweight tracklets (pos+vel only,
        #    no appearance); consulted ONLY at ambiguous moments (top-2 tie,
        #    post-occlusion re-lock), so ordinary frames are never touched ────
        self._dt_enabled = not self.origin_mode
        self._dt_max_age = 8          # frames without a match before a track dies
        self._dt_min_hits = 3         # observations before a track may influence
                                      # decisions (one-off detections are noise)
        self._dt_tgt_frac = 0.3       # a track chosen as target in more than this
                                      # fraction of its observations is treated as
                                      # the target's own track, never a distractor
        self._dt_tie_ratio = 0.7      # top-2 combined ratio that counts as a tie
        self._dt_relock_factor = 2.0  # re-locking onto a track prediction needs
                                      # this multiple of low_confidence_threshold
        self._dt_match_rel = 1.0      # association radius, × sqrt(target area)
        self._dt_hit_rel = 0.6        # consult radius,     × sqrt(target area)
        self._dtracks = []            # dicts: x, y, vx, vy, last_frame, hits
        self._coasting = False        # True while dead-reckoning through occlusion
        # search-grid → template-grid scale: both crops are target-centered, but
        # sampled with different context factors, so token offsets must be rescaled
        self._s2t_scale = (self.params.template_size / self.params.template_factor) / \
                          (self.params.search_size / self.params.search_factor)

    def _dprint(self, *args, **kwargs):
        """Debug printing, silenced unless params.debug is set."""
        if self.debug:
            print(*args, **kwargs)

    def _dump_region_cells(self, score_map, response, size_map, offset_map,
                           masks, resize_factor, H, W):
        """Diagnostic (OSTRACK_DUMP_CELLS): for the single detected region,
        decode EVERY in-mask cell's own box via its size_map/offset_map using
        the exact same transform as cal_bbox+map_box_back (self.state still
        holds the PREVIOUS box here = target prior). Prints each cell's box in
        image coords so we can check offline (vs GT) whether a cell other than
        the response-argmax gives a box that fits the target better. Print-only,
        no tracking side effects."""
        b = 0
        sz = self.feat_sz
        m = masks[b][0].astype(bool)
        ys, xs = np.where(m)
        argmax_idx = int(response[b, 0].flatten().argmax().item())
        amx_y, amx_x = argmax_idx // sz, argmax_idx % sz
        rows = []
        for yy, xx in zip(ys.tolist(), xs.tolist()):
            raw_s = float(score_map[b, 0, yy, xx].item())
            resp_s = float(response[b, 0, yy, xx].item())
            ox = float(offset_map[b, 0, yy, xx].item())
            oy = float(offset_map[b, 1, yy, xx].item())
            wn = float(size_map[b, 0, yy, xx].item())
            hn = float(size_map[b, 1, yy, xx].item())
            cxn, cyn = (xx + ox) / sz, (yy + oy) / sz
            pb = [v * self.params.search_size / resize_factor
                  for v in (cxn, cyn, wn, hn)]
            st = clip_box(self.map_box_back(pb, resize_factor), H, W, margin=10)
            rows.append((resp_s, raw_s, yy, xx, st))
        rows.sort(key=lambda r: -r[0])
        print(f"  [DumpCells] frame={self.frame_id} argmax=({amx_y},{amx_x}) "
              f"n_cells={len(rows)} (resp-desc; box=[x,y,w,h] image coords):")
        for resp_s, raw_s, yy, xx, st in rows:
            tag = " <-argmax" if (yy == amx_y and xx == amx_x) else ""
            print(f"    cell=({yy},{xx}) resp={resp_s:.4f} raw={raw_s:.4f} "
                  f"box=[{st[0]:.0f},{st[1]:.0f},{st[2]:.0f},{st[3]:.0f}]{tag}")

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
        # Distractor bank state (per sequence)
        self._distractor_bank = []
        self._anchor_center_feat = None  # lazily cached on the first track() call
        self._dtracks = []
        self._coasting = False

        # ── reset per-sequence state ─────────────────────────────────────────────
        self.prev_search_crop = None
        self.prev_state = None
        self._prev_state = None
        self.position_history = []      # [(dx, dy), ...] in pixel coordinates
        self.pred_residuals = []        # recent |actual - predicted| / target-size on locked frames
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

        # Candidates at CE-eliminated positions come back as ZERO vectors
        # (recover_tokens zero-pads), which silently vetoes a legitimate
        # candidate: sim=0 -> combined=0 -> bogus occlusion dead-reckoning
        # (GOT-10k_Val_000021 frames 35-36: the re-emerging dog scored 0.33-0.47
        # but its token was pruned, so every gate saw exactly 0). Fall back to
        # the mean of non-zero tokens in the 3x3 neighborhood; CE pruning is
        # sparse, so a real peak almost always has surviving neighbors.
        norms = candidate_features.norm(dim=-1)  # [B, K]
        if (norms < 1e-6).any():
            for b in range(B):
                for k in range(K):
                    if norms[b, k] >= 1e-6:
                        continue
                    cy = int(y[b, k].item())
                    cx = int(x[b, k].item())
                    neigh = []
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            ny, nx = cy + dy, cx + dx
                            if 0 <= ny < feat_sz and 0 <= nx < feat_sz:
                                tok = search_tokens[b, ny * feat_sz + nx]
                                if tok.norm() > 1e-6:
                                    neigh.append(tok)
                    if neigh:
                        candidate_features[b, k] = torch.stack(neigh).mean(dim=0)
                        self._dprint(f"  [Sim] CE-pruned token at ({cy},{cx}), "
                                     f"using {len(neigh)} non-zero neighbors")

        # Normalize candidate features (a still-zero vector stays zero -> sim 0)
        candidate_features_norm = torch.nn.functional.normalize(candidate_features, p=2, dim=-1)  # [B, K, C]

        # Cosine similarity: [B, K, C] * [B, 1, C] -> [B, K]
        similarities = (candidate_features_norm * prev_feat_normalized.unsqueeze(1)).sum(dim=-1)

        self._dprint(f"  [Sim Debug] token_indices={token_indices}, similarities={similarities}")

        return similarities

    def _findLocalPeaks(self, score_map, min_distance=3, threshold_fraction=0.3):
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

    def _predict_residual(self):
        """Mean recent dead-reckoning residual: on the last few locked frames,
        how far the actual displacement fell from what _predict_displacement had
        forecast, normalized by target size. Large = the motion model has been
        spatially unreliable (shake / acceleration) and must not veto the
        detector. Returns 0.0 until enough samples exist."""
        if len(self.pred_residuals) < 2:
            return 0.0
        n = min(5, len(self.pred_residuals))
        recent = self.pred_residuals[-n:]
        return sum(recent) / n

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

    def _decode_cell_box(self, size_map, offset_map, cy, cx, resize_factor, b=0):
        """Decode the box regressed at ONE feature cell (cy, cx) into image
        coords, matching cal_bbox + map_box_back exactly. Lets us score the
        box a given peak would produce without routing it through argmax."""
        ox = float(offset_map[b, 0, cy, cx])
        oy = float(offset_map[b, 1, cy, cx])
        w = float(size_map[b, 0, cy, cx])
        h = float(size_map[b, 1, cy, cx])
        scale = self.params.search_size / resize_factor
        cx_s = (cx + ox) / self.feat_sz * scale
        cy_s = (cy + oy) / self.feat_sz * scale
        return self.map_box_back([cx_s, cy_s, w * scale, h * scale], resize_factor)

    @staticmethod
    def _xywh_iou(a, b):
        """IoU of two [x, y, w, h] boxes in the same coordinate frame."""
        ax2, ay2 = a[0] + a[2], a[1] + a[3]
        bx2, by2 = b[0] + b[2], b[1] + b[3]
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        union = a[2] * a[3] + b[2] * b[3] - inter
        return inter / union if union > 1e-6 else 0.0

    # ── Distractor trajectory table ──────────────────────────────────────────
    def _grid_to_image(self, cy, cx, resize_factor, ref_state):
        """Map a search-grid token center (cy, cx) to image coords. Same
        geometry as map_box_back, but for a bare point and an explicit
        reference state (self.state may already hold the NEW box)."""
        cx_prev = ref_state[0] + 0.5 * ref_state[2]
        cy_prev = ref_state[1] + 0.5 * ref_state[3]
        half_side = 0.5 * self.params.search_size / resize_factor
        stride = self.params.search_size / self.feat_sz
        ix = (float(cx) + 0.5) * stride / resize_factor + (cx_prev - half_side)
        iy = (float(cy) + 0.5) * stride / resize_factor + (cy_prev - half_side)
        return ix, iy

    def _dt_predict(self, tr):
        dt = self.frame_id - tr['last_frame']
        return tr['x'] + tr['vx'] * dt, tr['y'] + tr['vy'] * dt

    def _dt_on_track(self, ix, iy, radius):
        """Is (ix, iy) within radius of a mature DISTRACTOR track's prediction?
        A track that keeps being observed while (almost) never being the chosen
        target is a confirmed distractor; likely-target tracks are skipped."""
        for tr in self._dtracks:
            if tr['hits'] < self._dt_min_hits or \
                    self.frame_id - tr['last_frame'] > self._dt_max_age or \
                    tr['tgt'] > self._dt_tgt_frac * tr['hits']:
                continue
            px, py = self._dt_predict(tr)
            if math.hypot(ix - px, iy - py) <= radius:
                return True
        return False

    def _dt_mark_target(self):
        """Credit the tracklet matched to this frame's final box center: it is
        (currently believed to be) the target, not a distractor."""
        cx = self.state[0] + 0.5 * self.state[2]
        cy = self.state[1] + 0.5 * self.state[3]
        r = self._dt_hit_rel * math.sqrt(max(self.state[2] * self.state[3], 1.0))
        best, best_d = None, r
        for tr in self._dtracks:
            px, py = self._dt_predict(tr)
            d = math.hypot(cx - px, cy - py)
            if d < best_d:
                best, best_d = tr, d
        if best is not None:
            best['tgt'] += 1

    def _dt_update(self, points):
        """Greedy nearest-neighbor association of rejected-region centers
        (image coords) to tracklets; EMA velocity; prune stale tracks."""
        r = self._dt_match_rel * math.sqrt(max(self.state[2] * self.state[3], 1.0))
        used = set()
        for ix, iy in points:
            best, best_d = None, r
            for ti, tr in enumerate(self._dtracks):
                if ti in used:
                    continue
                px, py = self._dt_predict(tr)
                d = math.hypot(ix - px, iy - py)
                if d < best_d:
                    best, best_d = ti, d
            if best is not None:
                tr = self._dtracks[best]
                dt = max(self.frame_id - tr['last_frame'], 1)
                tr['vx'] = 0.5 * tr['vx'] + 0.5 * (ix - tr['x']) / dt
                tr['vy'] = 0.5 * tr['vy'] + 0.5 * (iy - tr['y']) / dt
                tr['x'], tr['y'] = ix, iy
                tr['last_frame'] = self.frame_id
                tr['hits'] += 1
                used.add(best)
            else:
                self._dtracks.append({'x': ix, 'y': iy, 'vx': 0.0, 'vy': 0.0,
                                      'last_frame': self.frame_id, 'hits': 1,
                                      'tgt': 0})
        self._dtracks = [t for t in self._dtracks
                         if self.frame_id - t['last_frame'] <= self._dt_max_age]

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

        # Cache the first-frame template center feature (backbone-OUTPUT space)
        # as the "target" side of distractor-vs-target arbitration. Frame 1 still
        # runs the pristine template, so this is a clean anchor.
        if self._anchor_center_feat is None:
            lens_z = self.network.backbone.pos_embed_z.shape[1]
            zf = out_dict['backbone_feat'][:, :lens_z, :]
            zt_side = int(math.sqrt(lens_z))
            zc0, zc1 = zt_side // 2 - zt_side // 4, zt_side // 2 + zt_side // 4
            zcf = zf.view(1, zt_side, zt_side, -1)[:, zc0:zc1, zc0:zc1, :] \
                .reshape(1, -1, zf.shape[-1]).mean(dim=1)
            self._anchor_center_feat = torch.nn.functional.normalize(zcf, p=2, dim=-1)  # [1, C]

        # ============================================================
        # Stage 1: Interference Warning
        # ============================================================
        raw_max_idx = pred_score_map.flatten(1).argmax(dim=1)  # [B], flat index
        resp_max_idx = response.flatten(1).argmax(dim=1)       # [B], flat index

        raw_y = int(raw_max_idx[0].item() // self.feat_sz)
        raw_x = int(raw_max_idx[0].item() % self.feat_sz)

        interference_warning = (raw_max_idx != resp_max_idx).any().item()
        # Capture the pure peak-shift signal BEFORE the jump check ORs into it:
        # True iff the Hann window moved the argmax cell off the raw peak.
        peak_shifted = interference_warning
        feat_center = self.feat_sz // 2   # also used by Stage-2 position scoring
        if self.jump_v2_enabled:
            # Single continuity check in image coords: raw-peak-implied center
            # vs previous frame's center. Threshold adapts to the target's own
            # recent speed, so sustained fast motion is not flagged every frame.
            if not interference_warning:
                pix, piy = self._grid_to_image(raw_y, raw_x, resize_factor, prev_state)
                pcx = prev_state[0] + 0.5 * prev_state[2]
                pcy = prev_state[1] + 0.5 * prev_state[3]
                jump_px = math.hypot(pix - pcx, piy - pcy)
                cell_px = (self.params.search_size / self.feat_sz) / resize_factor
                base = float(np.median(self._speed_hist)) if self._speed_hist else 0.0
                jump_thr = max(self._jump_ratio * base, 2.0 * cell_px)
                if jump_px > jump_thr:
                    interference_warning = True
                    self._dprint(f'  [Warning] jump {jump_px:.1f}px > max({self._jump_ratio}*median '
                                 f'{base:.1f}, 2 cells {2.0 * cell_px:.1f}), possible interference')
        else:
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

        # ── Distractor trajectory feeding: ALL local peaks of the raw score map
        # are tracked every frame (KeepTrack-style), including the target's own
        # peak. Identity comes from history, not position: the tracklet matched
        # to each frame's FINAL box gets a target mark (_dt_mark_target); a
        # tracklet that keeps being observed while (almost) never being chosen
        # is a confirmed distractor. (A spatial exclusion zone around the crop
        # center was probed first — but close encounters put distractor peaks
        # exactly there, starving the table right when it was needed.)
        if self._dt_enabled:
            dt_peaks, _ = self._findLocalPeaks(pred_score_map, min_distance=3)
            self._dt_update([self._grid_to_image(py, px, resize_factor, prev_state)
                             for py, px in dt_peaks[0]])

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
            # Find connected regions using local peaks + region growing.
            # cand_map: Hann-modulated response (default) or raw score map.
            cand_map = response if self.s2_src_hann else pred_score_map
            max_score = cand_map.max().item()
            peaks, peak_scores = self._findLocalPeaks(
                cand_map, min_distance=self._peak_min_dist,
                threshold_fraction=self._peak_thresh_frac)
            centers, region_scores_list, masks = self._regionGrowing(
                cand_map, peaks, peak_scores, threshold_fraction=self._grow_thresh_frac)
            # Split source: the Hann map only DETECTS candidates (peak set and
            # growth boundaries); region scores are re-read from the RAW map.
            # Every downstream consumer (combined, occlusion gate, re-lock,
            # distractor bank) needs the absolute score scale — Hann-discounted
            # scores fake "all regions weak" for off-center targets and
            # false-trigger dead-reckoning (GOT-10k val 000021/000034).
            if self.s2_src_hann:
                region_scores_list = [
                    [float(pred_score_map[b, 0, cy, cx].item())
                     for cy, cx in centers[b]]
                    for b in range(pred_score_map.shape[0])]
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

                # Distractor penalty: suppress a candidate only by how much MORE it
                # resembles a remembered distractor than the target. Ordinary
                # candidates get penalty=1.0, so the occlusion low-confidence gate
                # downstream is untouched.
                if self._db_penalty and len(self._distractor_bank) > 0:
                    flat_idx = (positions_tensor[..., 0] * self.feat_sz +
                                positions_tensor[..., 1])  # [1, K]
                    cand_feat = search_tokens[b:b + 1].gather(
                        dim=1, index=flat_idx.unsqueeze(-1).expand(
                            -1, -1, search_tokens.shape[-1]).to(torch.int64))
                    cand_norm = torch.nn.functional.normalize(cand_feat, p=2, dim=-1)  # [1, K, C]
                    bank = torch.stack(self._distractor_bank).to(cand_norm.device)     # [M, C]
                    dist_sims = (cand_norm @ bank.t()).max(dim=-1).values              # [1, K]
                    penalty = 1.0 - torch.clamp(dist_sims - sims, min=0.0, max=1.0)
                else:
                    penalty = torch.ones_like(sims)

                # Combined: score × similarity × position_score × distractor_penalty
                combined_b = scores_b.unsqueeze(0) * sims * position_scores_tensor.unsqueeze(0) * penalty  # [1, K]
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

            # ── Trajectory tie-break: appearance cannot separate same-class
            # twins; position history can. Ambiguity is judged WITHOUT the
            # Gaussian position prior (it assumes a centered target — precisely
            # wrong mid-drift, and it crushes off-center candidates to ~0 so
            # combined-score ties never fire). If the combined winner sits on a
            # remembered distractor trajectory, hand the frame to the best
            # appearance-comparable candidate that does not.
            if self._dt_enabled and len(self._dtracks) > 0 and len(centers[0]) > 1 \
                    and raw_sims_list and len(raw_sims_list[0]) == len(centers[0]) \
                    and combined_list[0].numel() == len(centers[0]):
                comb0 = combined_list[0].flatten()
                i1 = best_idx_per_batch[0]
                q = [region_scores_list[0][k] * float(raw_sims_list[0][k])
                     for k in range(len(centers[0]))]
                hit_r = self._dt_hit_rel * math.sqrt(
                    max(prev_state[2] * prev_state[3], 1.0))
                p1 = self._grid_to_image(*centers[0][i1], resize_factor, prev_state)
                if q[i1] > 0 and self._dt_on_track(*p1, hit_r):
                    alt, alt_comb = None, -1.0
                    for k in range(len(centers[0])):
                        if k == i1 or q[k] < self._dt_tie_ratio * q[i1]:
                            continue
                        pk = self._grid_to_image(*centers[0][k], resize_factor, prev_state)
                        if self._dt_on_track(*pk, hit_r):
                            continue
                        if float(comb0[k].item()) > alt_comb:
                            alt, alt_comb = k, float(comb0[k].item())
                    if alt is not None:
                        best_idx_per_batch[0] = alt
                        best_combined_val[0] = alt_comb
                        self._dprint(f"  [DTrack] tie-break: region {i1} sits on a "
                                     f"distractor trajectory, switching to region {alt}")

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

            # Occlusion detection: check if all regions are very weak. Judged
            # WITHOUT the Gaussian position prior: the sigma=2 prior baked
            # into `combined` crushes fast off-center targets to ~0 and
            # triggers bogus dead-reckoning (GOT-10k val collapsed on this).
            # The trade-off is real — with no prior, night-scene look-alikes
            # can block occlusion mode (GTOT MotorNig) — but probes showed no
            # positional gate wins both domains (static σ6, motion-compensated
            # σ3, and a tracklet-identity gate each fixed one and broke the
            # other), and the benchmark priority is the general domain.
            max_combined = 0.0
            for b in range(pred_score_map.shape[0]):
                if b < len(raw_sims_list) and \
                        len(raw_sims_list[b]) == len(region_scores_list[b]):
                    for k in range(len(region_scores_list[b])):
                        max_combined = max(max_combined,
                                           region_scores_list[b][k]
                                           * float(raw_sims_list[b][k]))
                elif len(combined_list[b]) > 0 and combined_list[b].numel() > 0:
                    max_combined = max(max_combined, combined_list[b].max().item())
            low_confidence_threshold = 0.1

            # ── Trajectory re-lock gate: the first confident frame after coasting
            # through an occlusion is the classic moment to lock onto a distractor
            # standing where the target vanished. If the would-be winner sits on a
            # distractor trajectory prediction, demand a higher score; otherwise
            # coast one more frame (max_consecutive_predictions still caps this).
            if self._dt_enabled and self._coasting and len(self._dtracks) > 0 and \
                    len(centers[0]) > 0 and \
                    low_confidence_threshold <= max_combined < \
                    self._dt_relock_factor * low_confidence_threshold:
                bi = best_idx_per_batch[0]
                hit_r = self._dt_hit_rel * math.sqrt(
                    max(prev_state[2] * prev_state[3], 1.0))
                pb = self._grid_to_image(*centers[0][bi], resize_factor, prev_state)
                if self._dt_on_track(*pb, hit_r):
                    self._dprint(f"  [DTrack] re-lock veto: candidate on distractor "
                                 f"trajectory with weak score {max_combined:.3f}, keep coasting")
                    max_combined = 0.0

            # ── Motion-consistency veto: in the uncertain band
            # [low_confidence_threshold, _coast_iou_band), the winner region
            # peak would normally be decoded and trusted. But if its decoded box
            # barely overlaps the dead-reckoning prediction, it is most likely a
            # distractor standing where the target vanished -> coast instead.
            if self._coast_iou_enabled and len(centers[0]) > 0 and \
                    len(self.position_history) >= 2 and \
                    low_confidence_threshold <= max_combined < self._coast_iou_band:
                bi = best_idx_per_batch[0]
                cand_box = self._decode_cell_box(
                    out_dict['size_map'], out_dict['offset_map'],
                    int(centers[0][bi][0]), int(centers[0][bi][1]), resize_factor)
                pred_dx, pred_dy = self._predict_displacement()
                pcx = prev_state[0] + 0.5 * prev_state[2] + pred_dx
                pcy = prev_state[1] + 0.5 * prev_state[3] + pred_dy
                pred_box = [pcx - 0.5 * prev_state[2], pcy - 0.5 * prev_state[3],
                            prev_state[2], prev_state[3]]
                iou_cand = self._xywh_iou(cand_box, pred_box)
                resid = self._predict_residual() if self._coast_resid_enabled else 0.0
                unreliable = resid > self._coast_resid_thresh
                if iou_cand < self._coast_iou_thresh and not unreliable:
                    self._dprint(f"  [Occlusion] winner peak box IoU {iou_cand:.3f} < "
                                 f"{self._coast_iou_thresh} vs dead-reckoning "
                                 f"(combined {max_combined:.3f}, resid {resid:.2f}) "
                                 f"-> distractor, coast")
                    max_combined = 0.0
                elif iou_cand < self._coast_iou_thresh:
                    self._dprint(f"  [Occlusion] winner peak box IoU {iou_cand:.3f} low but "
                                 f"dead-reckoning unreliable (resid {resid:.2f}) "
                                 f"-> trust winner, no coast")

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
                self._coasting = True  # arms the trajectory re-lock gate
                response = self.output_window * pred_score_map
                pred_dx, pred_dy = self._predict_displacement()
                x1, y1, w, h = self.state
                pred_x = x1 + pred_dx
                pred_y = y1 + pred_dy
                # Dead-reckoning: keep the FULL predicted box inside the image by
                # SHIFTING it (top-left clamped to [0, W-w] / [0, H-h]) instead of
                # clipping off the out-of-bounds part. Clipping used to shrink w/h
                # when the target coasted past a border; here we preserve its true
                # size. If the box is larger than the image it is left/top-aligned.
                pred_x = min(max(0.0, pred_x), max(0.0, W - w))
                pred_y = min(max(0.0, pred_y), max(0.0, H - h))
                self.state = [pred_x, pred_y, w, h]
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
                if self.s2_single_raw and peak_shifted:
                    # Hann moved the peak on a single (no-distractor) target:
                    # read the RAW peak INSIDE the detected region, so the box
                    # comes from the best-regression cell without Hann's center
                    # pull, and the argmax cannot escape onto a suppressed
                    # off-center distractor outside the region.
                    response = torch.zeros_like(pred_score_map)
                    for b in range(pred_score_map.shape[0]):
                        m = torch.from_numpy(masks[b][0].astype(np.float32)).to(pred_score_map.device)
                        response[b, 0] = m * pred_score_map[b, 0]
                    if self._dualpeak_enabled:
                        # ── Dual-peak IoU arbitration (OSTRACK_DUALPEAK=1). One grown
                        # region can straddle BOTH the target and a distractor: the
                        # RAW peak may land on the distractor while the Hann peak
                        # lands on the target (or vice-versa when the target moved
                        # fast). Decode the box at each peak and keep whichever better
                        # matches a motion-predicted box from history (higher IoU).
                        m0 = torch.from_numpy(masks[0][0].astype(np.float32)).to(pred_score_map.device)
                        raw_masked = m0 * pred_score_map[0, 0]
                        hann_masked = m0 * (self.output_window * pred_score_map)[0, 0]
                        raw_idx = int(torch.argmax(raw_masked).item())
                        hann_idx = int(torch.argmax(hann_masked).item())
                        if raw_idx != hann_idx and len(self.position_history) >= 2:
                            rcy, rcx = raw_idx // self.feat_sz, raw_idx % self.feat_sz
                            hcy, hcx = hann_idx // self.feat_sz, hann_idx % self.feat_sz
                            raw_box = self._decode_cell_box(out_dict['size_map'], out_dict['offset_map'], rcy, rcx, resize_factor)
                            hann_box = self._decode_cell_box(out_dict['size_map'], out_dict['offset_map'], hcy, hcx, resize_factor)
                            pred_dx, pred_dy = self._predict_displacement()
                            pcx = self.state[0] + 0.5 * self.state[2] + pred_dx
                            pcy = self.state[1] + 0.5 * self.state[3] + pred_dy
                            pred_box = [pcx - 0.5 * self.state[2], pcy - 0.5 * self.state[3],
                                        self.state[2], self.state[3]]
                            iou_raw = self._xywh_iou(raw_box, pred_box)
                            iou_hann = self._xywh_iou(hann_box, pred_box)
                            if iou_hann > iou_raw:
                                response[0, 0] = hann_masked
                                self._dprint(f"  [Stage2] dual-peak: HANN wins (IoU {iou_hann:.3f} > "
                                             f"raw {iou_raw:.3f}), raw@({rcy},{rcx}) hann@({hcy},{hcx})")
                            else:
                                self._dprint(f"  [Stage2] dual-peak: RAW wins (IoU {iou_raw:.3f} >= "
                                             f"hann {iou_hann:.3f}), raw@({rcy},{rcx}) hann@({hcy},{hcx})")
                        else:
                            self._dprint(f"  [Stage2] single-region raw-peak (peak_shifted)")
                    else:
                        self._dprint(f"  [Stage2] single-region raw-peak (peak_shifted)")
                else:
                    # Single region - use normal response (Hann modulated)
                    response = self.output_window * pred_score_map
                _dc = os.environ.get('OSTRACK_DUMP_CELLS', '')
                if _dc and str(self.frame_id) in _dc.split(','):
                    self._dump_region_cells(pred_score_map, response,
                                            out_dict['size_map'], out_dict['offset_map'],
                                            masks, resize_factor, H, W)
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
                        # Only selected region has non-zero scores; boosted map
                        # uses the RAW map values — the mask (from the Hann
                        # detection source) already isolates the target, so box
                        # regression is not pulled toward the center by Hann.
                        # Clean SG=0 ablation: raw 0.8752 vs hann 0.8740, raw wins
                        boosted_score_map[b, 0] = best_mask * pred_score_map[b, 0] * boost_factor
                    else:
                        self._dprint(f"    ERROR: best_idx={bi} >= len(masks)={len(masks[b])}")

                # Predict from boosted score_map, re-applying the Hann window
                # (best config; disabled only in ORIGIN mode)
                if self.s2_hann_enabled:
                    response = self.output_window * boosted_score_map
                else:
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

            # ── Distractor bank write: rejected regions are CONFIRMED distractors.
            # Only on confident decisions (a low-confidence pick could mean a
            # "rejected" region is actually the target), and only strong regions
            # (weak ones are noise, not recurring distractors).
            if stage2_confident and not skip_boosting and len(centers[0]) > 1:
                best_ri = best_idx_per_batch[0]
                best_reg_score = max(region_scores_list[0])
                bcy, bcx = centers[0][best_ri]
                for ri, (cy, cx) in enumerate(centers[0]):
                    if ri == best_ri or region_scores_list[0][ri] < self._db_rel_score * best_reg_score:
                        continue
                    # Adjacent regions are usually TARGET fragments from partial
                    # occlusion (a pole splitting the target in two) — banking
                    # them would poison the bank. Only bank well-separated peaks.
                    if math.hypot(float(cy) - float(bcy), float(cx) - float(bcx)) < self._db_min_dist:
                        continue
                    feat = search_tokens[0, int(cy) * self.feat_sz + int(cx)]
                    if float(feat.norm().item()) > 1e-6:  # CE-eliminated slots are zeros
                        fn = torch.nn.functional.normalize(feat, p=2, dim=0).detach()
                        # Same-class twins (visually ~= the target) carry no
                        # discriminative signal; banking them punishes the target.
                        if self._anchor_center_feat is not None and \
                                float((fn * self._anchor_center_feat[0]).sum().item()) > self._db_max_anchor_sim:
                            continue
                        self._distractor_bank.append(fn)
                        if len(self._distractor_bank) > self._db_size:
                            self._distractor_bank.pop(0)
                        self._dprint(f"  [DistractorBank] +region {ri} at ({cy},{cx}), "
                                     f"bank size={len(self._distractor_bank)}")
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

            # Box-size guard is an inference-time addition too: ORIGIN mode must
            # accept the raw model output exactly like vanilla OSTrack.
            if self.size_guard_enabled and len(self.box_size_history) >= 3:
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
        # visdom may be None if the server is not running (_init_visdom
        # swallows the connection error); demo scripts also call track()
        # without info, so gt_bbox may be unavailable.
        if self.debug and self.use_visdom and self.visdom is not None:
            if info is not None and info.get('gt_bbox') is not None:
                self.visdom.register((image, info['gt_bbox'].tolist(), self.state), 'Tracking', 1, 'Tracking')
            else:
                self.visdom.register((image, self.state), 'Tracking', 1, 'Tracking')
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
        # Reaching here means a measurement was accepted this frame (the
        # dead-reckoning branch returns early) — disarm the re-lock gate and
        # credit the tracklet under the final box as the target.
        self._coasting = False
        if self._dt_enabled:
            self._dt_mark_target()

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
            # Record dead-reckoning residual (actual vs forecast) for the coast
            # veto's reliability gate, computed BEFORE this frame's displacement
            # is appended so the forecast matches what the veto used this frame.
            if self._coast_resid_enabled and len(self.position_history) >= 2:
                pdx, pdy = self._predict_displacement()
                tsz = 0.5 * (self.state[2] + self.state[3])
                if tsz > 1e-6:
                    self.pred_residuals.append(math.hypot(dx - pdx, dy - pdy) / tsz)
                    if len(self.pred_residuals) > 10:
                        self.pred_residuals.pop(0)
            if len(self.position_history) > 0 or (abs(dx) > 0.5 or abs(dy) > 0.5):
                self.position_history.append((dx, dy))
                if len(self.position_history) > 10:
                    self.position_history.pop(0)
            # Speed baseline for the Stage-1 jump check (v2). Shares this gate
            # deliberately: low-confidence Stage-2 frames must not inflate the
            # "normal speed" and mask real jumps on following frames.
            if self.jump_v2_enabled:
                self._speed_hist.append(math.hypot(dx, dy))
                if len(self._speed_hist) > self._speed_hist_len:
                    self._speed_hist.pop(0)
            self._prev_state = list(self.state)
        else:
            self._dprint(f"  [History Update] Skipped - low confidence")

        # ============================================================
        # Stage 3: TCM — Token Context Memory
        # Blend high-response search tokens into the template, all in the
        # block-0 INPUT space (patch embeddings), anchored to the first frame.
        # ============================================================
        if self.ref_pool_enabled and self._template_raw is not None and \
                self.frame_id % self._tcm_interval == 0:
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

                # ── Distractor veto: if the response-peak token looks more like a
                # remembered distractor than like the first-frame target, the box
                # has probably drifted onto a distractor WITHOUT tripping any
                # warning (the tunnel failure mode) — do not memorize this frame.
                veto = False
                if self._db_veto and len(self._distractor_bank) > 0 and \
                        self._anchor_center_feat is not None:
                    lens_x_s3 = self.network.backbone.pos_embed_x.shape[1]
                    peak_tok = out_dict['backbone_feat'][0, -lens_x_s3:, :][int(resp_max_idx[0].item())]
                    if float(peak_tok.norm().item()) > 1e-6:  # CE-eliminated slots are zeros
                        f = torch.nn.functional.normalize(peak_tok, p=2, dim=0)
                        sim_t = float((f * self._anchor_center_feat[0]).sum().item())
                        bank = torch.stack(self._distractor_bank)
                        sim_d = float((bank @ f).max().item())
                        veto = sim_d > sim_t + self._db_veto_margin
                        if veto:
                            self._dprint(f'  [TCM] frame={self.frame_id}, VETO: '
                                         f'sim_distractor={sim_d:.3f} > sim_target={sim_t:.3f}')

                if in_box >= self._tcm_min_tokens and not veto:
                    with torch.no_grad():
                        # Fresh patch-embed of the current search crop: same input space
                        # as the cached template tokens, and immune to CE zero-padding.
                        search_raw = self.network.backbone.patch_embed(x_dict.tensors)  # [1, Lx, C]
                    masked_scores = flat_scores.masked_fill(~box_mask.flatten().unsqueeze(0), float('-inf'))
                    # Optional quality filter (score_ratio=0.0 keeps plain in-box top-k;
                    # 0.5 proved too strict and starved pool updates).
                    box_peak = masked_scores.max()
                    n_good = int((masked_scores >= self._tcm_score_ratio * box_peak).sum().item()) \
                        if self._tcm_score_ratio > 0 else in_box
                    k = min(self._pool_size, n_good)
                    _, topk_pos = masked_scores.topk(k, dim=1)  # [1, K]
                    topk_feat = search_raw.gather(
                        dim=1, index=topk_pos.unsqueeze(-1).expand(-1, k, self.feat_dim))

                    # Spatially-aligned slot mapping: token offset from the response peak
                    # on the search grid, rescaled to the template grid (target-centered).
                    t_side = self._z_side
                    t_center = t_side // 2
                    peak_y = int(resp_max_idx[0].item() // self.feat_sz)
                    peak_x = int(resp_max_idx[0].item() % self.feat_sz)
                    pool = self._token_pool
                    n_updated = 0
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
