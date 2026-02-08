import bisect
import copy
import json
import math
import os
import random
import time
from functools import lru_cache
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from vitra.datasets.augment_utils import (
    augmentation_func,
    center_crop_short_side,
    project_to_image_space,
)

from vitra.datasets.interp_utils import interp_mano_state
from vitra.datasets.video_utils import load_video_decord, load_video_cv2
from vitra.datasets.dataset_utils import (
    compute_new_intrinsics_crop, 
    compute_new_intrinsics_resize, 
    calculate_fov,
    ActionFeature,
    StateFeature,
)
from vitra.utils.data_utils import (
    read_dataset_statistics,
    GaussianNormalizer,
)

class EpisodicDatasetCore(object):
    """Core dataset class for episodic hand manipulation data.
    
    Handles loading and processing of video frames, MANO hand parameters,
    and action sequences for hand-centric manipulation tasks.
    """
    def __init__(
        self, 
        video_root, 
        annotation_file, 
        label_folder, 
        training_path=None, 
        statistics_path=None, 
        augmentation=True, 
        flip_augmentation=True, 
        set_none_ratio=0.0, 
        action_type="angle", 
        use_rel=False, 
        upsample_factor=1.0,
        target_image_height=224,
        clip_len=2000,
        state_mask_prob=0.1,
        action_past_window_size=0,
        action_future_window_size=15,
        image_past_window_size=0,
        image_future_window_size=0,
        rel_mode="step",
        load_images=True,
    ):
        self.video_root = video_root                                                # ./data/VITRA_1M/Video/Somethingsomething-v2_root
        annotation_dict = np.load(annotation_file, allow_pickle=True)               # ./data/VITRA_1M/Annotation/ssv2/episode_frame_index.npz
        self.label_folder = label_folder                                            # ./data/VITRA_1M/Annotation/ssv2/episodic_annotations
        self.index_frame_pair       = annotation_dict['index_frame_pair'].copy()    # [(episode_id, frame_id), ...] for episode_id, we want to predict the future starting from frame_id        
        self.index_to_episode_id    = annotation_dict['index_to_episode_id'].copy() # the episode string each index from self.index_frame_pair corresponds to
        
        """Debugging print
        for x in self.index_frame_pair:
            if x[0] == 0:
            print(x)
            [0 0]
            [0 1]
            [0 2]
            [0 3]
            [0 4]
            [0 5]
            [0 6]
            [0 7]
            [0 8]
            [0 9]
            [ 0 10]
            [ 0 11]
            [ 0 12]
            [ 0 13]
            [ 0 14]
            [ 0 15]
            [ 0 16]
            [ 0 17]
            [ 0 18]
            [ 0 19]
            [ 0 20]
            [ 0 21]
            [ 0 22]
            [ 0 23]
        """

        if training_path is not None:
            self.training_idx = np.load(training_path, allow_pickle=True)
            self.num_valid_frames = len(self.training_idx)
        else:
            self.training_idx = None
            self.num_valid_frames = len(self.index_frame_pair)

        if statistics_path is not None:
            self.data_statistics = read_dataset_statistics(statistics_path)

        self.global_data_statistics = None
        self.clip_len = clip_len  # Video clip length in frames
        self.augmentation = augmentation
        self.target_image_height = target_image_height
        self.flip_augmentation = flip_augmentation
        self.set_none_ratio = set_none_ratio
        self.action_type = action_type  # "angle" (Euler angles) or "keypoints" (3D joint positions)
        self.use_rel = use_rel  # Whether to use relative delta as actions for hand poses (MANO poses)
        assert upsample_factor >= 1.0, "only support upsample_factor >= 1.0"
        self.upsample_factor = upsample_factor
        self.state_mask_prob = state_mask_prob  # Probability of masking state input

        self.action_past_window_size=action_past_window_size
        self.action_future_window_size=action_future_window_size
        self.image_past_window_size=image_past_window_size
        self.image_future_window_size=image_future_window_size
        self.rel_mode=rel_mode
        self.load_images=load_images
    def __len__(self):
        return self.num_valid_frames
    
    @staticmethod
    @lru_cache(maxsize=256)          # ~256 MB worst case if each npy ≈1 MB
    def _load_episode_npy(episode_path: str):
        """Load episode data from .npy file with caching.
        
        Uses LRU cache to keep up to 256 episodes in memory (~256 MB worst case).
        The cache automatically purges old entries when full.
        
        Args:
            episode_path: Path to the .npy file containing episode data
            
        Returns:
            Dictionary containing episode information
        """
        return np.load(episode_path, allow_pickle=True).item()

    def _load_or_cache_episode(self, episode_id):
        """
        Returns episode_result (raw dict) and the pre-extracted camera
        extrinsics (R_w2c, t_w2c).  No camera-space MANO tensors are cached.
        """
        root = self.label_folder                            # ./data/VITRA_1M/Annotation/ssv2/episodic_annotations
        epi_path = os.path.join(root, episode_id + '.npy')  # 
        epi = self._load_episode_npy(epi_path)

        extr = epi['extrinsics']                         # world to cam, (T,4,4)
        R_w2c, t_w2c = extr[:, :3, :3], extr[:, :3, 3]

        return epi, R_w2c, t_w2c

    def _mat2euler(self, R_batch: np.ndarray) -> np.ndarray:
        """Batched XYZ-Euler conversion using SciPy."""
        flat = R_batch.reshape(-1, 3, 3)
        eul = R.from_matrix(flat).as_euler('xyz', degrees=False)
        return eul

    def _prepare_smplx_window(self, smplx_dict,
                            R_w2c, t_w2c,
                            idx_window, idx_anchor,
                            *, anchor_frame=True, oob=None, start=None, end=None, upsample_factor=1.0):
        
        # ============================================================
        # 1. TEMPORAL WINDOW EXTENSION: W → W+1
        # ============================================================
        # T = total frames in episode, W = window length (typically 16)
        T, W = len(smplx_dict['body_pose']), len(idx_window)

        # Extend window by 1 frame to enable current/next-frame pairing
        # Example: if idx_window = [5,6,7,...,20], then idx_window_extend = [5,6,7,...,20,21]
        # The extra frame lets us compute (frame_i, frame_i+1) pairs for all W frames
        idx_window_extend = np.append(idx_window, np.clip(idx_window[-1] + 1, start, end))

        # ============================================================
        # 2. EXTRACT SMPLX DATA FOR EXTENDED WINDOW
        # ============================================================

        # Get validity mask for extended window (W+1,) - indicates which frames have valid hand reconstruction
        kept_extend = np.ones([len(idx_window_extend)]).astype(bool)
        
        # Extract MANO parameters for the extended window (all shapes: W+1 × ...)
        smplx_extend = smplx_dict["body_pose"][idx_window_extend]          # (W+1, 63)

        # ============================================================
        # 3. HANDLE OUT-OF-BOUNDS (OOB) FRAMES
        # ============================================================
        # OOB frames are those outside the valid episode range or text-annotated segment
        oob_indices = np.where(oob)[0]
        if len(oob_indices) > 0:
            # Mark OOB frames as invalid - they shouldn't contribute to training
            kept_extend[oob_indices] = False

            # Also check if the extra (W+1-th) frame is out of bounds
            if idx_window[-1] + 1 > end:
                kept_extend[-1] = False

        # For invalid frames, reset hand pose to identity (canonical MANO rest pose)
        # This prevents corrupted/extrapolated data from being used
        if not np.all(kept_extend):
            zeros = np.zeros((63), dtype=smplx_extend.dtype)
            smplx_extend[~kept_extend] = zeros

        # ============================================================
        # 4. TRANSFORM WORLD SPACE → CAMERA SPACE
        # ============================================================

        R_cam_extend = R_w2c[idx_anchor] #@ np.eye(3)[None,...]  # (W+1, 3, 3)

        t_cam_extend = t_w2c[idx_anchor]  # (W+1, 3)

        # ============================================================
        # 5. CONVERT SMPLX POSES FROM EULER ANGLES to ROTATION MATRICES (BATCHED)
        # ============================================================

        smplx_P_extend  = R.from_euler('xyz', smplx_extend.reshape(-1,3)).as_matrix().reshape(-1,21,3,3)

        # ============================================================
        # 8. SPLIT INTO CURRENT and NEXT-FRAME SEQUENCES
        # ============================================================
        # Create paired representations for action prediction
        # Given W+1 frames [f0, f1, f2, ..., fW], we create W pairs:
        #   Current: [f0, f1, f2, ..., f(W-1)]  (indices 0 to W-1)
        #   Next:    [f1, f2, f3, ..., fW]      (indices 1 to W)
        # This enables learning: Given state at time t, predict state at time t+1
        
        # Current frame tensors (first W frames: indices 0 to W-1)
        R_cam = R_cam_extend[:-1]                       # (W, 3, 3)
        t_cam = t_cam_extend[:-1]                       # (W, 3)
        smplx_euler = smplx_extend[:-1]                 # (W, 63)
        smplx_P = smplx_P_extend[:-1]                   # (W, 21, 3, 3)
        kept = kept_extend[:-1]                         # (W,)

        # Next frame tensors (last W frames: indices 1 to W, shifted by 1 timestep)
        R_cam_next = R_cam_extend[1:]                          # (W, 3, 3)
        t_cam_next = t_cam_extend[1:]                          # (W, 3)
        smplx_euler_next = smplx_extend[1:]                    # (W, 63)
        smplx_P_next = smplx_P_extend[1:]                   # (W, 21, 3, 3)
        kept_next = kept_extend[1:]                            # (W,)

        # ============================================================
        # 9. RETURN PAIRED CURRENT/NEXT REPRESENTATIONS
        # ============================================================
        return dict(
            # Current frame tensors (conditioning for the model)
            R_cam       = R_cam.astype(np.float32),                    # Wrist rotation in camera space
            t_cam       = t_cam.astype(np.float32),                    # Wrist translation in camera space
            smplx_euler = smplx_euler.astype(np.float32),        # SMPLX angles (Euler)
            smplx_P     = smplx_P.astype(np.float32),                # SMPLX rotation matrices
            kept        = kept,                                         # Valid frame mask

            # Next-frame tensors (prediction targets, same length W)
            R_cam_next          = R_cam_next.astype(np.float32),
            t_cam_next          = t_cam_next.astype(np.float32),
            smplx_euler_next    = smplx_euler_next.astype(np.float32),
            smplx_P_next        = smplx_P_next.astype(np.float32),
            kept_next           = kept_next,
        )

    def _prepare_side_window(self, side_dict,
                            R_w2c, t_w2c,
                            idx_window, idx_anchor,
                            *, anchor_frame=True, oob=None, start=None, end=None, upsample_factor=1.0):
        """
        Prepares hand pose data for ONE hand (left or right) across a temporal window.
        Transforms MANO hand parameters from world space to camera space and creates
        paired current/next-frame representations for action prediction.
        
        Args:
            side_dict: Dictionary containing MANO hand reconstruction for one hand:
                - 'global_orient_worldspace': (T,3,3) wrist rotation matrices in world coords
                - 'transl_worldspace': (T,3) wrist translation in world coords
                - 'hand_pose': (T,15,3,3) finger joint rotations (15 joints × 3×3 rotation matrix)
                - 'joints_worldspace': (T,21,3) hand keypoint positions in world coords
                - 'kept_frames': (T,) binary mask for valid hand reconstructions
            R_w2c: (T,3,3) world-to-camera rotation matrices for all frames
            t_w2c: (T,3) world-to-camera translation vectors for all frames
            idx_window: (W,) array of frame indices in the temporal window (typically W=16)
            idx_anchor: Scalar index of the anchor frame (current observation frame)
            oob: (W,) boolean array marking out-of-bounds frames in the window
            start: Start index of valid episode range
            end: End index of valid episode range
            upsample_factor: Temporal upsampling factor (>1 for smoother trajectories)
            
        Returns:
            Dictionary with paired current/next-frame tensors (each of length W):
                - R_cam, t_cam: Wrist pose in camera space
                - pose_euler: Finger angles in Euler format (45 dims = 15 joints × 3 angles)
                - hand_P: Finger rotation matrices (15×3×3)
                - joints_manospace: Keypoints in MANO canonical space (21×3)
                - kept: Valid frame mask
                - *_next: Same representations for the next timestep
        """
        
        # ============================================================
        # 1. TEMPORAL WINDOW EXTENSION: W → W+1
        # ============================================================
        # T = total frames in episode, W = window length (typically 16)
        T, W = len(side_dict['global_orient_worldspace']), len(idx_window)
        
        # Extend window by 1 frame to enable current/next-frame pairing
        # Example: if idx_window = [5,6,7,...,20], then idx_window_extend = [5,6,7,...,20,21]
        # The extra frame lets us compute (frame_i, frame_i+1) pairs for all W frames
        idx_window_extend = np.append(idx_window, np.clip(idx_window[-1] + 1, start, end))

        # ============================================================
        # 2. EXTRACT MANO HAND DATA FOR EXTENDED WINDOW
        # ============================================================
        # Get validity mask for extended window (W+1,) - indicates which frames have valid hand reconstruction
        kept_extend = side_dict['kept_frames'][idx_window_extend].astype(bool)

        # Extract MANO parameters for the extended window (all shapes: W+1 × ...)
        R_mano_extend = side_dict['global_orient_worldspace'][idx_window_extend]  # (W+1, 3, 3) wrist rotations in world space
        t_mano_extend = side_dict['transl_worldspace'][idx_window_extend]          # (W+1, 3) wrist translations in world space
        hand_P_extend = side_dict['hand_pose'][idx_window_extend]                  # (W+1, 15, 3, 3) finger joint rotations
        joints_worldspace_extend = side_dict['joints_worldspace'][idx_window_extend]  # (W+1, 21, 3) hand keypoints in world space

        # ============================================================
        # 3. HANDLE OUT-OF-BOUNDS (OOB) FRAMES
        # ============================================================
        # OOB frames are those outside the valid episode range or text-annotated segment
        oob_indices = np.where(oob)[0]
        if len(oob_indices) > 0:
            # Mark OOB frames as invalid - they shouldn't contribute to training
            kept_extend[oob_indices] = False

            # Also check if the extra (W+1-th) frame is out of bounds
            if idx_window[-1] + 1 > end:
                kept_extend[-1] = False

        # For invalid frames, reset hand pose to identity (canonical MANO rest pose)
        # This prevents corrupted/extrapolated data from being used
        if not np.all(kept_extend):
            identity = np.eye(3, dtype=hand_P_extend.dtype)  # 3×3 identity matrix
            # Create identity block for all 15 finger joints
            identity_block = np.broadcast_to(identity, (hand_P_extend.shape[1], 3, 3))  # (15, 3, 3)
            hand_P_extend[~kept_extend] = identity_block  # Reset invalid finger poses to identity
            R_mano_extend[~kept_extend] = identity        # Reset invalid wrist rotations to identity

        # ============================================================
        # 4. TRANSFORM WORLD SPACE → CAMERA SPACE
        # ============================================================
        # **Critical step**: Transform hand pose from world coordinates to camera coordinates
        # This makes hand poses relative to the current camera viewpoint (anchor frame)
        # All W+1 frames are transformed using the SAME anchor camera's extrinsics
        
        # Wrist rotation: R_cam = R_world2cam @ R_mano_world
        R_cam_extend  = R_w2c[idx_anchor] @ R_mano_extend  # (W+1, 3, 3)
        
        # Wrist translation: t_cam = R_world2cam @ t_mano_world + t_world2cam
        t_cam_extend  = (R_w2c[idx_anchor] @ t_mano_extend[..., None])[..., 0] + t_w2c[idx_anchor]  # (W+1, 3)
        

        # ============================================================
        # 5. CONVERT FINGER POSES TO EULER ANGLES (BATCHED)
        # ============================================================
        # Convert 15 finger joint rotation matrices to Euler angles for neural network processing
        # Input: hand_P_extend (W+1, 15, 3, 3) → flatten to (W+1*15, 3, 3)
        # Convert each 3×3 rotation → 3 Euler angles (xyz convention)
        # Output: (W+1, 45) where 45 = 15 joints × 3 angles
        pose_euler_extend  = R.from_matrix(hand_P_extend.reshape(-1,3,3))     \
                            .as_euler('xyz', degrees=False).reshape(-1,45)
        
        # ============================================================
        # 6. TRANSFORM KEYPOINTS TO MANO CANONICAL SPACE (BATCHED)
        # ============================================================
        # Convert 21 hand keypoints from world space → MANO canonical space (root-relative)
        # This representation is invariant to global hand position/rotation
        # Formula: joints_mano = R_mano^T @ (joints_world - t_mano)
        # - First: translate keypoints by negative wrist translation
        # - Then: apply inverse wrist rotation (R^T) to align with MANO coordinate frame
        joints_manospace_extend = (R_mano_extend.transpose(0, 2, 1) @ 
                                  (joints_worldspace_extend.transpose(0, 2, 1) - t_mano_extend[..., None])
                                  ).transpose(0,2,1)  # (W+1, 21, 3)

        # ============================================================
        # 7. OPTIONAL TEMPORAL UPSAMPLING
        # ============================================================
        # If upsample_factor > 1, interpolate between frames using PCHIP (Piecewise Cubic Hermite)
        # This creates smoother action trajectories for better motion prediction
        if upsample_factor > 1:
            # Interpolate all MANO state variables
            R_cam_extend, t_cam_extend, hand_P_extend, joints_manospace_extend, kept_extend = \
                interp_mano_state(R_cam_extend, t_cam_extend, hand_P_extend, 
                                 joints_manospace_extend, kept_extend, 
                                 upsample_factor, method="pchip")

            # Recompute Euler angles after interpolation
            pose_euler_extend = R.from_matrix(hand_P_extend.reshape(-1,3,3))     \
                            .as_euler('xyz', degrees=False).reshape(-1,45)
            
            # Truncate back to exactly W+1 frames (interpolation may create more)
            R_cam_extend = R_cam_extend[:W+1]
            t_cam_extend = t_cam_extend[:W+1]
            hand_P_extend = hand_P_extend[:W+1]
            pose_euler_extend = pose_euler_extend[:W+1]
            joints_manospace_extend = joints_manospace_extend[:W+1]
            kept_extend = kept_extend[:W+1]
        
        # ============================================================
        # 8. SPLIT INTO CURRENT and NEXT-FRAME SEQUENCES
        # ============================================================
        # Create paired representations for action prediction
        # Given W+1 frames [f0, f1, f2, ..., fW], we create W pairs:
        #   Current: [f0, f1, f2, ..., f(W-1)]  (indices 0 to W-1)
        #   Next:    [f1, f2, f3, ..., fW]      (indices 1 to W)
        # This enables learning: Given state at time t, predict state at time t+1
        
        # Current frame tensors (first W frames: indices 0 to W-1)
        R_cam = R_cam_extend[:-1]                      # (W, 3, 3)
        t_cam = t_cam_extend[:-1]                      # (W, 3)
        pose_euler = pose_euler_extend[:-1]            # (W, 45)
        hand_P = hand_P_extend[:-1]                    # (W, 15, 3, 3)
        joints_manospace = joints_manospace_extend[:-1] # (W, 21, 3)
        kept = kept_extend[:-1]                        # (W,)

        # Next frame tensors (last W frames: indices 1 to W, shifted by 1 timestep)
        R_cam_next = R_cam_extend[1:]                          # (W, 3, 3)
        t_cam_next = t_cam_extend[1:]                          # (W, 3)
        pose_euler_next = pose_euler_extend[1:]                # (W, 45)
        hand_P_next = hand_P_extend[1:]                        # (W, 15, 3, 3)
        joints_manospace_next = joints_manospace_extend[1:]    # (W, 21, 3)
        kept_next = kept_extend[1:]                            # (W,)

        # ============================================================
        # 9. RETURN PAIRED CURRENT/NEXT REPRESENTATIONS
        # ============================================================
        return dict(
            # Current frame tensors (conditioning for the model)
            R_cam=R_cam.astype(np.float32),                    # Wrist rotation in camera space
            t_cam=t_cam.astype(np.float32),                    # Wrist translation in camera space
            pose_euler=pose_euler.astype(np.float32),          # Finger angles (Euler)
            hand_P=hand_P.astype(np.float32),                  # Finger rotation matrices
            joints_manospace = joints_manospace.astype(np.float32),  # Keypoints in MANO space
            kept=kept,                                         # Valid frame mask

            # Next-frame tensors (prediction targets, same length W)
            R_cam_next = R_cam_next.astype(np.float32),
            t_cam_next = t_cam_next.astype(np.float32),
            pose_euler_next = pose_euler_next.astype(np.float32),
            hand_P_next = hand_P_next.astype(np.float32),
            joints_manospace_next = joints_manospace_next.astype(np.float32),
            kept_next = kept_next,
        )
    
    # ============================================================
    #  4.  Vectorised action-window constructor (ONE hand)
    # ============================================================
    def _make_action_window_vec(self, win, anchor_idx: int, *, rel_mode="step", action_type="angle"):
        """
        anchor_idx : the position of t0 inside the window (usually = past)
        rel_mode   : "step"   → Δ(t→t+1)
                    "anchor" → Δ(t→t0)
        action_type: "angle"  → Euler angles (xyz)
                    "keypoints " → root keypoints (21x3=63)
        """

        if "smplx_euler" in win:

            R_cur, t_cur            = win['R_cam'],        win['t_cam']
            R_nxt, t_nxt            = win['R_cam_next'],   win['t_cam_next']
            P_cur, P_nxt    = win['smplx_P'],   win['smplx_P_next']
            pose_next              = win['smplx_euler_next']
            kept, kept_n            = win['kept'],         win['kept_next']
            W = len(t_cur)

            # absolute pose of t+1
            if action_type == "keypoints":
                abs_next = kpoints_root_next.reshape(W, -1)
            elif action_type == "angle":
                abs_next = pose_next
            action_abs = np.concatenate(
                [t_nxt,
                self._mat2euler(R_nxt),
                abs_next],
                axis=-1).astype(np.float32)
            action_abs = action_abs.reshape(W, -1)

            # choose relative formulation
            if rel_mode == "step":
                t_rel = t_nxt - t_cur
                R_rel = R_nxt @ R_cur.transpose(0,2,1)
                P_rel = np.matmul(P_nxt, P_cur.transpose(0,1,3,2))
                valid = kept & kept_n

            elif rel_mode == "anchor":
                t_anchor  = t_cur[anchor_idx]
                R_anchor  = R_cur[anchor_idx]
                P_anchor  = P_cur[anchor_idx]

                # broadcast to all W rows
                t_rel = t_nxt - t_anchor
                R_rel = R_nxt @ R_anchor.T
                P_rel = np.matmul(P_nxt, P_anchor.transpose(0,2,1))
                valid = kept_n & kept[anchor_idx]

            else:
                raise ValueError('rel_mode must be "step" or "anchor"')

            pose_rel = R.from_matrix(P_rel.reshape(-1,3,3)) \
                        .as_euler('xyz',False).reshape(W,63)

            action_rel = np.concatenate(
                [t_rel,
                self._mat2euler(R_rel),
                pose_rel],
                axis=-1).astype(np.float32)

            action_abs[~valid] = 0.0
            action_rel[~valid] = 0.0
            return action_abs, action_rel, valid

        else:

            R_cur, t_cur  = win['R_cam'],        win['t_cam']       # [16, 3, 3] [16, 3]
            R_nxt, t_nxt  = win['R_cam_next'],   win['t_cam_next']  # [16, 3, 3] [16, 3]
            P_cur, P_nxt  = win['hand_P'],       win['hand_P_next'] # [16, 15, 3, 3] [16, 15, 3, 3]
            pose_next     = win['pose_euler_next']                  # [16, 45]
            kpoints_root_next = win['joints_manospace_next']        # [16, 21, 3]
            kept, kept_n  = win['kept'], win['kept_next']           # [16] [16]
            W = len(t_cur)                                          # 16

            # absolute pose of t+1
            if action_type == "keypoints":
                abs_next = kpoints_root_next.reshape(W, -1)
            elif action_type == "angle":
                abs_next = pose_next    # [16, 45]
            action_abs = np.concatenate(
                [t_nxt,
                self._mat2euler(R_nxt),
                abs_next],
                axis=-1).astype(np.float32)
            action_abs = action_abs.reshape(W, -1)

            # choose relative formulation
            if rel_mode == "step":
                t_rel = t_nxt - t_cur
                R_rel = R_nxt @ R_cur.transpose(0,2,1)
                P_rel = np.matmul(P_nxt, P_cur.transpose(0,1,3,2))
                valid = kept & kept_n

            elif rel_mode == "anchor":
                t_anchor  = t_cur[anchor_idx]
                R_anchor  = R_cur[anchor_idx]
                P_anchor  = P_cur[anchor_idx]

                # broadcast to all W rows
                t_rel = t_nxt - t_anchor
                R_rel = R_nxt @ R_anchor.T
                P_rel = np.matmul(P_nxt, P_anchor.transpose(0,2,1))
                valid = kept_n & kept[anchor_idx]

            else:
                raise ValueError('rel_mode must be "step" or "anchor"')

            pose_rel = R.from_matrix(P_rel.reshape(-1,3,3)) \
                        .as_euler('xyz',False).reshape(W,45)

            action_rel = np.concatenate(
                [t_rel,
                self._mat2euler(R_rel),
                pose_rel],
                axis=-1).astype(np.float32)

            action_abs[~valid] = 0.0
            action_rel[~valid] = 0.0
            return action_abs, action_rel, valid

    def _window_indices(self, frame_id, past, future, start, end):
        """
        Returns:
            idx_clip : (W,) indices clipped to [0, T-1]
            oob      : (W,) bool — slots that were originally OOB
        """
        win = np.arange(-past, future + 1) + frame_id                  # (W,)
        oob = (win < start) | (win > end)
        return win.clip(start, end), oob

    def _resolve_video_path(self, dataset_name: str = None, video_name: str = None, part_index: int = None) -> str:
        if dataset_name=='Ego4D':
            if self.clip_len is not None:
                video_path = os.path.join(self.video_root, video_name + '_part' + str(part_index+1) +'.mp4')
            else:
                video_path = os.path.join(self.video_root, video_name +'.mp4')
            return video_path
        elif dataset_name=='EgoExo4D':
            if self.clip_len is not None:
                video_path = os.path.join(self.video_root, video_name +'_part' + str(part_index+1) +'.mp4')
            else:
                video_path = os.path.join(self.video_root, video_name +'.mp4')
            return video_path
        elif dataset_name == 'epic':
            video_id = video_name.split('_')[0]
            if self.clip_len is not None:
                video_path = os.path.join(self.video_root, video_name+ '_part' + str(part_index+1) + '.mp4')
            else:
                video_path = os.path.join(self.video_root, video_name+ '.MP4')
            return video_path
        elif dataset_name == 'somethingsomethingv2':
            if self.clip_len is not None:
                video_path = os.path.join(self.video_root, video_name+ '_part' + str(part_index+1) + '.mp4')
            else:
                video_path = os.path.join(self.video_root, video_name+'.webm')
            return video_path
        else:
            raise ValueError(f'Unknown dataset prefix {dataset_name}')

    def _mano_forward(self, betas, pose_m):
        """Runs MANO once and returns (vertices, joints) on CPU NumPy."""
        beta_t  = torch.tensor(betas).unsqueeze(0).float().cuda()
        pose_t  = torch.tensor(pose_m).unsqueeze(0).float().cuda()
        out     = mano(betas=beta_t, hand_pose=pose_t)       # no global_orient
        return out.vertices[0].cpu().numpy(), out.joints[0].cpu().numpy()

    # ------------------------------------------------------------
    #  Grab the (past + future + 1) frame window
    # ------------------------------------------------------------

    def _pack_state(self, R_cam, t_cam, pose_euler, idx):
        return np.concatenate([t_cam[idx],
                            self._mat2euler(R_cam[idx][None,...])[0],
                            pose_euler[idx]])

    def _grab_window_images(self,
                            episode_id: str,
                            epi: dict,
                            frame_id: int,
                            past: int,
                            future: int
                            ):
        """
        Returns
        -------
        images : (L, H, W, 3)  uint8   – raw RGB frames
        mask   : (L,) bool                True where real frame, False where pad
        """
        dataset_name = episode_id.split('_')[0]
        # video_path   = self._resolve_video_path(dataset_name, epi['video_name'])
        decode_table = epi['video_decode_frame']                      # (T,)
        T            = len(decode_table)
        frame_in_video = epi['video_decode_frame'][frame_id]

        # ---------- build padded window indices ---------------------------
        # Not support multiple images now
        idx_win, oob = self._window_indices(frame_id, past, future, 0, T-1)     # (W,)

        if self.clip_len is not None:
            part_idx = frame_in_video // self.clip_len # clip_len = 2000
            frame_in_part = frame_in_video % self.clip_len
            video_path = self._resolve_video_path(dataset_name, epi['video_name'], part_idx)
            decode_ids = [frame_in_part]
        else:
            video_path = self._resolve_video_path(dataset_name, epi['video_name'])
            decode_ids = decode_table[idx_win]

        # ---------- read images --------------------
        # Retry mechanism: try up to 3 times to load video frames
        for attempt in range(3):
            try:
                imgs, _ = load_video_decord(video_path, frame_index=decode_ids, rotation=False)
                break  # Success, exit the retry loop
            except Exception as e:
                # if attempt == 2:
                #     raise  # Raise the exception after 3 failed attempts
                print(f"Warning: failed to load video frames from {video_path} (attempt {attempt+1}/3): {e}")
                time.sleep(0.1)

        images = np.stack(imgs, axis=0)           # (L,H,W,3) uint8
        mask   = ~oob                             # (L,) bool

        return images, mask

    def _find_matching_texts(self, text_list, frame_id):
        """Find text annotations that overlap with the given frame.
        
        Args:
            text_list: List of tuples (text, (start_frame, end_frame))
            frame_id: Current frame ID to check
        
        Returns:
            matching_texts: List of matching text annotations
            matching_ranges: List of corresponding time ranges (start_frame, end_frame)

        Note:
            Uses half-open interval [start_frame, end_frame)
        """
        matching_texts = []
        matching_ranges = []
        
        for text, (start_frame, end_frame) in text_list:
            # Check if frame_id is in the half-open interval [start_frame, end_frame)
            if start_frame <= frame_id < end_frame:
                matching_texts.append(text)
                matching_ranges.append((start_frame, end_frame))
        
        return matching_texts, matching_ranges

    def _random_select_text(
        self,
        text,
        text_rephrase,
        hand_type,
        clip_idx,
    ):

        text_list = [text]
        if text_rephrase and isinstance(text_rephrase[hand_type][clip_idx][0], list):
            text_list.extend(text_rephrase[hand_type][clip_idx][0])

        text_selected = random.choice(text_list).strip()
        if not text_selected.endswith('.'):
            text_selected += '.'
        return text_selected

    def _build_instruction(
        self,
        main_type,
        text_clip,
        text_rephrase,
        idx_win,
        oob,
        epi_len, # T
        frame_id,
        action_past_window_size,
        action_future_window_size,
    ):

        sub_type = 'right' if main_type == 'left' else 'left'
        
        # Build main text
        # text_clip[main_type][0]: ('Place the pink cup on the table.', (0, 26))
        main_text_selected = self._random_select_text(
            text_clip[main_type][0][0],
            text_rephrase,
            main_type,
            clip_idx=0,
        )

        # Build sub text
        sub_text_list = text_clip[sub_type]
        has_sub_text = len(sub_text_list) > 0
        
        sub_oob, sub_idx_win = oob, idx_win
        sub_text_selected = "None."
        sub_win = (0, epi_len)  # Default to the full range if no text available

        if has_sub_text:
            sub_matching_texts, sub_matching_ranges = self._find_matching_texts(sub_text_list, frame_id)
            if len(sub_matching_texts) > 0:
                selected_idx = random.randrange(len(sub_matching_texts))
                sub_win = sub_matching_ranges[selected_idx]
                sub_idx_win, sub_oob = self._window_indices(
                    frame_id,
                    action_past_window_size,
                    action_future_window_size, sub_win[0], sub_win[1]-1
                )     # (W,)

                sub_text_selected = self._random_select_text(
                    sub_matching_texts[selected_idx].strip(),
                    text_rephrase,
                    sub_type,
                    clip_idx=selected_idx,
                )

        # Assign left/right based on main_type
        is_main_left = (main_type == 'left')

        idx_win_left = idx_win if is_main_left else sub_idx_win
        idx_win_right = sub_idx_win if is_main_left else idx_win
        oob_left = oob if is_main_left else sub_oob
        oob_right = sub_oob if is_main_left else oob
        
        text_left = main_text_selected if is_main_left else sub_text_selected
        text_right = sub_text_selected if is_main_left else main_text_selected
        
        start_left = 0 if is_main_left else (sub_win[0] if has_sub_text else 0)
        start_right = (sub_win[0] if has_sub_text else 0) if is_main_left else 0
        end_left = epi_len - 1 if is_main_left else (sub_win[1] - 1 if has_sub_text else epi_len - 1)
        end_right = (sub_win[1] - 1 if has_sub_text else epi_len - 1) if is_main_left else epi_len - 1

        instruction = f"Left hand: {text_left} Right hand: {text_right}"

        return instruction, idx_win_left, oob_left, idx_win_right, oob_right, start_left, end_left, start_right, end_right

    def _get_2d_traj_cur_to_end(self, idx_frame, epi, intrinsics, hand_type, image_size):
        """Get the 2D trajectory of the hand palm from current frame to episode end.
        
        Args:
            idx_frame: Current frame index
            epi: Episode data dictionary
            intrinsics: Camera intrinsic matrix
            hand_type: 'left' or 'right' hand
            image_size: (H, W) tuple of image dimensions
            
        Returns:
            Normalized 2D palm trajectory in image space [0, 1]
        """
        H, W = image_size
        # intrinsics = epi['intrinsics'].copy()
        intrinsics = intrinsics.copy()
        # normalize intrinsics
        intrinsics[0] /= intrinsics[0,2]*2
        intrinsics[1] /= intrinsics[1,2]*2

        hand_joints_cur_to_end = epi[hand_type]['joints_worldspace'][idx_frame:] # (N, 21, 3)
        hand_palm_cur_to_end = np.mean(hand_joints_cur_to_end[:, [0,2,5,9,13,17], :], axis=1, keepdims=True) # (N, 1, 3)

        extrinsics = epi['extrinsics'].copy()
        extrinsics_cur = extrinsics[idx_frame] # world to cam
        R_world_to_cam = extrinsics_cur[None, :3, :3].repeat(len(hand_palm_cur_to_end), axis=0)
        t_world_to_cam = extrinsics_cur[None, :3, 3:].repeat(len(hand_palm_cur_to_end), axis=0)

        hand_palm_cur_to_end_cam = (R_world_to_cam @ hand_palm_cur_to_end.transpose(0, 2, 1) + t_world_to_cam).transpose(0, 2, 1)

        uv_palm_cur_to_end = project_to_image_space(hand_palm_cur_to_end_cam, intrinsics, (H, W)) # (N, M, 2)
        uv_palm_cur_to_end[..., 0] = np.clip(uv_palm_cur_to_end[..., 0], 0, W)
        uv_palm_cur_to_end[..., 1] = np.clip(uv_palm_cur_to_end[..., 1], 0, H)

        uv_palm_cur_to_end = uv_palm_cur_to_end.reshape(-1, 2)
        uv_palm_cur_to_end = uv_palm_cur_to_end.astype(np.float32)
        uv_palm_cur_to_end[:,0] /= W
        uv_palm_cur_to_end[:,1] /= H

        return uv_palm_cur_to_end

    def get_item_frame(
            self, episode_id, frame_id,
            action_past_window_size=0, 
            action_future_window_size=0,
            image_past_window_size=0, 
            image_future_window_size=0,
            rel_mode: str = "step",
            load_images: bool = True,
        ):
        """
        Vectorised dataloader.

        """
        # ------------------------------------------------------------------
        # 1. Load episode dict  +  extrinsics
        # https://github.com/microsoft/VITRA/blob/main/data/data.md#4-metadata-structure
        # ------------------------------------------------------------------
        
        # R_w2c [T,3,3], t_w2c [T,3]
        epi, R_w2c, t_w2c = self._load_or_cache_episode(episode_id)
        T  = len(epi['extrinsics']) # number of frames in the episode

        # ------------------------------------------------------------------
        # 2. Build frame-window indices
        # ------------------------------------------------------------------
        # idx_win: (W,)  oob: (W,)
        # idx_win are the indices clipped to [0,T-1]
        # oob out-of-bounds mask for the window
        idx_win, oob  = self._window_indices(frame_id,
                                        action_past_window_size,
                                        action_future_window_size, 0, T-1)
        W   = len(idx_win) # W = 16
        
        if "smplx_params" in epi:
            main_type   = None
            sub_type    = None

            instruction = None

            # only the body to consider so we can reuse
            idx_body    = idx_win
            oob_body    = oob
            start_body  = 0
            end_body    = T - 1

            win_body  = self._prepare_smplx_window(
                epi['smplx_params'],  R_w2c, t_w2c, idx_body, frame_id, anchor_frame=True, 
                oob=oob_body, start=start_body, end=end_body, upsample_factor=self.upsample_factor
            )
            idx_center = action_past_window_size          # local index of t0 in window

            abs_L, rel_L, msk_L = self._make_action_window_vec(
                win_body,  anchor_idx=idx_center, rel_mode=rel_mode, action_type=self.action_type
            ) 

            action_abs = np.concatenate([abs_L, abs_R], axis=1)   # (W,action_dim)
            action_rel = np.concatenate([rel_L, rel_R], axis=1)   # (W,102)
            action_mask = np.stack([msk_L, msk_R], axis=1)        # (W,2)

            cur = self._pack_state(win_body['R_cam'],
                        win_body['t_cam'],
                        win_body['smplx_euler'] if self.action_type=='angle' else win_body['smplx_curr'].reshape(W, -1),
                        idx_center)

            betas = []
            current_state = np.concatenate([cur, betas])          # 
            current_state_mask  = np.array([win_body['kept'][idx_center]])

            # ------------------------------------------------------------------
            # 5. RGB window
            # ------------------------------------------------------------------
            if load_images:
                image_list, image_mask = self._grab_window_images(
                    episode_id, epi,
                    frame_id,
                    image_past_window_size,
                    image_future_window_size
                )
                H = image_list[0].shape[0]
                W = image_list[0].shape[1]
            else:
                image_list = None
                image_mask = None
                H, W = epi['intrinsics'][1,2]*2, epi['intrinsics'][0,2]*2

            # ------------------------------------------------------------------
            # 6. Calculate New_intrinsics
            # ------------------------------------------------------------------

            dataset_name = episode_id.split('_')[0]
            intrinsics = epi['intrinsics']
            new_intrinsics = compute_new_intrinsics_resize(intrinsics, (H, W))

            fov = calculate_fov( 2 * new_intrinsics[1][2], 2 * new_intrinsics[0][2], new_intrinsics)

            if self.use_rel:
                action_list = action_rel
            else:
                # use abs action for hand pose only
                rel = action_rel[:, :action_rel.shape[1]//2]
                abs = action_abs[:, :action_abs.shape[1]//2]
                
                action_list = np.concatenate([rel[:, :6], abs[:, 6:]], axis=1)

            # ------------------------------------------------------------------
            # 8. Return to caller
            # ------------------------------------------------------------------

            result_dict = dict(
                instruction             = instruction,
                action_list             = action_list,          # (W,2*51) float32
                action_mask             = action_mask,          # (W,2)   bool
                current_state           = current_state,        # (2*61,)  float32
                current_state_mask      = current_state_mask,   # (2,) bool
                fov                     = fov,                  # (2,) float32
                intrinsics              = new_intrinsics,       # (3,3) float32
            )
            
            if image_list is not None:
                result_dict['image_list'] = image_list          # (W,H,W,3) uint8
            if image_mask is not None:
                result_dict['image_mask'] = image_mask          # (W,) bool

        else:

            main_type = epi['anno_type']
            sub_type = 'right' if main_type == 'left' else 'left'

            # ------------------------------------------------------------------
            # 3. Build instruction text
            # ------------------------------------------------------------------
            instruction, idx_win_left, oob_left, idx_win_right, oob_right, \
            start_left, end_left, start_right, end_right = self._build_instruction(
                main_type = main_type,
                text_clip = epi['text'],
                text_rephrase = epi.get('text_rephrase'),
                idx_win = idx_win,
                oob = oob,
                epi_len = T,
                frame_id = frame_id,
                action_past_window_size = action_past_window_size,
                action_future_window_size = action_future_window_size,
            )
            
            # ------------------------------------------------------------------
            # 4. Vectorised actions  (left + right)
            # ------------------------------------------------------------------
            win_left  = self._prepare_side_window(
                epi['left'],  R_w2c, t_w2c, idx_win_left, frame_id, anchor_frame=True, 
                oob=oob_left, start=start_left, end=end_left, upsample_factor=self.upsample_factor
            )
            win_right = self._prepare_side_window(
                epi['right'], R_w2c, t_w2c, idx_win_right, frame_id, anchor_frame=True, 
                oob=oob_right, start=start_right, end=end_right, upsample_factor=self.upsample_factor
            )
            idx_center = action_past_window_size          # local index of t0 in window
            
            # rel_mode: "step"  or  "anchor" / action_type: "angle" or "keypoints"
            # step: relative to previous frame, anchor: relative to t0
            abs_L, rel_L, msk_L = self._make_action_window_vec(
                win_left,  anchor_idx=idx_center, rel_mode=rel_mode, action_type=self.action_type
            ) 

            abs_R, rel_R, msk_R = self._make_action_window_vec(
                win_right, anchor_idx=idx_center, rel_mode=rel_mode, action_type=self.action_type
            ) 

            action_abs = np.concatenate([abs_L, abs_R], axis=1)   # (W,action_dim)
            action_rel = np.concatenate([rel_L, rel_R], axis=1)   # (W,102)
            action_mask = np.stack([msk_L, msk_R], axis=1)        # (W,2)

            cur_L = self._pack_state(win_left['R_cam'],
                        win_left['t_cam'],
                        win_left['pose_euler'] if self.action_type=='angle' else win_left['joints_manospace'].reshape(W, -1),
                        idx_center)

            cur_R = self._pack_state(win_right['R_cam'],
                                win_right['t_cam'],
                                win_right['pose_euler'] if self.action_type=='angle' else win_right['joints_manospace'].reshape(W, -1),
                                idx_center)

            betas_L = epi['left']['beta']
            betas_R = epi['right']['beta']

            current_state       = np.concatenate([cur_L, betas_L, cur_R, betas_R])          # 2 * (6+action_dim+10,)
            # current_state_mask  = np.array([msk_L[idx_center],
            #                                 msk_R[idx_center]])
            current_state_mask  = np.array([win_left['kept'][idx_center], win_right['kept'][idx_center]])

            # ------------------------------------------------------------------
            # 5. RGB window
            # ------------------------------------------------------------------
            if load_images:
                image_list, image_mask = self._grab_window_images(
                    episode_id, epi,
                    frame_id,
                    image_past_window_size,
                    image_future_window_size
                )
                H = image_list[0].shape[0]
                W = image_list[0].shape[1]
            else:
                image_list = None
                image_mask = None
                H, W = epi['intrinsics'][1,2]*2, epi['intrinsics'][0,2]*2
            
            # ------------------------------------------------------------------
            # 6. Calculate New_intrinsics
            # ------------------------------------------------------------------
            dataset_name = episode_id.split('_')[0]
            intrinsics = epi['intrinsics']

            if dataset_name == 'EgoExo4D':
                # For EgoExo4D, the fisheye camera images contain black borders after undistortion.
                # We remove these borders using a center crop. Specifically, the video frames are
                # first resized from 1408 to 448, and then center-cropped to 256.

                new_intrinsics = compute_new_intrinsics_crop(intrinsics, 1408, 256/448*1408, H)
                
            else:
                new_intrinsics = compute_new_intrinsics_resize(intrinsics, (H, W))

            # ------------------------------------------------------------------
            # 7. Do augmentation
            # ------------------------------------------------------------------
            if self.augmentation:
                try:
                    # randomly sample aspect ratio for augmentation
                    aspect_ratio = np.exp(random.uniform(np.log(1.0), np.log(2.0)))
                    target_size = (int(self.target_image_height * aspect_ratio), self.target_image_height)  # (W, H)
                    augment_params = {
                        'tgt_aspect': aspect_ratio, 
                        'flip_augmentation': self.flip_augmentation, 
                        'set_none_ratio': self.set_none_ratio,
                    }

                    uv_traj = self._get_2d_traj_cur_to_end(frame_id, epi, new_intrinsics, main_type, (H, W))
                    image_list, new_intrinsics, (action_abs, action_rel, action_mask), \
                    (current_state, current_state_mask), instruction = \
                        augmentation_func(
                            image = image_list, 
                            intrinsics = new_intrinsics,
                            actions = (action_abs, action_rel, action_mask),
                            states = (current_state, current_state_mask),
                            captions = instruction,
                            uv_traj = uv_traj,
                            target_size = target_size,
                            augment_params = augment_params,
                            sub_type = sub_type,
                        )

                except Exception as e:
                    print(f"Warning: Augmentation failed for episode {episode_id}, frame {frame_id}: {e}. Do center crop only")
                    import traceback
                    print(f"Warning: Augmentation failed for episode {episode_id}, frame {frame_id}")
                    print(f"Exception: {type(e).__name__}: {e}")
                    print(f"Traceback:\n{traceback.format_exc()}")
                    image_list = center_crop_short_side(image_list[0])[None, ...]
                    new_intrinsics[0][2] = 0.5 * image_list[0].shape[1]  # update the principal point
                    new_intrinsics[1][2] = 0.5 * image_list[0].shape[0]  # update the principal point
                
                if random.random() < self.state_mask_prob:
                    current_state_mask = np.array([False, False])
                    current_state[:] = 0.0

            fov = calculate_fov( 2 * new_intrinsics[1][2], 2 * new_intrinsics[0][2], new_intrinsics)

            if self.use_rel:
                action_list = action_rel
            else:
                # use abs action for hand pose only
                rel_L = action_rel[:, :action_rel.shape[1]//2]
                rel_R = action_rel[:, action_rel.shape[1]//2:]
                abs_L = action_abs[:, :action_abs.shape[1]//2]
                abs_R = action_abs[:, action_abs.shape[1]//2:]

                action_list = np.concatenate([rel_L[:, :6], abs_L[:, 6:], rel_R[:, :6], abs_R[:, 6:]], axis=1)

            # ------------------------------------------------------------------
            # 8. Return to caller
            # ------------------------------------------------------------------

            result_dict = dict(
                instruction             = instruction,
                action_list             = action_list,          # (W,2*51) float32
                action_mask             = action_mask,          # (W,2)   bool
                current_state           = current_state,        # (2*61,)  float32
                current_state_mask      = current_state_mask,   # (2,) bool
                fov                     = fov,                  # (2,) float32
                intrinsics              = new_intrinsics,       # (3,3) float32
            )
            
            if image_list is not None:
                result_dict['image_list'] = image_list          # (W,H,W,3) uint8
            if image_mask is not None:
                result_dict['image_mask'] = image_mask          # (W,) bool
            
        return result_dict

    def set_global_data_statistics(self, global_data_statistics):
        self.global_data_statistics = global_data_statistics
        if not hasattr(self, 'gaussian_normalizer'):
            self.gaussian_normalizer = GaussianNormalizer(self.global_data_statistics)

    def transform_trajectory(
        self,
        sample_dict: dict = None,
        normalization: bool = True,
    ):
        """Pad action and state dimensions to match a unified size."""
        action_np = sample_dict["action_list"]
        state_np = sample_dict["current_state"]
       
        action_dim = action_np.shape[1]
        state_dim = state_np.shape[0]
        if normalization:
            # Normalize left and right hand actions and states separately
            action_np = self.gaussian_normalizer.normalize_action(action_np)
            state_np = self.gaussian_normalizer.normalize_state(state_np)

        # ===== Pad to unified dimensions =====
        unified_action_dim = ActionFeature.ALL_FEATURES[1]   # 192
        unified_state_dim = StateFeature.ALL_FEATURES[1]     # 212
        unified_state, unified_state_mask = pad_state_human(
            state_np,
            sample_dict["current_state_mask"],
            action_dim,
            state_dim,
            unified_state_dim
        )
        unified_action, unified_action_mask = pad_action(
            action_np,
            sample_dict["action_mask"],
            action_dim,
            unified_action_dim
        )

        sample_dict["action_list"] = unified_action
        sample_dict["action_mask"] = unified_action_mask
        sample_dict["current_state"] = unified_state
        sample_dict["current_state_mask"] = unified_state_mask
        return sample_dict

    def __getitem__(self, idx):

        # 1. Dataset iteration gives you idx (e.g., 12345)
        if self.training_idx is not None:
            data_id = self.training_idx[idx]
        else:
            data_id = idx

        # 2. Look up which episode and frame
        corr = self.index_frame_pair[data_id]  # → [episode_id, frame_id]

        # 3. Get the actual episode name
        # Example: corr[0]=51063 → "somethingsomethingv2_84911_ep_000000"
        episode_id = self.index_to_episode_id[corr[0]]

        # 4. Load that episode's annotation file
        # Example: Load frame 15 from that episode
        sample = self.get_item_frame(
            episode_id, int(corr[1]),
            action_past_window_size=self.action_past_window_size,
            action_future_window_size=self.action_future_window_size,
            image_past_window_size=self.image_past_window_size,
            image_future_window_size=self.image_future_window_size,
            rel_mode=self.rel_mode,  # 'step'
            load_images=self.load_images
        )

        """
        # Structure: [episode_id, frame_id_within_episode]
        [0, 5]   # Dataset index N points to episode 0, frame 5
        [0, 6]   # Dataset index N+1 points to episode 0, frame 6
        [0, 22]  # Dataset index N+17 points to episode 0, frame 22

        for x in self.index_frame_pair:
            if x[0] == 51063:
                print(x)
        [51063     0]
        [51063     1]
        [51063     2]
        [51063     3]
        [51063     4]
        [51063     5]
        [51063     6]
        ...
        [51063    34]
        [51063    35]
        [51063    36]
        [51063    37]
        [51063    38]
        [51063    39]
        [51063    40]
        [51063    41]
        [51063    42]
        [51063    43]
        [51063    44]        
        """

        return sample

def pad_state_human(
    state: torch.Tensor,
    state_mask: torch.Tensor,
    action_dim: int,
    state_dim: int,
    unified_state_dim: int
):
    """
    Expand state mask, mask invalid state dims, and pad current_state to a standard size.

    Args:
        current_state (Tensor): original state tensor, shape [state_dim]
        current_state_mask (Tensor): per-hand state mask, shape [state_dim//2] or [state_dim]
        action_dim (int): original action dimension
        state_dim (int): original state dimension
        unified_state_dim (int): target padded state dimension

    Returns:
        Tuple[Tensor, Tensor]: 
            padded current_state [unified_state_dim],
            padded current_state_mask [unified_state_dim]
    """

    current_state = torch.tensor(state, dtype=torch.float32)
    current_state_mask = torch.tensor(state_mask, dtype=torch.bool)
    
    # Expand state mask from per-hand to per-dim
    expanded_state_mask = current_state_mask.repeat_interleave(state_dim // 2)

    # Mask out invalid state dimensions
    current_state_masked = current_state * expanded_state_mask.to(current_state.dtype)

    # Initialize output tensors
    padded_state = torch.zeros(unified_state_dim, dtype=current_state.dtype)
    padded_mask = torch.zeros(unified_state_dim, dtype=torch.bool)

    # Fill first half of state_dim (left hand), skipping MANO betas
    padded_state[:action_dim//2] = current_state_masked[:action_dim//2].clone()
    padded_mask[:action_dim//2] = expanded_state_mask[:action_dim//2].clone()

    # Fill second half of state_dim (right hand), skipping MANO betas
    padded_state[action_dim//2:action_dim] = current_state_masked[state_dim//2:state_dim//2+action_dim//2].clone()
    padded_mask[action_dim//2:action_dim] = expanded_state_mask[state_dim//2:state_dim//2+action_dim//2].clone()

    return padded_state, padded_mask

def pad_action(
    actions: torch.Tensor = None,
    action_mask: torch.Tensor = None,
    action_dim: int = None,
    unified_action_dim: int = None
):
    """
    Expand action mask per dimension, mask invalid actions, and pad actions to a unified size.

    Args:
        actions (Tensor or None): original actions tensor, shape [T, action_dim] or None.
        action_mask (Tensor): per-hand action mask, shape [T, 2].
        action_dim (int): original action dimension.
        unified_action_dim (int): target padded actions dimension.

    Returns:
        Tuple[Optional[Tensor], Tensor]:
            padded actions [T, unified_action_dim] or None,
            padded action mask [T, unified_action_dim]
    """
    
    action_mask = torch.tensor(action_mask, dtype=torch.bool)
    
    # Expand mask from per-hand to per-dimension
    mask_left = action_mask[:, 0].unsqueeze(1).expand(-1, action_dim // 2)
    mask_right = action_mask[:, 1].unsqueeze(1).expand(-1, action_dim // 2)
    expanded_action_mask = torch.cat((mask_left, mask_right), dim=1)

    # ---------------------------
    # Case 1: actions is None
    # ---------------------------
    if actions is None:
        padding_mask = torch.zeros(
            (action_mask.shape[0], unified_action_dim - action_dim),
            dtype=torch.bool
        )
        action_mask_padded = torch.cat((expanded_action_mask, padding_mask), dim=1)
        return None, action_mask_padded

    # ---------------------------
    # Case 2: actions exists
    # ---------------------------

    actions = torch.tensor(actions, dtype=torch.float32)
    # Mask invalid action dims
    actions_masked = actions * expanded_action_mask.to(actions.dtype)

    # Pad both actions and mask
    padding = torch.zeros(
        (actions.shape[0], unified_action_dim - action_dim),
        dtype=actions.dtype
    )

    actions_padded = torch.cat((actions_masked, padding), dim=1)
    action_mask_padded = torch.cat((expanded_action_mask, padding.bool()), dim=1)

    return actions_padded, action_mask_padded