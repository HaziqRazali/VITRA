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
        denoising_mode=False,
        denoising_noise_std=0.05,
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
        # Denoising mode: action = clean pose at the SAME timestep as the state.
        # State = noisy version (Gaussian noise added to the pose dims).
        # Requires fwd_pred_next_n=1 and loss_type='smplx_denoising' in config.
        self.denoising_mode = denoising_mode
        self.denoising_noise_std = denoising_noise_std
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

        R_cam_extend = R_w2c[0:W+1] #@ np.eye(3)[None,...]  # (W+1, 3, 3)

        t_cam_extend = t_w2c[0:W+1]  # (W+1, 3)

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
        R_cam_next = R_cam_extend[1:]                   # (W, 3, 3)
        t_cam_next = t_cam_extend[1:]                   # (W, 3)
        smplx_euler_next = smplx_extend[1:]             # (W, 63)
        smplx_P_next = smplx_P_extend[1:]               # (W, 21, 3, 3)
        kept_next = kept_extend[1:]                     # (W,)

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
        pose_euler_extend  = R.from_matrix(hand_P_extend.reshape(-1,3,3)).as_euler('xyz', degrees=False).reshape(-1,45) # [W+1, 45]
        
        # ============================================================
        # 6. TRANSFORM KEYPOINTS TO MANO CANONICAL SPACE (BATCHED)
        # ============================================================
        # Convert 21 hand keypoints from world space → MANO canonical space (root-relative)
        # This representation is invariant to global hand position/rotation
        # Formula: joints_mano = R_mano^T @ (joints_world - t_mano)
        # - First: translate keypoints by negative wrist translation
        # - Then: apply inverse wrist rotation (R^T) to align with MANO coordinate frame
        joints_manospace_extend = (R_mano_extend.transpose(0, 2, 1) @ (joints_worldspace_extend.transpose(0, 2, 1) - t_mano_extend[..., None])).transpose(0,2,1)  # (W+1, 21, 3)

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
            pose_euler_extend = R.from_matrix(hand_P_extend.reshape(-1,3,3)).as_euler('xyz', degrees=False).reshape(-1,45)
            
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

            R_cur, t_cur = win['R_cam'],        win['t_cam']        # [16, 3, 3]        [16, 3]
            R_nxt, t_nxt = win['R_cam_next'],   win['t_cam_next']   # [16, 3, 3]        [16, 3]
            P_cur, P_nxt = win['smplx_P'],      win['smplx_P_next'] # [16, 21, 3, 3]    [16, 21, 3, 3]
            pose_next    = win['smplx_euler_next']                  # [16, 63]
            kept, kept_n = win['kept'],         win['kept_next']    # [16] [16]
            W = len(t_cur)                                          # 16

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
                t_rel = t_nxt - t_cur                               # [16, 3]
                R_rel = R_nxt @ R_cur.transpose(0,2,1)              # [16, 3, 3]
                P_rel = np.matmul(P_nxt, P_cur.transpose(0,1,3,2))  # [16, 21, 3, 3]
                valid = kept & kept_n                               # [16]

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

            pose_rel = R.from_matrix(P_rel.reshape(-1,3,3)).as_euler('xyz',False).reshape(W,63) # [16, 63]

            action_rel = np.concatenate(
                [t_rel,
                self._mat2euler(R_rel),
                pose_rel],
                axis=-1).astype(np.float32) # [16, 69]

            action_abs[~valid] = 0.0    # [16, 69]
            action_rel[~valid] = 0.0    # [16, 69]
            return action_abs, action_rel, valid

        else:

            R_cur, t_cur  = win['R_cam'],        win['t_cam']       # [16, 3, 3]        [16, 3]
            R_nxt, t_nxt  = win['R_cam_next'],   win['t_cam_next']  # [16, 3, 3]        [16, 3]
            P_cur, P_nxt  = win['hand_P'],       win['hand_P_next'] # [16, 15, 3, 3]    [16, 15, 3, 3]
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
                axis=-1).astype(np.float32)         # [16, 51]
            action_abs = action_abs.reshape(W, -1)  # [16, 51]

            # choose relative formulation
            if rel_mode == "step":
                t_rel = t_nxt - t_cur                               # [16, 3]
                R_rel = R_nxt @ R_cur.transpose(0,2,1)              # [16, 3, 3]
                P_rel = np.matmul(P_nxt, P_cur.transpose(0,1,3,2))  # [16, 15, 3, 3]
                valid = kept & kept_n                               # [16]

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

            pose_rel = R.from_matrix(P_rel.reshape(-1,3,3)).as_euler('xyz',False).reshape(W,45) # [16, 45]

            action_rel = np.concatenate(
                [t_rel,
                self._mat2euler(R_rel),
                pose_rel],
                axis=-1).astype(np.float32) # [16, 51]

            action_abs[~valid] = 0.0 # [16, 51]
            action_rel[~valid] = 0.0 # [16, 51]
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

        elif dataset_name == 'idea400':
            if self.clip_len is not None:
                video_path = os.path.join(self.video_root, video_name+ '_part' + str(part_index+1) + '.mp4')
            else:
                video_path = os.path.join(self.video_root, video_name+'.mp4')
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
        
        if epi['video_decode_frame'] is None:

            T = epi["smplx_params"]["root_pose"].shape[0] # T
            frame_in_video = frame_id # scalar

            # ---------- build padded window indices ---------------------------
            # Not support multiple images now
            idx_win, oob = self._window_indices(frame_id, past, future, 0, T-1)     # (W,)
            
            if self.clip_len is not None:
                print(f"Function not implemented?")
                sys.exit()
            else:
                video_path = self._resolve_video_path(epi['dataset_name'], epi['video_name'])
                decode_ids = idx_win # 

            # ---------- read images --------------------
            # Retry mechanism: try up to 3 times to load video frames
            imgs = None
            for attempt in range(3):
                try:
                    imgs, _ = load_video_decord(video_path, frame_index=decode_ids, rotation=False)
                    break  # Success, exit the retry loop
                except Exception as e:
                    # if attempt == 2:
                    #     raise  # Raise the exception after 3 failed attempts
                    print(f"Warning: failed to load video frames from {video_path} (attempt {attempt+1}/3): {e}")
                    time.sleep(0.1)

            if imgs is None:
                raise RuntimeError(f"Failed to load video after 3 attempts: {video_path}")

            images = np.stack(imgs, axis=0)           # (L,H,W,3) uint8
            mask   = ~oob                             # (L,) bool

            return images, mask

        # original datasets
        else:

            # somethingsomethingv2_84911_ep_000000 -> somethingsomethingv2
            dataset_name = episode_id.split('_')[0]
            # video_path   = self._resolve_video_path(dataset_name, epi['video_name'])
            decode_table = epi['video_decode_frame']                    # (T,)
            T            = len(decode_table)                            # T
            frame_in_video = epi['video_decode_frame'][frame_id]        # frame_id=1, frame_in_video=10  

            # ---------- build padded window indices ---------------------------
            # Not support multiple images now
            idx_win, oob = self._window_indices(frame_id, past, future, 0, T-1)     # (W=1,) False

            if self.clip_len is not None:
                part_idx = frame_in_video // self.clip_len # clip_len = 2000
                frame_in_part = frame_in_video % self.clip_len
                video_path = self._resolve_video_path(dataset_name, epi['video_name'], part_idx)
                decode_ids = [frame_in_part]
            else:
                video_path = self._resolve_video_path(dataset_name, epi['video_name'])
                decode_ids = decode_table[idx_win]  # [10]

            # ---------- read images --------------------
            # Retry mechanism: try up to 3 times to load video frames
            imgs = None
            for attempt in range(3):
                try:
                    imgs, _ = load_video_decord(video_path, frame_index=decode_ids, rotation=False)
                    break  # Success, exit the retry loop
                except Exception as e:
                    # if attempt == 2:
                    #     raise  # Raise the exception after 3 failed attempts
                    print(f"Warning: failed to load video frames from {video_path} (attempt {attempt+1}/3): {e}")
                    time.sleep(0.1)

            if imgs is None:
                raise RuntimeError(f"Failed to load video after 3 attempts: {video_path}")

            images = np.stack(imgs, axis=0)           # (1, 240, 360, 3) (L,H,W,3) uint8
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
            text_body   = epi["text"]["body"]
            if frame_id >= T:
                print(episode_id, len(text_body), T, frame_id)
                sys.exit()
            text_frame_id = min(frame_id, len(text_body) - 1)
            instruction = text_body[text_frame_id]

            # only the body to consider so we can reuse
            idx_body    = idx_win
            oob_body    = oob
            start_body  = 0
            end_body    = T - 1

            """
            # ============================================================
            # 3. PREPARE SMPLX BODY POSE TEMPORAL WINDOW
            # ============================================================
            # Transforms SMPLX body pose parameters across a temporal window from world space
            # to camera space and creates paired current/next-frame representations for training.
            #
            # SMPLX is a parametric body model (extension of SMPL) that represents full body pose
            # using 21 body joints (63 parameters: 21 joints × 3 Euler angles in xyz format).
            # Unlike MANO which models hands in detail, SMPLX captures the full body skeleton.
            #
            # INPUTS:
            # -------
            # - epi['smplx_params']: Dictionary containing SMPLX body reconstruction data
            #       * 'body_pose': (T, 63) - Full body joint angles in Euler format (21 joints × 3)
            #       * Additional SMPLX parameters (shape betas, root orientation, etc.)
            #   where T = total frames in the episode
            #
            # - R_w2c: (T, 3, 3) - World-to-camera rotation matrices for each frame
            #   Transforms from world coordinate system to camera view at each timestep
            #
            # - t_w2c: (T, 3) - World-to-camera translation vectors
            #   Camera position in world coordinates at each timestep
            #
            # - idx_body: (W,) - Frame indices for the temporal window (typically W=16)
            #   Specifies which frames from the episode to include in this window
            #   Example: [5, 6, 7, ..., 20] for a 16-frame window starting at frame 5
            #
            # - frame_id: Scalar - The anchor frame index (current observation timestep)
            #   All poses will be transformed relative to this frame's camera view
            #
            # - oob_body: (W,) bool - Out-of-bounds mask marking invalid frames
            #   True where frame indices fall outside the valid episode range or 
            #   annotated text segment. Invalid frames will be reset to identity pose.
            #
            # - start_body/end_body: Valid frame range boundaries for the body data
            #   Typically (0, T-1) when processing full episode without text constraints
            #
            # - upsample_factor: Temporal upsampling ratio (≥1.0)
            #   If >1, interpolates between frames using PCHIP for smoother trajectories
            #
            # PROCESSING STEPS:
            # ----------------
            # 1. Extends window by +1 frame: W → W+1 to enable (current, next) pairing
            #    Example: [f₀, f₁, ..., f₁₅] → [f₀, f₁, ..., f₁₅, f₁₆]
            #
            # 2. Extracts SMPLX body_pose (63-dim Euler angles) for W+1 frames
            #
            # 3. Handles out-of-bounds frames: Sets invalid poses to identity (rest pose)
            #    to prevent corrupted data from extrapolation beyond episode boundaries
            #
            # 4. Converts Euler angles → Rotation matrices for geometric transformations
            #    (63,) → (21, 3, 3) per frame for 21 body joints
            #
            # 5. Transforms from world space → camera space using anchor frame's extrinsics
            #    This makes all body poses relative to the current camera viewpoint,
            #    which is critical for egocentric action prediction
            #
            # 6. Optional temporal upsampling via PCHIP interpolation if upsample_factor > 1
            #    Creates smoother motion trajectories for better action prediction
            #
            # 7. Splits W+1 frames into paired (current, next) sequences of length W:
            #    - Current: [f₀, f₁, ..., f₁₅] (frames at time t)
            #    - Next:    [f₁, f₂, ..., f₁₆] (frames at time t+1)
            #    Enables supervised learning of Δpose₍ₜ→ₜ₊₁₎
            #
            # OUTPUT DICTIONARY (win_body):
            # ----------------------------
            # All arrays have shape (W, ...) after current/next splitting:
            #
            # Current frame representations:
            # - 'R_cam': (W, 3, 3) - Camera-space body root orientations
            # - 't_cam': (W, 3) - Camera-space body root translations  
            # - 'smplx_euler': (W, 63) - Body joint angles in Euler format (21 joints × 3)
            # - 'smplx_P': (W, 21, 3, 3) - Body joint rotation matrices
            # - 'kept': (W,) bool - Validity mask for current frames
            #
            # Next frame representations (for learning temporal dynamics):
            # - 'R_cam_next': (W, 3, 3) - Body root orientation at t+1
            # - 't_cam_next': (W, 3) - Body root translation at t+1
            # - 'smplx_euler_next': (W, 63) - Body joint angles at t+1
            # - 'smplx_P_next': (W, 21, 3, 3) - Body joint rotations at t+1
            # - 'kept_next': (W,) bool - Validity mask for next frames
            #
            # USAGE EXAMPLE:
            # -------------
            # Given frame_id=10, action_past_window_size=0, action_future_window_size=15:
            #   idx_body = [10, 11, 12, ..., 25] (W=16 frames)
            #   
            # win_body contains:
            #   - smplx_euler[0] = body pose at frame 10 (current state)
            #   - smplx_euler_next[0] = body pose at frame 11 (target for prediction)
            #   - ...
            #   - smplx_euler[15] = body pose at frame 25
            #   - smplx_euler_next[15] = body pose at frame 26
            #
            # This paired structure allows the model to learn:
            #   action₍ₜ₎ = f(state₍ₜ₎, state₍ₜ₊₁₎)
            # ============================================================
            """
            win_body  = self._prepare_smplx_window(
                epi['smplx_params'],  R_w2c, t_w2c, idx_body, frame_id, anchor_frame=True, 
                oob=oob_body, start=start_body, end=end_body, upsample_factor=self.upsample_factor
            )
            idx_center = action_past_window_size          # local index of t0 in window

            """
            # ============================================================
            # 4. COMPUTE ABSOLUTE & RELATIVE BODY POSE ACTIONS
            # ============================================================
            # Converts paired (current, next) SMPLX body pose representations into
            # action sequences for supervised learning. Computes both absolute and
            # relative pose deltas across the temporal window.
            #
            # PURPOSE:
            # --------
            # Transforms the raw SMPLX body pose window into trainable action representations
            # that the model will learn to predict. Actions encode the change in body pose
            # from one timestep to the next, enabling the model to learn motion dynamics.
            #
            # INPUTS:
            # -------
            # - win_body: Dictionary from _prepare_smplx_window containing:
            #     * R_cam, t_cam: (W, 3, 3) and (W, 3) - Current frame body root poses
            #     * R_cam_next, t_cam_next: (W, 3, 3) and (W, 3) - Next frame body root poses
            #     * smplx_euler: (W, 63) - Current frame body joint angles (21 joints × 3)
            #     * smplx_euler_next: (W, 63) - Next frame body joint angles
            #     * smplx_P: (W, 21, 3, 3) - Current frame joint rotation matrices
            #     * smplx_P_next: (W, 21, 3, 3) - Next frame joint rotation matrices
            #     * kept, kept_next: (W,) bool - Validity masks for current/next frames
            #   where W = temporal window length (typically 16)
            #
            # - anchor_idx: Scalar index of the anchor frame within the window
            #   Typically = action_past_window_size (e.g., 0 if no past context)
            #   This is the "current observation" frame t₀
            #
            # - rel_mode: String specifying the reference frame for relative actions
            #   * "step": Compute frame-to-frame deltas Δ(tᵢ → tᵢ₊₁)
            #     Encodes how pose changes from each frame to its immediate successor
            #     Example: If at frame 10, compute change from frame 10→11
            #   
            #   * "anchor": Compute deltas relative to anchor frame Δ(t₀ → tᵢ₊₁)
            #     All poses are expressed relative to the current observation
            #     Example: If anchor is frame 10, compute 10→11, 10→12, ..., 10→25
            #
            # - action_type: String specifying pose representation format
            #   * "angle": Use Euler angles (xyz convention) for body joints
            #     Output: 63 dimensions (21 joints × 3 Euler angles)
            #   
            #   * "keypoints": Use 3D joint positions in MANO canonical space
            #     Output: 63 dimensions (21 joints × 3 coordinates)
            #     (Note: For SMPLX, keypoints are not extracted in current implementation)
            #
            # PROCESSING PIPELINE:
            # -------------------
            # 1. Extract current and next-frame body pose components:
            #    - Root translation: t_cur, t_nxt (camera space)
            #    - Root rotation: R_cur, R_nxt (3×3 matrices)
            #    - Joint rotations: P_cur, P_nxt (21×3×3 matrices)
            #    - Joint angles: smplx_euler_next (63-dim Euler)
            #    - Validity: kept, kept_next (boolean masks)
            #
            # 2. Compute ABSOLUTE actions (target poses at t+1):
            #    action_abs = [t_next, euler(R_next), body_pose_next]
            #    Shape: (W, 69) = (W, 3 + 3 + 63)
            #    - Translation at t+1: (3,) in camera space
            #    - Root rotation at t+1: (3,) Euler angles
            #    - Body joint angles at t+1: (63,) Euler angles or keypoints
            #
            # 3. Compute RELATIVE actions (pose deltas):
            #    
            #    If rel_mode == "step" (frame-to-frame):
            #      Δt = t_next - t_cur                    # Translation delta
            #      ΔR = R_next @ R_cur^T                  # Rotation delta (composition)
            #      ΔP = P_next @ P_cur^T                  # Joint rotation deltas (21 joints)
            #      valid = kept & kept_next               # Both frames must be valid
            #    
            #    If rel_mode == "anchor" (anchor-relative):
            #      Δt = t_next - t_anchor                 # Delta from anchor to next
            #      ΔR = R_next @ R_anchor^T               # Rotation from anchor to next
            #      ΔP = P_next @ P_anchor^T               # Joint deltas from anchor
            #      valid = kept_next & kept[anchor_idx]   # Anchor and next must be valid
            #    
            #    Convert rotation deltas to Euler angles:
            #      ΔP_euler = euler(ΔP)                   # (W, 63) Euler angle deltas
            #    
            #    Combine into action vector:
            #      action_rel = [Δt, euler(ΔR), ΔP_euler]
            #      Shape: (W, 69) = (W, 3 + 3 + 63)
            #
            # 4. Apply validity masking:
            #    - Set action_abs[~valid] = 0.0  (zero out invalid absolute actions)
            #    - Set action_rel[~valid] = 0.0  (zero out invalid relative actions)
            #    This prevents the model from learning from corrupted/extrapolated data
            #
            # OUTPUT TENSORS:
            # --------------
            # abs_body: (W, 69) float32 - Absolute body pose actions
            #   Format: [t_next(3), euler(R_next)(3), body_joints_next(63)]
            #   Represents the target body pose at each timestep t+1
            #   Used when training with absolute pose prediction
            #
            # rel_body: (W, 69) float32 - Relative body pose actions  
            #   Format: [Δt(3), Δeuler(R)(3), Δbody_joints(63)]
            #   Represents the change in body pose from reference to t+1
            #   Used when training with relative pose delta prediction (default)
            #
            # msk_body: (W,) bool - Action validity mask
            #   True where both current and next frames have valid SMPLX reconstruction
            #   False for frames outside episode boundaries or failed reconstructions
            #   Used to mask loss computation during training
            #
            # COORDINATE FRAME:
            # ----------------
            # All actions are in CAMERA SPACE (not world space):
            # - Translations are in meters relative to camera origin
            # - Rotations are relative to camera coordinate frame
            # This camera-centric representation is critical for egocentric action
            # prediction, as it makes poses invariant to global camera motion.
            #
            # USAGE EXAMPLE:
            # -------------
            # Given:
            #   - frame_id = 10, action_past_window_size = 0, action_future_window_size = 15
            #   - idx_body = [10, 11, 12, ..., 25] (W=16)
            #   - idx_center = 0 (anchor is first frame in window)
            #   - rel_mode = "step"
            #
            # Processing:
            #   win_body contains paired poses:
            #     smplx_euler[0] = pose at frame 10, smplx_euler_next[0] = pose at frame 11
            #     smplx_euler[1] = pose at frame 11, smplx_euler_next[1] = pose at frame 12
            #     ...
            #     smplx_euler[15] = pose at frame 25, smplx_euler_next[15] = pose at frame 26
            #
            # Output:
            #   abs_body[0] = absolute pose at frame 11
            #   rel_body[0] = pose delta from frame 10→11
            #   abs_body[1] = absolute pose at frame 12
            #   rel_body[1] = pose delta from frame 11→12
            #   ...
            #   abs_body[15] = absolute pose at frame 26
            #   rel_body[15] = pose delta from frame 25→26
            #
            # The model learns to predict:
            #   Given: Current observation at frame t
            #   Predict: rel_body[i] = Δpose to reach frame t+i+1
            # ============================================================
            """
            abs_body, rel_body, msk_body = self._make_action_window_vec(
                win_body,  anchor_idx=idx_center, rel_mode=rel_mode, action_type=self.action_type
            ) # [16, 69] [16, 69] [16,]

            action_abs  = abs_body                                          # (W, 69) single body
            action_rel  = rel_body                                          # (W, 69) single body
            # CRITICAL: Reshape mask to (W, 1) for compatibility with pad_action function
            # pad_action expects shape (W, num_entities) where num_entities=2 for dual-hand, 1 for single-body
            action_mask = msk_body[:, np.newaxis]                           # (W, 1) single body mask

            """
            # ============================================================
            # 5. CONSTRUCT CURRENT STATE (SMPLX SINGLE-BODY)
            # ============================================================
            # Builds the current observation state at anchor frame (frame_id).
            # This represents "what the robot/model sees NOW" before predicting future actions.
            #
            # CRITICAL DISTINCTION: STATE vs ACTION (SMPLX)
            # =============================================
            #
            # **current_state**:
            #   - TEMPORAL: Single timestep (snapshot at time t₀)
            #   - CONTENT: Only pose parameters (NO betas for SMPLX in this implementation)
            #   - FORMAT: Always ABSOLUTE pose in camera space
            #   - PURPOSE: Current observation input to the model
            #   - SHAPE: (69,) = t_cam(3) + euler(R_cam)(3) + body_pose(63)
            #     * t_cam: (3,) body root translation in camera space
            #     * euler(R_cam): (3,) body root rotation as Euler angles
            #     * body_pose: (63,) 21 body joints × 3 Euler angles
            #
            # **action_list**:
            #   - TEMPORAL: Sequence of W timesteps (future trajectory from t₁ to t_W)
            #   - CONTENT: Only pose parameters (matching state for SMPLX)
            #   - FORMAT: Can be RELATIVE (deltas, default) or ABSOLUTE (target poses)
            #   - PURPOSE: Target predictions for the model to learn
            #   - SHAPE: (W, 69) = W timesteps × 69 dims per timestep
            #     * Per timestep: t_cam(3) + euler(R_cam)(3) + body_pose(63) = 69
            #
            # KEY DIFFERENCES FROM MANO:
            # --------------------------
            # 1. **No shape parameters (betas)**:
            #    - MANO state includes 10 betas per hand (hand size/shape)
            #    - SMPLX omits betas in this implementation (could be added if needed)
            #    - Result: SMPLX state and action have SAME dimensionality (69)
            #
            # 2. **Single entity vs dual entities**:
            #    - SMPLX: 1 body → state (69,), action (W, 69), mask (1,) and (W, 1)
            #    - MANO: 2 hands → state (122,), action (W, 102), mask (2,) and (W, 2)
            #
            # 3. **Temporal dimension still differs**:
            #    - STATE is 1D: (69,) - Single snapshot "I see THIS body pose now"
            #    - ACTIONS are 2D: (W, 69) - Trajectory "Move body like THIS over next W frames"
            #
            # 4. **Absolute vs Relative still applies**:
            #    - STATE is always ABSOLUTE: "Body is at position [x,y,z] with joints at [θ₁, θ₂, ...]"
            #    - ACTIONS can be RELATIVE: "Move body by Δx, Δy, Δz with joint deltas [Δθ₁, Δθ₂, ...]"
            #
            # CONCRETE EXAMPLE:
            # ----------------
            # frame_id = 50, action_future_window_size = 15 (W=16)
            #
            # current_state (69 dims):
            #   [0.0, 0.5, 1.2, 0.0, 0.0, 0.0, <63 body joint angles>]
            #   → Represents: "At frame 50, body root is at (0.0, 0.5, 1.2) with upright orientation"
            #
            # action_list (16, 69) - if use_rel=True:
            #   Row 0:  [Δpose from frame 50→51] = [0.01, 0.02, 0.0, 0.0, 0.0, 0.01, ...]
            #   Row 1:  [Δpose from frame 51→52] = [0.01, 0.02, 0.0, 0.0, 0.0, 0.01, ...]
            #   ...
            #   Row 15: [Δpose from frame 65→66] = [0.01, 0.01, -0.01, 0.0, 0.0, 0.0, ...]
            #   → Represents: "To reach frame 51, move body by [+0.01, +0.02, 0.0] with joint deltas [...]"
            #
            # MODEL TRAINING:
            # --------------
            # Input:  current_state (69,) + image_list + instruction
            # Output: Predict action_list (W, 69)
            # Loss:   Compare predicted actions vs ground truth actions
            # 
            # At inference:
            #   1. Observe current body state at frame t
            #   2. Model predicts future body motions (actions)
            #   3. Execute actions to control robot body
            #   4. Update state based on executed actions
            #   5. Repeat for next timestep
            # ============================================================
            """
            cur = self._pack_state(win_body['R_cam'],
                        win_body['t_cam'],
                        win_body['smplx_euler'] if self.action_type=='angle' else win_body['smplx_curr'].reshape(W, -1),
                        idx_center) # [3 + 3 + 63] = 69 dims

            betas = epi["smplx_params"]["shape"][frame_id]

            # In denoising mode, inject Gaussian noise into the 69-dim pose (not betas)
            # to create the noisy state input. Save the clean pose for the action target.
            if self.denoising_mode:
                cur_clean = cur.copy()
                cur = cur + np.random.randn(*cur.shape).astype(np.float32) * self.denoising_noise_std

            # Strip body_6d (root trans+rot, dims 0:6) in denoising mode — joints only.
            if self.denoising_mode:
                cur       = cur[6:]        # (63,) joints only
                cur_clean = cur_clean[6:]  # (63,) joints only

            current_state = cur  # (63,) joints only in denoising mode, (69,) otherwise
            current_state_mask  = np.array([win_body['kept'][idx_center]])  # (1,) single body mask
            
            # NOTE: The padding functions (pad_state_human, pad_action) now handle both
            # dual-hand MANO (num_entities=2) and single-body SMPLX (num_entities=1) cases.
            #   - For SMPLX: state_mask is (1,), action_mask is (W, 1)
            #   - For MANO: state_mask is (2,), action_mask is (W, 2)
            # The functions automatically detect num_entities and apply appropriate logic.

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

            #dataset_name = episode_id.split('_')[0]
            intrinsics = epi['intrinsics']
            new_intrinsics = compute_new_intrinsics_resize(intrinsics, (H, W))

            fov = calculate_fov( 2 * new_intrinsics[1][2], 2 * new_intrinsics[0][2], new_intrinsics)

            """
            # ============================================================
            # 7. SELECT ACTION REPRESENTATION MODE
            # ============================================================
            # Determines whether to use fully relative or hybrid action representation.
            # This is controlled by the `use_rel` flag from the config file.
            #
            # CONTEXT:
            # --------
            # At this point, we have two action representations computed:
            #   - action_rel: (W, 69) - Fully relative actions [Δt, Δeuler(R), Δbody_joints]
            #   - action_abs: (W, 69) - Fully absolute actions [t_next, euler(R_next), body_joints_next]
            #
            # Action dimensions (69 total for SMPLX body):
            #   [0:3]   - Translation (t) in camera space
            #   [3:6]   - Root rotation as Euler angles (R)
            #   [6:69]  - Body joint angles (63 dims = 21 joints × 3 Euler angles)
            #
            # TWO MODES:
            # ----------
            # Mode 1: use_rel=True (FULLY RELATIVE)
            #   Use pure relative actions throughout
            #   action_list = action_rel = [Δt, Δeuler(R), Δbody_joints]
            #   
            #   Advantages:
            #     - Easier for model to learn small deltas
            #     - More stable training gradients
            #     - Better handles long-horizon predictions
            #
            # Mode 2: use_rel=False (HYBRID: Relative root + Absolute joints)
            #   Use relative for root pose, absolute for joint angles
            #   action_list = [Δt, Δeuler(R), body_joints_abs]
            #   
            #   Reasoning:
            #     - Root translation/rotation benefit from relative deltas (small changes)
            #     - Joint angles may be easier to predict in absolute form (target pose)
            #     - Hybrid approach can improve prediction accuracy for articulated poses
            #
            # WHY THE `//2` OPERATION?
            # ------------------------
            # The code `rel = action_rel[:, :action_rel.shape[1]//2]` appears here but
            # is somewhat misleading for single-body SMPLX:
            #
            # 1. For SMPLX (single body): shape[1] = 69, so //2 = 34
            #    - Takes first 34 dims: [t(3), euler(R)(3), partial_body_joints(28)]
            #    - This splitting doesn't align with semantic boundaries
            #    - The //2 is essentially UNUSED in the final concatenation
            #
            # 2. The ACTUAL logic being applied (look at the concatenation):
            #    action_list = np.concatenate([rel[:, :6], abs[:, 6:]], axis=1)
            #    
            #    Breaking this down:
            #    - rel[:, :6]:  First 6 dims from relative = [Δt(3), Δeuler(R)(3)]
            #    - abs[:, 6:]:  Dims 6+ from absolute = [body_joints_abs(63)]
            #    
            #    Result: [Δt(3), Δeuler(R)(3), body_joints_abs(63)] = 69 dims
            #    (Relative root pose + Absolute joint angles)
            #
            # 3. WHY is `//2` there then? (ANSWER: Copy-pasted from MANO code)
            #    This code mirrors the dual-hand (MANO) case where `//2` is ESSENTIAL:
            #    - action_rel/abs have shape (W, 102) = left_hand(51) + right_hand(51)
            #    - The //2 FIRST splits into left vs right: [:, :51] and [:, 51:]
            #    - THEN each hand independently gets hybrid treatment: rel[:6] + abs[6:]
            #    - See the MANO code path (else block) for the correct usage
            #    
            #    For SMPLX single-body, the //2 is semantically unnecessary and
            #    could be removed (just use `action_rel` and `action_abs` directly).
            #    However, it's kept for code consistency with MANO and doesn't affect
            #    the output since [:, :6] and [:, 6:] operate on the correct ranges.
            #
            # TODO: Consider simplifying to:
            #       action_list = np.concatenate([action_rel[:, :6], action_abs[:, 6:]], axis=1)
            #
            # FINAL OUTPUT:
            # ------------
            # action_list: (W, 69) float32
            #   When use_rel=True:  [Δt, Δeuler(R), Δbody_joints] (fully relative)
            #   When use_rel=False: [Δt, Δeuler(R), body_joints_abs] (hybrid)
            #
            # This action_list will be normalized and padded later before 
            # being fed to the model for training.
            # ============================================================
            """
            if self.denoising_mode:
                # Denoising target: clean body joints at the SAME timestep (body_6d stripped).
                # action_list shape: (1, 63) = [joints(63)]  — root trans/rot excluded
                # action_mask  shape: (1, 1)  - single timestep, single body
                # NOTE: fwd_pred_next_n must be 1 and loss_type='smplx_denoising'.
                action_list = cur_clean[np.newaxis, :].astype(np.float32)  # (1, 63) joints only (body_6d stripped)
                action_mask = np.array([[win_body['kept'][idx_center]]], dtype=bool)                 # (1, 1)
            elif self.use_rel:
                action_list = action_rel
            else:
                # use abs action for body joint angles only (keep root relative)
                rel = action_rel 
                abs = action_abs

                # Hybrid: Relative root (translation+rotation) + Absolute body joints
                action_list = np.concatenate([rel[:, :6], abs[:, 6:]], axis=1)  # (W, 69) = 6 + 63

            """
            # ============================================================
            # 8. PACKAGE RETURN DICTIONARY (SMPLX SINGLE-BODY)
            # ============================================================
            # Returns a complete training sample containing:
            #   - Current observation (state + images)
            #   - Future trajectory to predict (actions)
            #   - Task description (instruction)
            #   - Camera parameters (intrinsics, fov)
            #
            # SUMMARY: STATE vs ACTION DIMENSIONS (SMPLX)
            # ===========================================
            #
            # **current_state**: (69,) float32 - SINGLE TIMESTEP
            #   Format: [body_root_translation(3), body_root_rotation(3), body_joints(63)]
            #   Component breakdown:
            #     - t_cam: (3,) body root translation in camera space
            #     - euler(R_cam): (3,) body root rotation as Euler angles
            #     - body_pose: (63,) body joint angles (21 joints × 3 Euler angles)
            #   Total: 3 + 3 + 63 = 69 dims
            #   Note: No betas (shape parameters) included in this SMPLX implementation
            #
            # **action_list**: (W, 69) float32 - W TIMESTEPS
            #   Format: Same 69-dim structure as state, repeated for W timesteps
            #   Component breakdown per timestep:
            #     - t_cam: (3,) body translation (or Δt if relative)
            #     - euler(R_cam): (3,) body rotation (or Δeuler if relative)
            #     - body_pose: (63,) body joint angles (or Δpose if relative)
            #   Total: 3 + 3 + 63 = 69 dims per timestep
            #   Across W timesteps: (W, 69)
            #
            # **current_state_mask**: (1,) bool - Body validity at anchor frame
            #   [body_valid]
            #   True if body has valid SMPLX reconstruction at frame_id
            #
            # **action_mask**: (W, 1) bool - Body validity across trajectory
            #   [:, 0] = body validity for each of W timesteps
            #   True if both current AND next frame have valid body data
            #
            # **instruction**: str - Natural language task description
            #   Example: "Person is walking forward while waving."
            #   Describes the full-body motion being performed
            #
            # **fov**: (2,) float32 - Field of view [fov_x, fov_y] in radians
            #   Derived from camera intrinsics, used for perspective-aware prediction
            #
            # **intrinsics**: (3, 3) float32 - Camera intrinsic matrix K
            #   [[fx,  0, cx],
            #    [ 0, fy, cy],
            #    [ 0,  0,  1]]
            #   After augmentation/resizing, used for 3D-2D projection
            #
            # COMPARISON WITH MANO:
            # --------------------
            # SMPLX: state(69,)  action(W,69)  - Same dims, single body
            # MANO:  state(122,) action(W,102) - Different dims (betas included), dual hands
            #
            # DOWNSTREAM PROCESSING:
            # ---------------------
            # This dict is passed to transform_trajectory() which:
            #   1. Normalizes actions and states using dataset statistics
            #   2. Pads to unified dimensions (action: 192, state: 212)
            #   3. Converts to PyTorch tensors for model training
            # ============================================================
            """

            result_dict = dict(
                instruction             = instruction,           # str - task description (e.g., from video captions)
                action_list             = action_list,           # (W, 69) float32 - future body trajectory
                action_mask             = action_mask,           # (W, 1) bool - body validity per timestep
                current_state           = current_state,         # (69,) float32 - current observation
                current_state_mask      = current_state_mask,    # (1,) bool - body validity at anchor
                fov                     = fov,                   # (2,) float32 - [fov_x, fov_y]
                intrinsics              = new_intrinsics,        # (3, 3) float32 - camera intrinsics K
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
            
            """
            # ============================================================
            # 4. PREPARE MANO HAND POSE TEMPORAL WINDOWS (DUAL-HAND)
            # ============================================================
            # Transforms MANO hand parameters for BOTH hands across temporal windows from world space
            # to camera space and creates paired current/next-frame representations for training.
            #
            # MANO is a parametric hand model that represents hand pose using:
            # - Wrist pose: 3D translation + 3×3 rotation matrix (6 DOF)
            # - Finger pose: 15 finger joints × 3×3 rotation matrices (45 DOF)
            # - Hand shape: 10 PCA coefficients (beta parameters)
            # Total: 61 parameters per hand (excluding shape which is per-episode constant)
            #
            # Unlike SMPLX which models the full body, MANO provides detailed articulated models
            # for each hand separately, enabling fine-grained manipulation prediction.
            #
            # KEY DIFFERENCE FROM SMPLX:
            # -------------------------
            # - SMPLX: Single body model with text-independent temporal window
            #   * idx_body is derived directly from frame_id
            #   * start_body, end_body span the full episode [0, T-1]
            #   * No hand-specific text annotations to consider
            #
            # - MANO: Dual hands with text-DEPENDENT temporal windows
            #   * idx_win_left and idx_win_right can DIFFER between hands
            #   * Each hand has its own text annotation with time boundaries
            #   * start_left, end_left constrain left hand's valid range
            #   * start_right, end_right constrain right hand's valid range
            #   * Example: "Left hand: Pick up cup [frames 5-30]. Right hand: Hold plate [frames 15-40]."
            #     → Left window might be [5,6,...,20], Right window [15,16,...,30]
            #
            # INPUTS (PER HAND):
            # ------------------
            # - epi['left'] / epi['right']: Dictionary containing MANO hand reconstruction data
            #     * 'global_orient_worldspace': (T, 3, 3) - Wrist rotation matrices in world coords
            #     * 'transl_worldspace': (T, 3) - Wrist translations in world coords
            #     * 'hand_pose': (T, 15, 3, 3) - Finger joint rotations (15 joints × 3×3 rotation)
            #     * 'joints_worldspace': (T, 21, 3) - Hand keypoint positions in world coords
            #     * 'kept_frames': (T,) - Binary mask for valid hand reconstructions
            #     * 'beta': (10,) - MANO shape parameters (PCA coefficients)
            #   where T = total frames in the episode
            #
            # - R_w2c: (T, 3, 3) - World-to-camera rotation matrices for each frame
            #   Transforms from world coordinate system to camera view at each timestep
            #
            # - t_w2c: (T, 3) - World-to-camera translation vectors
            #   Camera position in world coordinates at each timestep
            #
            # - idx_win_left / idx_win_right: (W_L,) and (W_R,) - Frame indices for each hand's window
            #   Specifies which frames to include. These can have DIFFERENT lengths and ranges:
            #   * W_L (left window length) may differ from W_R (right window length)
            #   * idx_win_left might be [5,6,...,20] while idx_win_right is [15,16,...,30]
            #   This asymmetry arises when hands have non-overlapping text annotations
            #
            # - frame_id: Scalar - The anchor frame index (current observation timestep)
            #   All poses will be transformed relative to this frame's camera view
            #   Note: frame_id is typically within BOTH left and right windows, but edge cases exist
            #
            # - oob_left / oob_right: (W_L,) and (W_R,) bool - Out-of-bounds masks per hand
            #   True where frame indices fall outside the valid episode range OR outside
            #   the hand's text-annotated segment. Invalid frames will be reset to identity pose.
            #   Example: If left hand text covers frames [10-30] but window extends to [5-35],
            #            then oob_left will be True for frames [5-9] and [31-35]
            #
            # - start_left, end_left / start_right, end_right: Valid frame range boundaries per hand
            #   Derived from text annotations: frames where the hand is actively performing its task
            #   Unlike SMPLX (always 0, T-1), these are TEXT-CONSTRAINED subsets of the episode
            #
            # - upsample_factor: Temporal upsampling ratio (≥1.0)
            #   If >1, interpolates between frames using PCHIP for smoother trajectories
            #
            # PROCESSING STEPS (PER HAND):
            # ---------------------------
            # 1. Extends window by +1 frame: W → W+1 to enable (current, next) pairing
            #    Example: [f₀, f₁, ..., f₁₅] → [f₀, f₁, ..., f₁₅, f₁₆]
            #
            # 2. Extracts MANO parameters for W+1 frames:
            #    - Wrist pose: R_mano (3×3), t_mano (3,)
            #    - Finger joints: hand_P (15×3×3)
            #    - Keypoints: joints_worldspace (21×3)
            #    - Validity: kept_frames (bool)
            #
            # 3. Handles out-of-bounds frames: Sets invalid poses to identity (rest pose)
            #    This is CRITICAL for text-based datasets where windows can extend beyond
            #    annotated segments. Prevents corrupted data from extrapolation.
            #
            # 4. Converts finger rotation matrices → Euler angles for neural network processing
            #    (15×3×3) → (45,) per frame, using xyz Euler convention
            #
            # 5. Transforms keypoints from world space → MANO canonical space (root-relative)
            #    Formula: joints_mano = R_mano^T @ (joints_world - t_mano)
            #    This representation is invariant to global hand position/rotation
            #
            # 6. Transforms wrist pose from world space → camera space using anchor frame's extrinsics
            #    - R_cam = R_world2cam @ R_mano_world
            #    - t_cam = R_world2cam @ t_mano_world + t_world2cam
            #    Makes hand poses egocentric (relative to current camera viewpoint)
            #
            # 7. Optional temporal upsampling via PCHIP interpolation if upsample_factor > 1
            #    Creates smoother motion trajectories for better action prediction
            #
            # 8. Splits W+1 frames into paired (current, next) sequences of length W:
            #    - Current: [f₀, f₁, ..., f_{W-1}] (frames at time t)
            #    - Next:    [f₁, f₂, ..., f_W] (frames at time t+1)
            #    Enables supervised learning of Δpose_{t→t+1}
            #
            # OUTPUT DICTIONARIES (PER HAND):
            # -------------------------------
            # win_left and win_right each contain arrays of shape (W, ...):
            #
            # Current frame representations:
            # - 'R_cam': (W, 3, 3) - Camera-space wrist orientations
            # - 't_cam': (W, 3) - Camera-space wrist translations
            # - 'pose_euler': (W, 45) - Finger angles in Euler format (15 joints × 3)
            # - 'hand_P': (W, 15, 3, 3) - Finger joint rotation matrices
            # - 'joints_manospace': (W, 21, 3) - Hand keypoints in MANO canonical space
            # - 'kept': (W,) bool - Validity mask for current frames
            #
            # Next frame representations (for learning temporal dynamics):
            # - 'R_cam_next': (W, 3, 3) - Wrist orientation at t+1
            # - 't_cam_next': (W, 3) - Wrist translation at t+1
            # - 'pose_euler_next': (W, 45) - Finger angles at t+1
            # - 'hand_P_next': (W, 15, 3, 3) - Finger rotations at t+1
            # - 'joints_manospace_next': (W, 21, 3) - Keypoints at t+1
            # - 'kept_next': (W,) bool - Validity mask for next frames
            #
            # USAGE EXAMPLE:
            # -------------
            # Given:
            #   - frame_id = 10, action_past_window_size = 0, action_future_window_size = 15
            #   - Left hand text: "Pick cup" [frames 5-25]
            #   - Right hand text: "Hold plate" [frames 10-30]
            #   
            # After _build_instruction:
            #   - idx_win_left = [10,11,...,25], start_left=5, end_left=25
            #   - idx_win_right = [10,11,...,25], start_right=10, end_right=30
            #   
            # win_left and win_right each contain 16 frames:
            #   - pose_euler[0] = left/right hand pose at frame 10 (current state)
            #   - pose_euler_next[0] = left/right hand pose at frame 11 (target prediction)
            #   - ...
            #   - pose_euler[15] = pose at frame 25
            #   - pose_euler_next[15] = pose at frame 26
            #
            # This paired structure allows the model to learn:
            #   action_t = f(state_t, state_{t+1})
            # for each hand independently, then combine them for dual-hand prediction.
            # ============================================================
            """
            win_left  = self._prepare_side_window(
                epi['left'],  R_w2c, t_w2c, idx_win_left, frame_id, anchor_frame=True, 
                oob=oob_left, start=start_left, end=end_left, upsample_factor=self.upsample_factor
            )
            win_right = self._prepare_side_window(
                epi['right'], R_w2c, t_w2c, idx_win_right, frame_id, anchor_frame=True, 
                oob=oob_right, start=start_right, end=end_right, upsample_factor=self.upsample_factor
            )
            idx_center = action_past_window_size # 0, local index of t0 in window
            
            """
            # ============================================================
            # 5. COMPUTE ABSOLUTE & RELATIVE HAND POSE ACTIONS (DUAL-HAND)
            # ============================================================
            # Converts paired (current, next) MANO hand pose representations into
            # action sequences for supervised learning. Computes both absolute and
            # relative pose deltas across the temporal window FOR EACH HAND.
            #
            # PURPOSE:
            # --------
            # Transforms the raw MANO hand pose windows into trainable action representations
            # that the model will learn to predict. Actions encode the change in hand pose
            # from one timestep to the next, enabling the model to learn bimanual manipulation
            # dynamics and coordination between left and right hands.
            #
            # INPUTS (PER HAND):
            # ------------------
            # - win_left / win_right: Dictionaries from _prepare_side_window containing:
            #     * R_cam, t_cam: (W, 3, 3) and (W, 3) - Current frame wrist poses
            #     * R_cam_next, t_cam_next: (W, 3, 3) and (W, 3) - Next frame wrist poses
            #     * pose_euler: (W, 45) - Current frame finger angles (15 joints × 3 Euler)
            #     * pose_euler_next: (W, 45) - Next frame finger angles
            #     * hand_P: (W, 15, 3, 3) - Current frame finger rotation matrices
            #     * hand_P_next: (W, 15, 3, 3) - Next frame finger rotation matrices
            #     * joints_manospace: (W, 21, 3) - Current frame keypoints in MANO space
            #     * joints_manospace_next: (W, 21, 3) - Next frame keypoints
            #     * kept, kept_next: (W,) bool - Validity masks for current/next frames
            #   where W = temporal window length (typically 16)
            #   Note: W can differ between left and right hands if text annotations differ!
            #
            # - anchor_idx: Scalar index of the anchor frame within the window
            #   Typically = action_past_window_size (e.g., 0 if no past context)
            #   This is the "current observation" frame t₀
            #
            # - rel_mode: String specifying the reference frame for relative actions
            #   * "step": Compute frame-to-frame deltas Δ(t_i → t_{i+1})
            #     Encodes how pose changes from each frame to its immediate successor
            #     Example: If at frame 10, compute change from frame 10→11
            #     This is the DEFAULT mode for training
            #   
            #   * "anchor": Compute deltas relative to anchor frame Δ(t₀ → t_{i+1})
            #     All poses are expressed relative to the current observation
            #     Example: If anchor is frame 10, compute 10→11, 10→12, ..., 10→25
            #     Useful for long-horizon prediction from a fixed reference
            #
            # - action_type: String specifying pose representation format
            #   * "angle": Use Euler angles (xyz convention) for finger joints
            #     Output per hand: 51 dimensions = 3 (wrist_t) + 3 (wrist_R) + 45 (fingers)
            #     This is the DEFAULT mode for compact neural network inputs
            #   
            #   * "keypoints": Use 3D joint positions in MANO canonical space
            #     Output per hand: 66 dimensions = 3 (wrist_t) + 3 (wrist_R) + 63 (21 joints × 3)
            #     Provides geometric interpretability but higher dimensionality
            #
            # PROCESSING PIPELINE (PER HAND):
            # -------------------------------
            # 1. Extract current and next-frame hand pose components:
            #    - Wrist translation: t_cur, t_nxt (camera space)
            #    - Wrist rotation: R_cur, R_nxt (3×3 matrices)
            #    - Finger rotations: P_cur, P_nxt (15×3×3 matrices)
            #    - Finger angles: pose_euler, pose_euler_next (45-dim Euler)
            #    - Keypoints: joints_manospace, joints_manospace_next (21×3)
            #    - Validity: kept, kept_next (boolean masks)
            #
            # 2. Compute ABSOLUTE actions (target poses at t+1):
            #    
            #    If action_type == "angle":
            #      action_abs = [t_next, euler(R_next), finger_euler_next]
            #      Shape: (W, 51) = (W, 3 + 3 + 45)
            #    
            #    If action_type == "keypoints":
            #      action_abs = [t_next, euler(R_next), joints_next.flatten()]
            #      Shape: (W, 66) = (W, 3 + 3 + 63)
            #    
            #    Components:
            #    - Wrist translation at t+1: (3,) in camera space
            #    - Wrist rotation at t+1: (3,) as Euler angles
            #    - Finger configuration at t+1: (45,) Euler or (63,) keypoints
            #
            # 3. Compute RELATIVE actions (pose deltas):
            #    
            #    If rel_mode == "step" (frame-to-frame):
            #      Δt = t_next - t_cur                    # Translation delta
            #      ΔR = R_next @ R_cur^T                  # Rotation delta (composition)
            #      ΔP = P_next @ P_cur^T                  # Finger rotation deltas (15 joints)
            #      Δjoints = joints_next - joints_cur     # Keypoint deltas (if keypoints mode)
            #      valid = kept & kept_next               # Both frames must be valid
            #    
            #    If rel_mode == "anchor" (anchor-relative):
            #      Δt = t_next - t_anchor                 # Delta from anchor to next
            #      ΔR = R_next @ R_anchor^T               # Rotation from anchor to next
            #      ΔP = P_next @ P_anchor^T               # Finger deltas from anchor
            #      Δjoints = joints_next - joints_anchor  # Keypoint deltas from anchor
            #      valid = kept_next & kept[anchor_idx]   # Anchor and next must be valid
            #    
            #    Convert rotation deltas to Euler angles:
            #      ΔP_euler = euler(ΔP)                   # (W, 45) Euler angle deltas
            #    
            #    Combine into action vector:
            #      If action_type == "angle":
            #        action_rel = [Δt, euler(ΔR), ΔP_euler]
            #        Shape: (W, 51) = (W, 3 + 3 + 45)
            #      
            #      If action_type == "keypoints":
            #        action_rel = [Δt, euler(ΔR), Δjoints.flatten()]
            #        Shape: (W, 66) = (W, 3 + 3 + 63)
            #
            # 4. Apply validity masking:
            #    - Set action_abs[~valid] = 0.0  (zero out invalid absolute actions)
            #    - Set action_rel[~valid] = 0.0  (zero out invalid relative actions)
            #    This prevents the model from learning from corrupted/extrapolated data
            #
            # OUTPUT TENSORS (PER HAND):
            # -------------------------
            # Left hand:
            #   abs_L: (W, 51) or (W, 66) float32 - Absolute left hand pose actions
            #   rel_L: (W, 51) or (W, 66) float32 - Relative left hand pose actions
            #   msk_L: (W,) bool - Left hand action validity mask
            #
            # Right hand:
            #   abs_R: (W, 51) or (W, 66) float32 - Absolute right hand pose actions
            #   rel_R: (W, 51) or (W, 66) float32 - Relative right hand pose actions
            #   msk_R: (W,) bool - Right hand action validity mask
            #
            # Action format (assuming action_type="angle", the default):
            #   [wrist_t(3), wrist_R_euler(3), finger_joints(45)] = 51 dims per hand
            #   
            #   Represents:
            #   - Wrist position change/target in camera space (meters)
            #   - Wrist orientation change/target as Euler angles (radians)
            #   - 15 finger joints × 3 Euler angles (radians)
            #
            # COORDINATE FRAME:
            # ----------------
            # All actions are in CAMERA SPACE (not world space):
            # - Translations are in meters relative to camera origin
            # - Rotations are relative to camera coordinate frame
            # This camera-centric representation is critical for egocentric action
            # prediction, as it makes poses invariant to global camera motion.
            #
            # DUAL-HAND COORDINATION:
            # -----------------------
            # By computing actions for both hands independently then concatenating,
            # the model can learn:
            # 1. Independent hand motions (e.g., left picks, right holds)
            # 2. Coordinated bimanual tasks (e.g., both hands lift together)
            # 3. Hand-specific timing (e.g., left starts before right)
            #
            # The text instruction guides which hand does what:
            #   "Left hand: Pick up cup. Right hand: Hold plate."
            # enables the model to associate text semantics with hand-specific actions.
            #
            # USAGE EXAMPLE:
            # -------------
            # Given:
            #   - frame_id = 10, action_past_window_size = 0, action_future_window_size = 15
            #   - idx_win_left = idx_win_right = [10, 11, 12, ..., 25] (W=16)
            #   - idx_center = 0 (anchor is first frame in window)
            #   - rel_mode = "step"
            #   - action_type = "angle"
            #
            # Processing:
            #   For left hand:
            #     win_left contains paired poses:
            #       pose_euler[0] = left hand at frame 10, pose_euler_next[0] = at frame 11
            #       pose_euler[1] = at frame 11, pose_euler_next[1] = at frame 12
            #       ...
            #       pose_euler[15] = at frame 25, pose_euler_next[15] = at frame 26
            #   
            #   Similar for right hand in win_right
            #
            # Output:
            #   abs_L[0] = absolute left hand pose at frame 11 (51 dims)
            #   rel_L[0] = left hand pose delta from frame 10→11 (51 dims)
            #   msk_L[0] = True if both frames 10 and 11 have valid left hand data
            #   
            #   abs_R[0] = absolute right hand pose at frame 11 (51 dims)
            #   rel_R[0] = right hand pose delta from frame 10→11 (51 dims)
            #   msk_R[0] = True if both frames 10 and 11 have valid right hand data
            #   
            #   ... (similarly for indices 1 through 15)
            #
            # These per-hand actions will be concatenated into:
            #   action_abs = [abs_L, abs_R] (W, 102) - Combined absolute actions
            #   action_rel = [rel_L, rel_R] (W, 102) - Combined relative actions
            #   action_mask = [msk_L, msk_R] (W, 2) - Per-hand validity masks
            #
            # The model learns to predict:
            #   Given: Current bimanual observation at frame t and text instruction
            #   Predict: rel_L[i], rel_R[i] = Δpose for each hand to reach frame t+i+1
            # ============================================================
            """
            abs_L, rel_L, msk_L = self._make_action_window_vec(
                win_left,  anchor_idx=idx_center, rel_mode=rel_mode, action_type=self.action_type
            ) # [16, 51] [16, 51] [16,]

            abs_R, rel_R, msk_R = self._make_action_window_vec(
                win_right, anchor_idx=idx_center, rel_mode=rel_mode, action_type=self.action_type
            ) # [16, 51] [16, 51] [16,]

            action_abs = np.concatenate([abs_L, abs_R], axis=1)   # (W,102)
            action_rel = np.concatenate([rel_L, rel_R], axis=1)   # (W,102)
            action_mask = np.stack([msk_L, msk_R], axis=1)        # (W,2)

            """
            # ============================================================
            # 6. CONSTRUCT CURRENT STATE (MANO DUAL-HAND)
            # ============================================================
            # Builds the current observation state at anchor frame (frame_id).
            # This represents "what the robot/model sees NOW" before predicting future actions.
            #
            # CRITICAL DISTINCTION: STATE vs ACTION
            # =====================================
            #
            # **current_state**:
            #   - TEMPORAL: Single timestep (snapshot at time t₀)
            #   - CONTENT: Includes pose AND shape parameters (betas)
            #   - FORMAT: Always ABSOLUTE pose in camera space
            #   - PURPOSE: Current observation input to the model
            #   - SHAPE: (122,) = 2 hands × 61 dims per hand
            #     * Per hand: t_cam(3) + euler(R_cam)(3) + pose_euler(45) + betas(10) = 61
            #
            # **action_list**:
            #   - TEMPORAL: Sequence of W timesteps (future trajectory from t₁ to t_W)
            #   - CONTENT: Only pose parameters (NO betas - shape is constant per episode)
            #   - FORMAT: Can be RELATIVE (deltas, default) or ABSOLUTE (target poses)
            #   - PURPOSE: Target predictions for the model to learn
            #   - SHAPE: (W, 102) = W timesteps × 2 hands × 51 dims per hand
            #     * Per hand: t_cam(3) + euler(R_cam)(3) + pose_euler(45) = 51
            #
            # WHY ARE THEY DIFFERENT?
            # -----------------------
            # 1. **Shape parameters (betas)**:
            #    - Included in STATE: Model needs to know hand size/shape to understand observations
            #    - Excluded from ACTIONS: Hand shape is CONSTANT during an episode (doesn't change)
            #      * Including betas in actions would be redundant (same 10 values repeated W times)
            #      * Saves memory: 102 dims vs 122 dims per timestep
            #
            # 2. **Temporal dimension**:
            #    - STATE is 1D: (122,) - Single snapshot "I see THIS hand configuration now"
            #    - ACTIONS are 2D: (W, 102) - Trajectory "Move hands like THIS over next W frames"
            #
            # 3. **Absolute vs Relative**:
            #    - STATE is always ABSOLUTE: "Hands are at position [x,y,z] with rotation [r,p,y]"
            #    - ACTIONS can be RELATIVE: "Move hands by Δx, Δy, Δz with Δrotation"
            #      * Relative actions are easier for neural networks to learn (smaller values)
            #      * Enables compositional prediction: action_t+1 = f(state_t, Δaction)
            #
            # CONCRETE EXAMPLE:
            # ----------------
            # frame_id = 100, action_future_window_size = 15 (W=16)
            #
            # current_state (122 dims):
            #   Left:  [0.25, 0.13, 0.45, 0.1, -0.2, 0.0, <45 finger angles>, <10 betas>]
            #   Right: [0.30, 0.10, 0.50, 0.0,  0.1, 0.0, <45 finger angles>, <10 betas>]
            #   → Represents: "At frame 100, left hand is at (0.25, 0.13, 0.45) with fingers curled"
            #
            # action_list (16, 102) - if use_rel=True:
            #   Row 0:  [Left Δpose from frame 100→101, Right Δpose from frame 100→101]
            #   Row 1:  [Left Δpose from frame 101→102, Right Δpose from frame 101→102]
            #   ...
            #   Row 15: [Left Δpose from frame 115→116, Right Δpose from frame 115→116]
            #   → Represents: "To reach frame 101, move left hand by [+0.01, +0.02, -0.01, ...]"
            #
            # MODEL TRAINING:
            # --------------
            # Input:  current_state (122,) + image_list + instruction
            # Output: Predict action_list (W, 102)
            # Loss:   Compare predicted actions vs ground truth actions
            # 
            # At inference:
            #   1. Observe current state at frame t
            #   2. Model predicts future hand motions (actions)
            #   3. Execute actions to control robot hands
            #   4. Update state based on executed actions
            #   5. Repeat for next timestep
            # ============================================================
            """
            cur_L = self._pack_state(win_left['R_cam'],
                        win_left['t_cam'],
                        win_left['pose_euler'] if self.action_type=='angle' else win_left['joints_manospace'].reshape(W, -1),
                        idx_center) # [3 + 3 + 45] = 51 dims (NO betas yet)

            cur_R = self._pack_state(win_right['R_cam'],
                        win_right['t_cam'],
                        win_right['pose_euler'] if self.action_type=='angle' else win_right['joints_manospace'].reshape(W, -1),
                        idx_center) # [3 + 3 + 45] = 51 dims (NO betas yet)

            betas_L = epi['left']['beta']   # [10] - MANO shape parameters (constant per episode)
            betas_R = epi['right']['beta']  # [10] - MANO shape parameters (constant per episode)
                                                                                   
            # Construct full state with shape parameters
            # Left hand:  t_cam(3) + euler(R_cam)(3) + pose_euler(45) + betas(10) = 61 dims
            # Right hand: t_cam(3) + euler(R_cam)(3) + pose_euler(45) + betas(10) = 61 dims
            # Total: 122 dims
            current_state       = np.concatenate([cur_L, betas_L, cur_R, betas_R]) # 2 * (3+3+45+10,) = 122
            current_state_mask  = np.array([win_left['kept'][idx_center], win_right['kept'][idx_center]]) # [2]

            # ------------------------------------------------------------------
            # 5. RGB window
            # ------------------------------------------------------------------
            if load_images:
                image_list, image_mask = self._grab_window_images(
                    episode_id, epi,
                    frame_id,
                    image_past_window_size,
                    image_future_window_size
                ) # len(image_list) = 1
                H = image_list[0].shape[0] # 240
                W = image_list[0].shape[1] # 360
            else:
                image_list = None
                image_mask = None
                H, W = epi['intrinsics'][1,2]*2, epi['intrinsics'][0,2]*2
            
            # ------------------------------------------------------------------
            # 6. Calculate New_intrinsics
            # ------------------------------------------------------------------
            dataset_name = episode_id.split('_')[0] # somethingsomethingv2
            intrinsics = epi['intrinsics']          # [3, 3]

            if dataset_name == 'EgoExo4D':
                # For EgoExo4D, the fisheye camera images contain black borders after undistortion.
                # We remove these borders using a center crop. Specifically, the video frames are
                # first resized from 1408 to 448, and then center-cropped to 256.

                new_intrinsics = compute_new_intrinsics_crop(intrinsics, 1408, 256/448*1408, H)
                
            else:
                new_intrinsics = compute_new_intrinsics_resize(intrinsics, (H, W)) # [3, 3]

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

            fov = calculate_fov( 2 * new_intrinsics[1][2], 2 * new_intrinsics[0][2], new_intrinsics) # [2]

            """
            # ============================================================
            # 7. SELECT ACTION REPRESENTATION MODE (MANO dual-hand)
            # ============================================================
            # Same concept as SMPLX above, but for dual hands (left + right).
            #
            # At this point:
            #   - action_rel: (W, 102) = [left_hand(51), right_hand(51)]
            #   - action_abs: (W, 102) = [left_hand(51), right_hand(51)]
            #
            # Each hand has 51 dimensions:
            #   [0:3]   - Translation (wrist position)
            #   [3:6]   - Wrist rotation (Euler angles)
            #   [6:51]  - Finger joint angles (45 dims = 15 joints × 3 Euler angles)
            #
            # WHY `//2` IS ESSENTIAL HERE:
            # ---------------------------
            # Unlike SMPLX, the `//2` operation is CRITICAL for MANO because:
            #   1. action_rel and action_abs contain TWO hands concatenated
            #   2. We need to FIRST separate left hand from right hand
            #   3. THEN apply the hybrid logic (rel root + abs joints) to EACH hand
            #
            # Process:
            #   Step 1: Split by //2 to separate hands
            #     rel_L = action_rel[:, :51]   # Left hand relative actions
            #     rel_R = action_rel[:, 51:]   # Right hand relative actions
            #     abs_L = action_abs[:, :51]   # Left hand absolute actions
            #     abs_R = action_abs[:, 51:]   # Right hand absolute actions
            #
            #   Step 2: For EACH hand, apply hybrid (relative root + absolute joints)
            #     Left:  [rel_L[:, :6], abs_L[:, 6:]] = [Δt, ΔR, fingers_abs]
            #     Right: [rel_R[:, :6], abs_R[:, 6:]] = [Δt, ΔR, fingers_abs]
            #
            #   Step 3: Concatenate both hands
            #     action_list = [left_hybrid(51), right_hybrid(51)] = 102 dims
            #
            # So the [:, :6] slicing happens AFTER the //2 split, operating on
            # each hand's 51-dim representation independently.
            # ============================================================
            """
            if self.use_rel:
                action_list = action_rel
            else:
                # use abs action for hand pose only (hybrid mode for dual hands)
                
                # Step 1: Split concatenated hands into left and right
                rel_L = action_rel[:, :action_rel.shape[1]//2] # [16, 51] left hand
                rel_R = action_rel[:, action_rel.shape[1]//2:] # [16, 51] right hand
                abs_L = action_abs[:, :action_abs.shape[1]//2] # [16, 51] left hand
                abs_R = action_abs[:, action_abs.shape[1]//2:] # [16, 51] right hand

                # Step 2 & 3: Apply hybrid logic to each hand, then concatenate
                # For each hand: [relative_root(6), absolute_fingers(45)]
                action_list = np.concatenate([
                    rel_L[:, :6], abs_L[:, 6:],  # Left hand hybrid (51 dims)
                    rel_R[:, :6], abs_R[:, 6:]   # Right hand hybrid (51 dims)
                ], axis=1) # [16, 102] total

            """
            # ============================================================
            # 8. PACKAGE RETURN DICTIONARY (MANO DUAL-HAND)
            # ============================================================
            # Returns a complete training sample containing:
            #   - Current observation (state + images)
            #   - Future trajectory to predict (actions)
            #   - Task description (instruction)
            #   - Camera parameters (intrinsics, fov)
            #
            # SUMMARY: STATE vs ACTION DIMENSIONS
            # ===================================
            #
            # **current_state**: (122,) float32 - SINGLE TIMESTEP with SHAPE
            #   Format: [left_hand(61), right_hand(61)]
            #   Per hand breakdown:
            #     - t_cam: (3,) wrist translation in camera space
            #     - euler(R_cam): (3,) wrist rotation as Euler angles
            #     - pose_euler: (45,) finger joint angles (15 joints × 3)
            #     - betas: (10,) MANO shape parameters (PCA coefficients)
            #   Total: 3 + 3 + 45 + 10 = 61 per hand × 2 hands = 122 dims
            #
            # **action_list**: (W, 102) float32 - W TIMESTEPS without SHAPE
            #   Format: [left_hand(51), right_hand(51)] at each of W timesteps
            #   Per hand breakdown (NO betas):
            #     - t_cam: (3,) wrist translation (or Δt if relative)
            #     - euler(R_cam): (3,) wrist rotation (or Δeuler if relative)
            #     - pose_euler: (45,) finger angles (or Δpose if relative)
            #   Total: 3 + 3 + 45 = 51 per hand × 2 hands = 102 dims per timestep
            #   Across W timesteps: (W, 102)
            #
            # **current_state_mask**: (2,) bool - Per-hand validity at anchor frame
            #   [left_valid, right_valid]
            #   True if hand has valid MANO reconstruction at frame_id
            #
            # **action_mask**: (W, 2) bool - Per-hand validity across trajectory
            #   [:, 0] = left hand validity for each of W timesteps
            #   [:, 1] = right hand validity for each of W timesteps
            #   True if both current AND next frame have valid hand data
            #
            # **instruction**: str - Natural language task description
            #   Example: "Left hand: Pick up the cup. Right hand: Hold the plate."
            #   Guides which hand performs which action
            #
            # **fov**: (2,) float32 - Field of view [fov_x, fov_y] in radians
            #   Derived from camera intrinsics, used for perspective-aware prediction
            #
            # **intrinsics**: (3, 3) float32 - Camera intrinsic matrix K
            #   [[fx,  0, cx],
            #    [ 0, fy, cy],
            #    [ 0,  0,  1]]
            #   After augmentation/resizing, used for 3D-2D projection
            #
            # DOWNSTREAM PROCESSING:
            # ---------------------
            # This dict is passed to transform_trajectory() which:
            #   1. Normalizes actions and states using dataset statistics
            #   2. Pads to unified dimensions (action: 192, state: 212)
            #   3. Converts to PyTorch tensors for model training
            # ============================================================
            """

            # current_state: Model input (what the robot sees now)
            # action_list: Model output (what the robot should do next)
            result_dict = dict(
                instruction             = instruction,          # str - "Left hand: ... Right hand: ..."
                action_list             = action_list,          # (W, 102) float32 - future hand trajectory
                action_mask             = action_mask,          # (W, 2) bool - per-hand validity per timestep
                current_state           = current_state,        # (122,) float32 - current observation with shape
                current_state_mask      = current_state_mask,   # (2,) bool - per-hand validity at anchor
                fov                     = fov,                  # (2,) float32 - [fov_x, fov_y]
                intrinsics              = new_intrinsics,       # (3, 3) float32 - camera intrinsics K
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

        action_np = sample_dict["action_list"]   # [16, 69] or [16, 102]
        state_np  = sample_dict["current_state"] # [79] or [122]
       
        action_dim = action_np.shape[1] # 69 for SMPLX, 102 for MANO dual-hand
        state_dim  = state_np.shape[0]  # 79 for SMPLX, 122 for MANO dual-hand
        if normalization:
            # Normalize left and right hand actions and states separately
            action_np = self.gaussian_normalizer.normalize_action(action_np)    # [16, 102]
            state_np  = self.gaussian_normalizer.normalize_state(state_np)      # [122]

        # ===== Pad to unified dimensions =====
        unified_action_dim = ActionFeature.ALL_FEATURES[1]   # 192
        unified_state_dim  = StateFeature.ALL_FEATURES[1]    # 212
        unified_state, unified_state_mask = pad_state_human(
            state_np,
            sample_dict["current_state_mask"],
            action_dim,
            state_dim,
            unified_state_dim
        ) # [212]
        unified_action, unified_action_mask = pad_action(
            action_np,
            sample_dict["action_mask"],
            action_dim,
            unified_action_dim
        ) # [16, 192]

        sample_dict["action_list"] = unified_action             # [16, 192]
        sample_dict["action_mask"] = unified_action_mask        # [16, 192]
        sample_dict["current_state"] = unified_state            # [212]
        sample_dict["current_state_mask"] = unified_state_mask  # [212]
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
        current_state_mask (Tensor): per-entity state mask, shape [num_entities]
            - For dual-hand MANO: shape (2,) for [left, right]
            - For single-body SMPLX: shape (1,) for [body]
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
    
    num_entities = len(current_state_mask)  # 2 for MANO, 1 for SMPLX
    
    # Expand state mask from per-entity to per-dim
    if num_entities == 1:
        # Single-body case (SMPLX): Just expand to full state_dim
        expanded_state_mask = current_state_mask.repeat_interleave(state_dim)  # (state_dim,)
    else:
        # Dual-hand case (MANO): Expand each hand mask to half state_dim
        expanded_state_mask = current_state_mask.repeat_interleave(state_dim // num_entities)  # (state_dim,)

    # Mask out invalid state dimensions
    current_state_masked = current_state * expanded_state_mask.to(current_state.dtype)

    # Initialize output tensors
    padded_state = torch.zeros(unified_state_dim, dtype=current_state.dtype)
    padded_mask = torch.zeros(unified_state_dim, dtype=torch.bool)

    if num_entities == 1:
        # Single-body case: Fill directly without splitting
        padded_state[:action_dim] = current_state_masked[:action_dim].clone()
        padded_mask[:action_dim] = expanded_state_mask[:action_dim].clone()
    else:
        # Dual-hand case: Fill left and right hands separately (skip MANO betas)
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
        action_mask (Tensor): per-entity action mask, shape [T, num_entities]
            - For dual-hand MANO: shape [T, 2] for [left, right]
            - For single-body SMPLX: shape [T, 1] for [body]
        action_dim (int): original action dimension.
        unified_action_dim (int): target padded actions dimension.

    Returns:
        Tuple[Optional[Tensor], Tensor]:
            padded actions [T, unified_action_dim] or None,
            padded action mask [T, unified_action_dim]
    """
    
    action_mask = torch.tensor(action_mask, dtype=torch.bool)
    
    num_entities = action_mask.shape[1]  # 2 for MANO, 1 for SMPLX
    
    # Expand mask from per-entity to per-dimension
    if num_entities == 1:
        # Single-body case (SMPLX): Expand single mask to full action_dim
        expanded_action_mask = action_mask[:, 0].unsqueeze(1).expand(-1, action_dim)
    else:
        # Dual-hand case (MANO): Expand each hand mask to half action_dim
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