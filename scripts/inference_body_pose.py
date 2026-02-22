"""
Body Pose Inference on Validation Set
======================================
Iterates over the dataset built from the config's train_dataset settings, runs VLA
inference on each sample, and saves the body-pose predictions to a folder as .npz files.

Each output file is named  {output_dir}/{idx:06d}.npz  and contains:

  body_pose     : (sample_times, T, 63)  – 21 joints × 3 Euler angles (xyz)
  transl        : (sample_times, T, 3)   – camera-space root translation   [69-dim action only]
  global_orient : (sample_times, T, 3)   – camera-space root rotation (Euler) [69-dim only]
  raw_action    : (sample_times, T, raw_action_dim)  – full unnormalised prediction

  gt_body_pose  : (T, 63)               – ground-truth body joint angles
  gt_transl     : (T, 3)                – ground-truth root translation       [69-dim only]
  gt_global_orient : (T, 3)             – ground-truth root rotation (Euler)  [69-dim only]
  gt_raw_action : (T, raw_action_dim)   – full GT action sequence

A companion  {idx:06d}_meta.json  is written alongside with:
  episode_id, frame_id, instruction, dataset_name

Usage
-----
  python scripts/inference_body_pose.py \\
      --config vitra/configs/pose_pretrain_cheston.json \\
      --model_path data/vla_checkpoint/vitra_vla_3b/checkpoints/pretrain_TB96_B32_bf16True/checkpoints/epoch=0-step=700.ckpt/weights.pt \\
      --output_dir ./body_pose_preds \\
      --max_samples 500 \\
      --sample_times 4
"""

import os
import sys
import json
import argparse
import random
import traceback
import numpy as np
import torch
import multiprocessing as mp
from pathlib import Path
from PIL import Image, ImageOps
from tqdm import tqdm

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from vitra.utils.data_utils import resize_short_side_to_target, load_normalizer
from vitra.utils.config_utils import load_config
from vitra.datasets.human_dataset import EpisodicDatasetCore, pad_state_human, pad_action
from vitra.datasets.dataset_utils import ActionFeature, StateFeature
from vitra.datasets.data_mixture import HAND_MIXTURES


# ---------------------------------------------------------------------------
# Helpers: build EpisodicDatasetCore from config paths
# ---------------------------------------------------------------------------

DATASET_PATHS = {
    "idea400": {
        "annotation_file": "Annotation/idea400/episode_frame_index.npz",
        "label_folder":    "Annotation/idea400/episodic_annotations",
        "statistics_path": "Annotation/statistics/idea400_statistics.json",
        "video_root":      "Video/idea400",
    },
    "ssv2": {
        "annotation_file": "Annotation/ssv2/episode_frame_index.npz",
        "label_folder":    "Annotation/ssv2/episodic_annotations",
        "statistics_path": "Annotation/statistics/ssv2_angle_statistics.json",
        "video_root":      "Video/Somethingsomething-v2_root",
    },
    "ego4d_cooking_and_cleaning": {
        "annotation_file": "Annotation/ego4d_cooking_and_cleaning/episode_frame_index.npz",
        "label_folder":    "Annotation/ego4d_cooking_and_cleaning/episodic_annotations",
        "statistics_path": "Annotation/statistics/ego4d_cooking_and_cleaning_angle_statistics.json",
        "video_root":      "Video/Ego4D_root",
    },
    "egoexo4d": {
        "annotation_file": "Annotation/egoexo4d/episode_frame_index.npz",
        "label_folder":    "Annotation/egoexo4d/episodic_annotations",
        "statistics_path": "Annotation/statistics/egoexo4d_angle_statistics.json",
        "video_root":      "Video/EgoExo4D_root",
    },
    "epic": {
        "annotation_file": "Annotation/epic/episode_frame_index.npz",
        "label_folder":    "Annotation/epic/episodic_annotations",
        "statistics_path": "Annotation/statistics/epic_angle_statistics.json",
        "video_root":      "Video/Epic-Kitchen_root",
    },
}


def _build_core(data_root: str, dataset_name: str, chunk_size: int,
                denoising_mode: bool, denoising_noise_std: float) -> EpisodicDatasetCore:
    """Instantiate an EpisodicDatasetCore for the given dataset."""
    if dataset_name not in DATASET_PATHS:
        raise ValueError(f"Unknown dataset_name '{dataset_name}'. "
                         f"Supported: {list(DATASET_PATHS)}")
    p = DATASET_PATHS[dataset_name]
    return EpisodicDatasetCore(
        video_root      = os.path.join(data_root, p["video_root"]),
        annotation_file = os.path.join(data_root, p["annotation_file"]),
        label_folder    = os.path.join(data_root, p["label_folder"]),
        training_path   = None,
        statistics_path = os.path.join(data_root, p["statistics_path"]),
        augmentation    = False,
        flip_augmentation = 1.0,
        set_none_ratio  = 0.0,
        action_type     = "angle",
        use_rel         = False,
        clip_len        = None,
        state_mask_prob = 0.0,          # never mask state during inference
        action_past_window_size  = 0,
        action_future_window_size = chunk_size - 1,
        load_images     = True,
        target_image_height = 224,
        denoising_mode  = denoising_mode,
        denoising_noise_std = denoising_noise_std,
    )


# ---------------------------------------------------------------------------
# VLA inference worker (persistent, runs in a separate spawn process)
# ---------------------------------------------------------------------------

def _vla_body_worker(configs_dict, task_queue, result_queue):
    """
    Persistent VLA worker. Handles one-body (SMPLX) inference.
    Normalises + pads inputs, runs model.predict_action(), unnormalises output.
    When use_normalization=False the inputs/outputs bypass Gaussian scaling.
    """
    from vitra.models import load_model
    from vitra.utils.data_utils import load_normalizer
    from vitra.datasets.human_dataset import pad_state_human, pad_action
    from vitra.datasets.dataset_utils import ActionFeature, StateFeature

    model = normalizer = None
    try:
        print("[VLA-Body Process] Loading model …")
        model = load_model(configs_dict).cuda()
        model.eval()

        use_normalization = configs_dict.get("_use_normalization", True)
        if use_normalization:
            normalizer = load_normalizer(configs_dict)
            raw_action_dim = int(normalizer.action_mean.shape[0])
            raw_state_dim  = int(normalizer.state_mean.shape[0])
        else:
            normalizer     = None
            raw_action_dim = int(configs_dict.get("_raw_action_dim", 69))
            raw_state_dim  = int(configs_dict.get("_raw_state_dim",  69))

        print(f"[VLA-Body Process] Ready.  use_normalization={use_normalization}  "
              f"raw_action_dim={raw_action_dim}  raw_state_dim={raw_state_dim}")
        result_queue.put({"type": "ready",
                          "raw_action_dim": raw_action_dim,
                          "raw_state_dim":  raw_state_dim})

        while True:
            task = task_queue.get()

            if task["type"] == "shutdown":
                break

            elif task["type"] == "predict":
                try:
                    image        = task["image"]        # np.ndarray (H, W, 3) uint8
                    instruction  = task["instruction"]  # str
                    state        = task["state"]        # (raw_state_dim,) float32
                    fov          = task["fov"]           # (2,) float32
                    chunk_size   = task["chunk_size"]   # int
                    num_ddim     = task.get("num_ddim_steps", 10)
                    cfg_scale    = task.get("cfg_scale", 5.0)
                    sample_times = task.get("sample_times", 1)

                    unified_action_dim = ActionFeature.ALL_FEATURES[1]   # 192
                    unified_state_dim  = StateFeature.ALL_FEATURES[1]    # 212

                    # Crop state to raw_state_dim in case dataset returns more
                    state = state[:raw_state_dim]

                    # Normalise state (skip if model was trained without normalisation)
                    norm_state = (normalizer.normalize_state(state.copy())
                                  if normalizer is not None else state.copy())

                    # Single-body masks  (num_entities = 1)
                    state_mask  = np.array([True], dtype=bool)           # (1,)
                    action_mask = np.ones((chunk_size, 1), dtype=bool)   # (T, 1)

                    # Pad to unified dims
                    unified_state, unified_state_mask = pad_state_human(
                        state        = norm_state,
                        state_mask   = state_mask,
                        action_dim   = raw_action_dim,
                        state_dim    = raw_state_dim,
                        unified_state_dim = unified_state_dim,
                    )
                    _, unified_action_mask = pad_action(
                        actions       = None,
                        action_mask   = action_mask,
                        action_dim    = raw_action_dim,
                        unified_action_dim = unified_action_dim,
                    )

                    # To GPU
                    fov_t               = torch.from_numpy(fov).unsqueeze(0)
                    unified_state       = unified_state.unsqueeze(0)
                    unified_state_mask  = unified_state_mask.unsqueeze(0)
                    unified_action_mask = unified_action_mask.unsqueeze(0)

                    # Inference
                    norm_action = model.predict_action(
                        image              = image,
                        instruction        = instruction,
                        current_state      = unified_state,
                        current_state_mask = unified_state_mask,
                        action_mask_torch  = unified_action_mask,
                        num_ddim_steps     = num_ddim,
                        cfg_scale          = cfg_scale,
                        fov                = fov_t,
                        sample_times       = sample_times,
                    )

                    # Extract and unnormalise
                    norm_action = norm_action[:, :, :raw_action_dim]    # (S, T, raw_action_dim)
                    if normalizer is not None:
                        unnorm_action = normalizer.unnormalize_action(norm_action)
                    else:
                        unnorm_action = norm_action   # already in original scale
                    if isinstance(unnorm_action, torch.Tensor):
                        unnorm_action = unnorm_action.cpu().numpy()

                    result_queue.put({"type": "result", "success": True,
                                      "data": unnorm_action,
                                      "raw_action_dim": raw_action_dim})

                except Exception as exc:
                    result_queue.put({"type": "result", "success": False,
                                      "error": str(exc),
                                      "traceback": traceback.format_exc()})

    except Exception as exc:
        result_queue.put({"type": "error", "error": str(exc),
                          "traceback": traceback.format_exc()})
    finally:
        del model, normalizer
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("[VLA-Body Process] Cleaned up.")


class VLABodyService:
    """Thin wrapper around _vla_body_worker."""

    def __init__(self, configs):
        ctx = mp.get_context("spawn")
        self._task_q   = ctx.Queue()
        self._result_q = ctx.Queue()
        self._proc = ctx.Process(target=_vla_body_worker,
                                 args=(configs, self._task_q, self._result_q))
        self._proc.start()
        msg = self._result_q.get()
        if msg["type"] == "ready":
            self.raw_action_dim = msg["raw_action_dim"]
            self.raw_state_dim  = msg["raw_state_dim"]
            print(f"VLA body service ready  "
                  f"(raw_action_dim={self.raw_action_dim}, "
                  f"raw_state_dim={self.raw_state_dim})")
        else:
            raise RuntimeError(f"Worker init failed: {msg.get('error')}\n"
                               f"{msg.get('traceback','')}")

    def predict(self, image, instruction, state, fov, chunk_size,
                num_ddim_steps=10, cfg_scale=5.0, sample_times=1):
        self._task_q.put(dict(type="predict", image=image,
                              instruction=instruction, state=state,
                              fov=fov, chunk_size=chunk_size,
                              num_ddim_steps=num_ddim_steps,
                              cfg_scale=cfg_scale,
                              sample_times=sample_times))
        res = self._result_q.get()
        if res["type"] == "result" and res["success"]:
            return res["data"], res["raw_action_dim"]
        raise RuntimeError(f"Inference failed: {res.get('error')}\n"
                           f"{res.get('traceback','')}")

    def shutdown(self):
        self._task_q.put({"type": "shutdown"})
        self._proc.join(timeout=15)
        if self._proc.is_alive():
            self._proc.terminate()
            self._proc.join()


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def _save_prediction(output_dir: str, idx: int,
                     unnorm_action: np.ndarray,
                     raw_action_dim: int,
                     meta: dict,
                     gt_action: np.ndarray = None):
    """Save one prediction + optional GT + metadata to output_dir.

    Parameters
    ----------
    unnorm_action : (sample_times, T, raw_action_dim)
    gt_action     : (T, raw_action_dim)  – ground-truth action sequence from the dataset
    """
    stem = f"{idx:06d}"

    save_kwargs = {"raw_action": unnorm_action}

    if raw_action_dim <= 63:
        # action IS body_pose  (pose_pretrain / denoising)
        save_kwargs["body_pose"] = unnorm_action                    # (S, T, 63)
    else:
        # action = t(3) + R(3) + body_pose(63) = 69 dims  (human_pretrain)
        save_kwargs["transl"]        = unnorm_action[:, :, 0:3]
        save_kwargs["global_orient"] = unnorm_action[:, :, 3:6]
        save_kwargs["body_pose"]     = unnorm_action[:, :, 6:69]

    # ── ground-truth ──────────────────────────────────────────────────────
    if gt_action is not None:
        save_kwargs["gt_raw_action"] = gt_action                    # (T, raw_action_dim)
        if raw_action_dim <= 63:
            save_kwargs["gt_body_pose"] = gt_action                 # (T, 63)
        else:
            save_kwargs["gt_transl"]        = gt_action[:, 0:3]
            save_kwargs["gt_global_orient"] = gt_action[:, 3:6]
            save_kwargs["gt_body_pose"]     = gt_action[:, 6:69]

    np.savez(os.path.join(output_dir, stem + ".npz"), **save_kwargs)

    with open(os.path.join(output_dir, stem + "_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run body-pose VLA inference on the dataset and save results to a folder.")

    # ── model ──────────────────────────────────────────────────────────────
    parser.add_argument("--config", required=True,
                        help="Path to training config JSON  "
                             "(e.g. vitra/configs/pose_pretrain_cheston.json)")
    parser.add_argument("--model_path", default=None,
                        help="Path to weights.pt checkpoint (overrides config's model_load_path)")
    parser.add_argument("--statistics_path", default=None,
                        help="Path to dataset statistics JSON (overrides config)")

    # ── dataset ─────────────────────────────────────────────────────────────
    parser.add_argument("--data_root", default=None,
                        help="Root data directory (default: config's train_dataset.data_root_dir)")
    parser.add_argument("--dataset_name", default=None,
                        help="Single dataset name override, e.g. 'idea400'.  "
                             "Default: first dataset from config's data_mix.")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of dataset samples to run inference on. "
                             "Omit to process the entire dataset.")
    parser.add_argument("--shuffle", action="store_true",
                        help="Shuffle indices before selecting max_samples.")
    parser.add_argument("--seed", type=int, default=42)

    # ── inference settings ──────────────────────────────────────────────────
    parser.add_argument("--sample_times", type=int, default=4,
                        help="Number of independent action samples per image.")
    parser.add_argument("--num_ddim_steps", type=int, default=10)
    parser.add_argument("--cfg_scale", type=float, default=5.0)

    # ── output ──────────────────────────────────────────────────────────────
    parser.add_argument("--output_dir", default="./body_pose_preds",
                        help="Folder where per-sample .npz and _meta.json files are written.")

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # ── load config ────────────────────────────────────────────────────────
    configs = load_config(args.config)
    if args.model_path:
        configs["model_load_path"] = args.model_path
    if args.statistics_path:
        configs["statistics_path"] = args.statistics_path

    chunk_size        = configs.get("fwd_pred_next_n", 1)
    denoising_mode    = configs["train_dataset"].get("denoising_mode", False)
    denoising_std     = configs["train_dataset"].get("denoising_noise_std", 0.05)
    use_normalization = configs["train_dataset"].get("normalization", True)
    data_root         = args.data_root or configs["train_dataset"]["data_root_dir"]
    data_mix          = configs["train_dataset"]["data_mix"]

    # State/action dims depend on denoising_mode (strips root when True)
    # denoising → state=(63,), action=(63,); normal → state=(69,), action=(69,)
    raw_dim = 63 if denoising_mode else 69

    # Auto-derive statistics_path from data_root + dataset_name when normalization
    # is enabled but no explicit path was given in the config.
    if use_normalization and not configs.get("statistics_path"):
        # Will be resolved after dataset_name is known (see below)
        _auto_stats = True
    else:
        _auto_stats = False

    # Pass runtime flags to the worker via the configs dict (private keys)
    configs["_use_normalization"] = use_normalization
    configs["_raw_action_dim"]    = raw_dim
    configs["_raw_state_dim"]     = raw_dim

    # Resolve dataset_name from data_mix mapping, or use override
    if args.dataset_name:
        dataset_name = args.dataset_name
    else:
        if data_mix in HAND_MIXTURES:
            dataset_name = HAND_MIXTURES[data_mix][0][0]   # first dataset in mix
        else:
            dataset_name = data_mix
    print(f"Dataset : {dataset_name}  (data_mix={data_mix})")
    print(f"chunk_size={chunk_size}  denoising_mode={denoising_mode}  "
          f"use_normalization={use_normalization}")

    # Auto-derive statistics_path now that we know the dataset_name
    if _auto_stats and dataset_name in DATASET_PATHS:
        auto_stats_path = os.path.join(
            data_root, DATASET_PATHS[dataset_name]["statistics_path"])
        if os.path.exists(auto_stats_path):
            configs["statistics_path"] = auto_stats_path
            print(f"Auto-derived statistics_path: {auto_stats_path}")
        else:
            print(f"[WARN] statistics_path not found at {auto_stats_path}; "
                  f"disabling normalization.")
            configs["_use_normalization"] = False
            use_normalization = False

    # ── build dataset ──────────────────────────────────────────────────────
    print("Loading dataset …")
    core = _build_core(data_root, dataset_name, chunk_size, denoising_mode, denoising_std)
    n_total = len(core)
    print(f"Dataset size: {n_total} samples")

    indices = list(range(n_total))
    if args.shuffle:
        random.shuffle(indices)
    if args.max_samples is not None:
        indices = indices[:args.max_samples]
    print(f"Running inference on {len(indices)} samples.")

    # ── output directory ───────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Saving predictions to  {args.output_dir}/")

    # ── start VLA service ──────────────────────────────────────────────────
    service = VLABodyService(configs)

    n_ok = n_skip = n_err = 0
    try:
        for run_idx, ds_idx in enumerate(tqdm(indices, desc="Inference")):
            try:
                raw = core[ds_idx]
            except Exception as exc:
                print(f"[WARN] skip ds_idx={ds_idx}: {exc}")
                n_skip += 1
                continue

            # ── extract fields ──────────────────────────────────────────
            image_np     = raw["image_list"][-1]                       # (H, W, 3) uint8
            instruction  = raw["instruction"]                          # str
            current_state = raw["current_state"].astype(np.float32)   # (69,) SMPLX
            fov          = raw["fov"].astype(np.float32)              # (2,)
            # GT action sequence: (T, raw_action_dim) — float32 from the dataset
            gt_action    = raw["action_list"].astype(np.float32)       # (T, 69 or 63)

            # ── run inference ───────────────────────────────────────────
            try:
                unnorm_action, raw_action_dim = service.predict(
                    image        = image_np,
                    instruction  = instruction,
                    state        = current_state,
                    fov          = fov,
                    chunk_size   = chunk_size,
                    num_ddim_steps = args.num_ddim_steps,
                    cfg_scale    = args.cfg_scale,
                    sample_times = args.sample_times,
                )
            except Exception as exc:
                print(f"[ERR] inference failed at ds_idx={ds_idx}: {exc}")
                n_err += 1
                continue

            # ── recover episode/frame metadata ──────────────────────────
            corr       = core.index_frame_pair[ds_idx]
            episode_id = core.index_to_episode_id[corr[0]]
            frame_id   = int(corr[1])

            meta = {
                "run_idx":    run_idx,
                "ds_idx":     ds_idx,
                "episode_id": str(episode_id),
                "frame_id":   frame_id,
                "instruction": instruction,
                "dataset_name": dataset_name,
                "chunk_size": chunk_size,
                "sample_times": args.sample_times,
                "raw_action_dim": raw_action_dim,
            }

            _save_prediction(args.output_dir, run_idx, unnorm_action, raw_action_dim, meta,
                             gt_action=gt_action)
            n_ok += 1

    finally:
        service.shutdown()

    print(f"\nDone.  ok={n_ok}  skipped={n_skip}  errors={n_err}")
    print(f"Results → {args.output_dir}/")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
