"""
Visualize Body Pose Predictions with Open3D
============================================
Loads .npz prediction files saved by inference_body_pose.py and displays the
ground-truth and predicted SMPLX body poses using Open3D.

For each entry:
  - Ground truth pose  : blue mesh, placed at x = 0
  - Predicted sample i : red/warm mesh, placed at x = 1.5 * (i + 1)

Close the Open3D window (Q / Esc / window X) to advance to the next entry.

Usage
-----
  python scripts/visualize_body_pose.py \\
      --pred_dir   ./body_pose_preds \\
      --smplx_model_dir /home/haziq/VITRA/data/smplx \\
      [--timestep 0]          # which frame in the chunk sequence to visualize \\
      [--start_idx 0]         # start from this file index \\
      [--max_entries 500]     # stop after N entries
"""

import os
import sys
import json
import argparse
import glob
import numpy as np
import torch
from pathlib import Path
from scipy.spatial.transform import Rotation as R

import open3d as o3d

# ─────────────────────────────────────────────────────────────────────────────
# Euler (xyz) helpers
# ─────────────────────────────────────────────────────────────────────────────

def euler_xyz_to_aa(euler: np.ndarray) -> np.ndarray:
    """Convert Euler-xyz angles to axis-angle vectors (same shape)."""
    flat  = euler.reshape(-1, 3)
    aa    = R.from_euler('xyz', flat).as_rotvec()
    return aa.reshape(euler.shape).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# SMPLX forward pass → Open3D mesh
# ─────────────────────────────────────────────────────────────────────────────

def build_smplx_mesh(smplx_model,
                     body_pose_euler: np.ndarray,          # (63,)
                     global_orient_euler: np.ndarray = None,  # (3,) or None
                     transl: np.ndarray = None,            # (3,) or None
                     color: list = None,
                     x_offset: float = 0.0) -> o3d.geometry.TriangleMesh:
    """
    Run SMPLX forward pass and return a coloured Open3D TriangleMesh.

    Parameters
    ----------
    body_pose_euler   : (63,) body joint angles in Euler-xyz format
    global_orient_euler : (3,) root orientation in Euler-xyz format; zeros if None
    transl            : (3,) root translation; zeros if None
    color             : RGB list [0-1]; defaults to light grey
    x_offset          : translate mesh along x axis for side-by-side display
    """
    if color is None:
        color = [0.7, 0.7, 0.7]

    # ── convert Euler → axis-angle ────────────────────────────────────────
    body_pose_aa = euler_xyz_to_aa(body_pose_euler)           # (63,)

    if global_orient_euler is not None:
        global_orient_aa = euler_xyz_to_aa(global_orient_euler)  # (3,)
    else:
        global_orient_aa = np.zeros(3, dtype=np.float32)

    if transl is None:
        transl = np.zeros(3, dtype=np.float32)

    # ── SMPLX forward pass ────────────────────────────────────────────────
    with torch.no_grad():
        output = smplx_model(
            body_pose     = torch.tensor(body_pose_aa,      dtype=torch.float32).unsqueeze(0),
            global_orient = torch.tensor(global_orient_aa,  dtype=torch.float32).unsqueeze(0),
            transl        = torch.tensor(transl,             dtype=torch.float32).unsqueeze(0),
        )

    verts = output.vertices[0].cpu().numpy()   # (V, 3)
    faces = smplx_model.faces                  # (F, 3)  numpy int array

    # ── build Open3D mesh ─────────────────────────────────────────────────
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices  = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.paint_uniform_color(color)
    mesh.compute_vertex_normals()

    if x_offset != 0.0:
        mesh.translate([x_offset, 0.0, 0.0])

    return mesh


# ─────────────────────────────────────────────────────────────────────────────
# Load smplx library
# ─────────────────────────────────────────────────────────────────────────────

def load_smplx_model(smplx_model_dir: str):
    """Load SMPLX neutral model.  Requires the `smplx` Python package."""
    try:
        import smplx as smplx_lib
    except ImportError:
        print("[ERROR] smplx package not found.  Install with: pip install smplx")
        sys.exit(1)

    model = smplx_lib.create(
        model_path     = smplx_model_dir,
        model_type     = 'smplx',
        gender         = 'neutral',
        use_pca        = False,
        use_face_contour = False,
        batch_size     = 1,
    )
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Prediction colours (for multiple samples)
# ─────────────────────────────────────────────────────────────────────────────

PRED_COLORS = [
    [0.9, 0.2, 0.2],   # red
    [0.9, 0.6, 0.1],   # orange
    [0.8, 0.8, 0.0],   # yellow
    [0.5, 0.0, 0.8],   # purple
    [0.0, 0.7, 0.4],   # teal
    [0.2, 0.5, 1.0],   # light blue
]

GT_COLOR   = [0.2, 0.6, 1.0]   # cyan-blue
X_SPACING  = 1.6               # metres between meshes


# ─────────────────────────────────────────────────────────────────────────────
# Main visualisation loop
# ─────────────────────────────────────────────────────────────────────────────

def visualize_entry(smplx_model, npz_path: str, meta: dict,
                    timestep: int = 0) -> bool:
    """
    Render one entry.  Returns True if the window was closed normally,
    False if something went wrong.
    """
    data = np.load(npz_path)

    has_gt = "gt_body_pose" in data

    # ── unpack predicted body pose ────────────────────────────────────────
    pred_body_pose  = data["body_pose"]     # (S, T, 63) or (S, T, 63)
    S = pred_body_pose.shape[0]             # number of prediction samples
    T = pred_body_pose.shape[1]
    t = min(timestep, T - 1)

    has_transl = "transl"        in data
    has_orient = "global_orient" in data

    # ── collect meshes ────────────────────────────────────────────────────
    geometries = []
    labels     = []          # for console info

    # 1) GT
    if has_gt:
        gt_bp = data["gt_body_pose"][t]                         # (63,)
        gt_go = data["gt_global_orient"][t] if "gt_global_orient" in data else None
        gt_tr = data["gt_transl"][t]         if "gt_transl"        in data else None

        gt_mesh = build_smplx_mesh(
            smplx_model,
            body_pose_euler     = gt_bp,
            global_orient_euler = gt_go,
            transl              = gt_tr,
            color               = GT_COLOR,
            x_offset            = 0.0,
        )
        geometries.append(gt_mesh)
        labels.append("GT  @ x=0.0")
    else:
        print("  [WARN] No GT found in", os.path.basename(npz_path),
              " – only predictions will be shown.")

    # 2) Predictions
    for s in range(S):
        bp = pred_body_pose[s, t]                               # (63,)
        go = data["global_orient"][s, t] if has_orient else None
        tr = data["transl"][s, t]        if has_transl else None
        color   = PRED_COLORS[s % len(PRED_COLORS)]
        x_off   = X_SPACING * (s + 1)
        pred_mesh = build_smplx_mesh(
            smplx_model,
            body_pose_euler     = bp,
            global_orient_euler = go,
            transl              = tr,
            color               = color,
            x_offset            = x_off,
        )
        geometries.append(pred_mesh)
        labels.append(f"Pred {s} @ x={x_off:.1f}")

    # ── console info ──────────────────────────────────────────────────────
    idx_str  = os.path.splitext(os.path.basename(npz_path))[0]
    instr    = meta.get("instruction", "(no instruction)")
    ds_name  = meta.get("dataset_name", "?")
    ep_id    = meta.get("episode_id",   "?")
    print(f"\n{'─'*64}")
    print(f"  File       : {idx_str}.npz")
    print(f"  Dataset    : {ds_name}   episode={ep_id}   timestep t={t}/{T-1}")
    print(f"  Instruction: {instr[:120]}")
    for lbl in labels:
        print(f"               {lbl}")
    print(f"  (Close window or press Q to advance to next entry)")

    # ── add a coordinate frame for reference ──────────────────────────────
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
    coord.translate([0.0, -1.2, 0.0])
    geometries.append(coord)

    # ── draw ──────────────────────────────────────────────────────────────
    win_title = (f"[{idx_str}] {ds_name}  |  t={t}  |  "
                 f"{instr[:60]}{'…' if len(instr)>60 else ''}")
    o3d.visualization.draw_geometries(
        geometries,
        window_name   = win_title,
        width         = 1280,
        height        = 720,
        mesh_show_back_face = True,
    )
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualize body pose predictions (GT vs predicted) with Open3D.")
    parser.add_argument("--pred_dir",
                        default="./body_pose_preds",
                        help="Directory containing the .npz prediction files.")
    parser.add_argument("--smplx_model_dir",
                        default="/home/haziq/VITRA/data/smplx",
                        help="Directory containing SMPLX_NEUTRAL.npz (or .pkl).")
    parser.add_argument("--timestep", type=int, default=0,
                        help="Which timestep in the chunk sequence to visualize (default: 0).")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Start from this entry index (useful to resume).")
    parser.add_argument("--max_entries", type=int, default=None,
                        help="Stop after visualizing this many entries.")
    args = parser.parse_args()

    # ── collect files ───────────────────────────────────────────────────────
    npz_files = sorted(glob.glob(os.path.join(args.pred_dir, "*.npz")))
    # Exclude any file that is obviously named with a GT-only suffix
    npz_files = [f for f in npz_files if "_gt" not in os.path.basename(f)]

    if not npz_files:
        print(f"[ERROR] No .npz files found in {args.pred_dir}")
        sys.exit(1)

    npz_files = npz_files[args.start_idx:]
    if args.max_entries is not None:
        npz_files = npz_files[:args.max_entries]

    print(f"Found {len(npz_files)} prediction files (starting at idx {args.start_idx}).")
    print(f"Loading SMPLX neutral model from  {args.smplx_model_dir} …")

    smplx_model = load_smplx_model(args.smplx_model_dir)
    print("SMPLX model ready.\n")
    print("Legend:")
    print(f"  blue  = Ground truth")
    for i, c in enumerate(PRED_COLORS[:6]):
        print(f"  pred {i} = RGB{tuple(int(x*255) for x in c)}")
    print(f"\nEach mesh is separated by {X_SPACING:.1f} m along the x-axis.\n")

    # ── iterate ─────────────────────────────────────────────────────────────
    for npz_path in npz_files:
        meta_path = npz_path.replace(".npz", "_meta.json")
        meta = {}
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)

        try:
            visualize_entry(smplx_model, npz_path, meta, timestep=args.timestep)
        except Exception as exc:
            print(f"[ERR] Failed to visualize {npz_path}: {exc}")
            import traceback; traceback.print_exc()

    print("\nAll entries shown.")


if __name__ == "__main__":
    main()
