"""
live rerun viewer for EKF replay data.
called by replay.py --rerun, or run directly on a .npz or .csv file.
"""

from __future__ import annotations
import argparse
import sys
import os
import numpy as np

# ===========================================================================
# display toggles — set False to disable a panel group
# ===========================================================================
TRAJECTORY_DISPLAY      = True   # drone trajectory + pose + orientation axes
STATE_DISPLAY           = True   # state scalars (x, y, z, vx, vy, vz, d0–d2)
COV_DIAG_DISPLAY        = True   # covariance diagonal scalars
FEATURE_CLOUD_DISPLAY   = True   # 3-D world-frame feature cloud + bearing image
FEATURE_DEPTH_DISPLAY   = True   # per-feature world depth trace
INPUT_FEATURES_DISPLAY  = True   # raw image-plane px/py from CSV tracks
FEATURE_COV_DISPLAY     = True   # per-feature uncertainty (σ_ρ/ρ, σ_bx, σ_rho)


def _rotate_vector_by_quaternion(quaternion: tuple[float, float, float, float], vector: tuple[float, float, float]) -> np.ndarray:
    """rotate a 3D vector by a quaternion in (w, x, y, z) order."""
    w, x, y, z = quaternion
    vx, vy, vz = vector
    q_vec = np.array([x, y, z], dtype=np.float32)
    v = np.array([vx, vy, vz], dtype=np.float32)
    t = 2.0 * np.cross(q_vec, v)
    return v + w * t + np.cross(q_vec, t)


def run(data: dict, start_delay_s: float = 0.0) -> None:
    """
    stream replay data (output of replay.replay()) into a rerun recording.
    start_delay_s skips the first seconds of the replay before visualizing.
    """
    try:
        import rerun as rr
    except ImportError:
        print("rerun-sdk not found.  Install with:  pip install rerun-sdk")
        return

    rr.init("vio_replay", spawn=True)

    ts   = data["timestamps"]
    st   = data["states"]        # (N, 9)
    qs   = data["quaternions"]   # (N, 4)
    Pd   = data["cov_diagonals"] # (N, 9)
    feat = data["feature_snaps"]
    n    = len(ts)
    inp_feat = data.get("input_feature_snaps", [[] for _ in range(n)])
    cov_feat = data.get("feature_cov_snaps",   [[] for _ in range(n)])

    if start_delay_s and n > 0:
        cutoff_ms = float(ts[0]) + float(start_delay_s) * 1000.0
        start_idx = int(np.searchsorted(ts, cutoff_ms, side="left"))
        if start_idx >= n:
            print(f"Start delay {start_delay_s:.1f}s is beyond the end of the replay data.")
            return
        if start_idx > 0:
            print(f"Skipping first {start_idx} steps and starting at {float(ts[start_idx]) / 1000.0:.3f}s.")
            ts = ts[start_idx:]
            st = st[start_idx:]
            qs = qs[start_idx:]
            Pd = Pd[start_idx:]
            feat = feat[start_idx:]
            inp_feat = inp_feat[start_idx:]
            cov_feat = cov_feat[start_idx:]
            n = len(ts)

    state_keys = ("x", "y", "z", "vx", "vy", "vz", "d0", "d1", "d2")
    cov_keys   = ("P_x", "P_y", "P_z",
                  "P_vx", "P_vy", "P_vz",
                  "P_d0", "P_d1", "P_d2")

    feature_depths: dict[int, list] = {}   # fid -> [(t_s, Zw)]
    trajectory_pts: list = []               # growing list of [x,y,z] for line strip

    # log a static 320×320 black image once so rerun makes a 2D pixel-space panel.
    # everything logged under "camera/" shares this pixel coordinate space.
    rr.set_time_seconds("time", float(ts[0]) / 1000.0)
    rr.log("camera", rr.Image(np.zeros((320, 320, 3), dtype=np.uint8)))

    for i in range(n):
        t_s = float(ts[i]) / 1000.0
        rr.set_time_seconds("time", t_s)

        # ---- drone trajectory + pose + axes ----------------------------------
        if TRAJECTORY_DISPLAY:
            x, y, z = float(st[i, 0]), float(st[i, 1]), float(st[i, 2])
            rr.log("drone/trajectory", rr.Points3D([[x, y, z]], radii=0.01))

            trajectory_pts.append([x, y, z])
            if len(trajectory_pts) >= 2:
                traj_arr = np.array(trajectory_pts, dtype=np.float32)
                rr.log("drone/trajectory_line",
                       rr.LineStrips3D([traj_arr]))
                rr.log("drone/trajectory_xz",
                       rr.LineStrips2D([traj_arr[:, [0, 2]]]))
                rr.log("drone/trajectory_xy",
                       rr.LineStrips2D([traj_arr[:, [0, 1]]]))

            w, qx, qy, qz = (float(qs[i, k]) for k in range(4))
            quaternion = (w, qx, qy, qz)
            rr.log("drone/pose", rr.Transform3D(
                translation=[x, y, z],
                rotation=rr.Quaternion(xyzw=[qx, qy, qz, w]),
            ))

            axis_length = 0.1
            origins = np.tile([x, y, z], (3, 1)).astype(np.float32)
            axes = np.vstack([
                _rotate_vector_by_quaternion(quaternion, (axis_length, 0.0, 0.0)),
                _rotate_vector_by_quaternion(quaternion, (0.0, axis_length, 0.0)),
                _rotate_vector_by_quaternion(quaternion, (0.0, 0.0, axis_length)),
            ]).astype(np.float32)
            colors = np.array([
                [255, 0, 0, 255],
                [0, 255, 0, 255],
                [0, 0, 255, 255],
            ], dtype=np.uint8)
            rr.log("drone/pose_axes", rr.Arrows3D(
                vectors=axes,
                origins=origins,
                colors=colors,
                radii=0.01,
            ))

        # ---- state time series -----------------------------------------------
        if STATE_DISPLAY:
            for j, key in enumerate(state_keys):
                rr.log(f"state/{key}", rr.Scalars(float(st[i, j])))

        # ---- covariance diagonal ---------------------------------------------
        if COV_DIAG_DISPLAY:
            for j, key in enumerate(cov_keys):
                val = float(Pd[i, j])
                rr.log(f"covariance/{key}", rr.Scalars(val if val > 0 else 1e-9))

        # ---- feature cloud + bearing image -----------------------------------
        snap = feat[i]
        if FEATURE_CLOUD_DISPLAY and snap:
            pts = np.array([[f[4], f[5], f[6]] for f in snap], dtype=np.float32)
            ids = [f[0] for f in snap]
            rr.log("features/cloud",
                   rr.Points3D(pts, radii=0.02,
                                labels=[str(fid) for fid in ids]))

            pts_image = np.array([[float(f[1]), float(f[2])] for f in snap], dtype=np.float32)
            try:
                rr.log("features/image",
                       rr.Points2D(pts_image, radii=0.03,
                                   labels=[str(fid) for fid in ids]))
            except AttributeError:
                pts_image_3d = np.column_stack((pts_image, np.zeros((len(pts_image), 1), dtype=np.float32)))
                rr.log("features/image",
                       rr.Points3D(pts_image_3d, radii=0.03,
                                   labels=[str(fid) for fid in ids]))

        # ---- per-feature depth trace -----------------------------------------
        if FEATURE_DEPTH_DISPLAY and snap:
            for f in snap:
                fid = f[0]
                zw  = float(f[6])
                rr.log(f"features/depth_f{fid}", rr.Scalars(zw))
                feature_depths.setdefault(fid, []).append((t_s, zw))

        # ---- raw input pixel features from CSV --------------------------------
        if INPUT_FEATURES_DISPLAY and inp_feat[i]:
            raw = inp_feat[i]
            pts_raw = np.array([[float(r[1]), float(r[2])] for r in raw], dtype=np.float32)
            labels_raw = [str(r[0]) for r in raw]
            try:
                rr.log("features/raw_pixels",
                       rr.Points2D(pts_raw, radii=0.03, labels=labels_raw))
            except AttributeError:
                pts_raw_3d = np.column_stack((pts_raw, np.zeros((len(pts_raw), 1), dtype=np.float32)))
                rr.log("features/raw_pixels",
                       rr.Points3D(pts_raw_3d, radii=0.03, labels=labels_raw))
            # also log to camera/ so they overlay on the 2D pixel-space panel
            rr.log("camera/features",
                   rr.Points2D(pts_raw, radii=3.0, labels=labels_raw))

        # ---- per-feature covariance / uncertainty ----------------------------
        if FEATURE_COV_DISPLAY and cov_feat[i]:
            rel_depth_vals = []
            for entry in cov_feat[i]:
                fid, sigma_bx, sigma_by, sigma_rho, rel_depth_unc = entry
                rr.log(f"features/uncertainty/rel_depth/f{fid}",
                       rr.Scalars(float(rel_depth_unc)))
                rr.log(f"features/uncertainty/sigma_rho/f{fid}",
                       rr.Scalars(float(sigma_rho)))
                rr.log(f"features/uncertainty/sigma_bearing/f{fid}",
                       rr.Scalars(float((sigma_bx + sigma_by) * 0.5)))
                rel_depth_vals.append(rel_depth_unc)
            if rel_depth_vals:
                rr.log("features/uncertainty/median_rel_depth",
                       rr.Scalars(float(np.median(rel_depth_vals))))

    print(f"Logged {n} steps to rerun.  "
          f"Tracked {len(feature_depths)} unique features.")


# ---------------------------------------------------------------------------
# standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize.py <result.npz | rec_NNN.csv> [--start_delay_s N]")
        sys.exit(1)

    path = sys.argv[1]
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--start_delay_s", type=float, default=0.0,
                        help="skip the first N seconds of replay before visualization")
    parsed, _ = parser.parse_known_args(sys.argv[2:])

    if path.endswith(".npz"):
        npz  = np.load(path, allow_pickle=True)
        data = {k: npz[k] for k in npz.files}
        # feature_snaps and the ragged lists are not saved in .npz; use empty placeholders
        if "feature_snaps" not in data:
            n = len(data["timestamps"])
            data["feature_snaps"] = [[] for _ in range(n)]
        if "input_feature_snaps" not in data:
            n = len(data["timestamps"])
            data["input_feature_snaps"] = [[] for _ in range(n)]
        if "feature_cov_snaps" not in data:
            n = len(data["timestamps"])
            data["feature_cov_snaps"] = [[] for _ in range(n)]
        run(data, start_delay_s=parsed.start_delay_s)
    elif path.endswith(".csv"):
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import replay as _replay
        parser = _replay.build_parser()
        parser.add_argument("--start_delay_s", type=float, default=0.0,
                            help="skip the first N seconds of replay before visualization")
        args   = parser.parse_args([path] + sys.argv[2:])
        data   = _replay.replay(args)
        run(data, start_delay_s=args.start_delay_s)
    else:
        print("Unsupported file type (expected .npz or .csv)")
        sys.exit(1)
