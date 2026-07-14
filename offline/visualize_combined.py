"""
shows the EKF replay and OptiTrack GT together in one rerun session on a shared time axis.
reads a sync-offset YAML (from sync_times.py) to line up both clocks.
"""

from __future__ import annotations
import sys
import os
import argparse
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from helpers import (
    load_optitrack_ground_truth as _load_optitrack_gt,
    rotate_vector_by_quaternion,
    anchor_ground_truth,
    detect_liftoff_imu,
    load_drone_imu,
)
import replay as _replay


TRAJECTORY_DISPLAY      = True
POSITION_DISPLAY        = False   # px, py, pz state over time #check axis
VELOCITY_DISPLAY        = False   # world-frame velocities (drone + GT)
STATE_DISPLAY           = False
COV_DIAG_DISPLAY        = False
FEATURE_CLOUD_DISPLAY   = True
FEATURE_DEPTH_DISPLAY   = False
INPUT_FEATURES_DISPLAY  = True
FEATURE_COV_DISPLAY     = False
GT_DISPLAY              = True   # GT trajectory + pose axes



def load_offset_yaml(path: str) -> dict:
    """parse a sync_times.py offset YAML into a plain dict."""
    try:
        import yaml
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data or {}
    except ImportError:
        result: dict = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if ":" in line:
                    k, _, v = line.partition(":")
                    result[k.strip()] = v.strip()
        # cast known numeric keys
        for key in ("offset_s", "score", "start_gt_s", "end_gt_s", "flying_delay_s", "fallback_rho"):
            if key in result:
                try:
                    result[key] = float(result[key])
                except ValueError:
                    pass
        return result


# ---------------------------------------------------------------------------
# anchor helper — find matching drone + GT poses at a given time
# ---------------------------------------------------------------------------

def _find_anchor_poses(
    drone_data: dict,
    gt_data: dict,
    anchor_t_s: float,
) -> tuple | None:
    """return (drone_pos, drone_quat, gt_pos, gt_quat) at anchor_t_s, or None."""
    drone_ts = drone_data["timestamps"].astype(np.float64) / 1000.0
    gt_ts    = gt_data["timestamps"].astype(np.float64)

    di = int(np.argmin(np.abs(drone_ts - anchor_t_s)))
    gi = int(np.argmin(np.abs(gt_ts    - anchor_t_s)))

    if abs(drone_ts[di] - anchor_t_s) > 2.0:
        print(f"  WARNING: nearest drone sample is {abs(drone_ts[di]-anchor_t_s):.2f} s "
              "from anchor time — skipping anchoring.")
        return None
    if abs(gt_ts[gi] - anchor_t_s) > 2.0:
        print(f"  WARNING: nearest GT sample is {abs(gt_ts[gi]-anchor_t_s):.2f} s "
              "from anchor time — skipping anchoring.")
        return None

    st = drone_data["states"]
    qs = drone_data["quaternions"]
    drone_pos  = np.array([st[di, 0], st[di, 1], st[di, 2]], dtype=np.float32)
    w, qx, qy, qz = (float(qs[di, k]) for k in range(4))
    drone_quat = (w, qx, qy, qz)

    gt_pos  = np.array(gt_data["positions"][gi], dtype=np.float32)
    gw, gqx, gqy, gqz = (float(gt_data["quaternions"][gi, k]) for k in range(4))
    gt_quat = (gw, gqx, gqy, gqz)

    print(f"  Anchor at t={anchor_t_s:.2f} s  (drone idx={di}, gt idx={gi})")
    print(f"    Drone pos : {drone_pos}  quat_w={w:.3f}")
    print(f"    GT    pos : {gt_pos}  quat_w={gw:.3f}")
    return drone_pos, drone_quat, gt_pos, gt_quat


# ---------------------------------------------------------------------------
# rotation helper (local copy, same as visualize.py)
# ---------------------------------------------------------------------------

def _rotate_vec(q: tuple, v: tuple) -> np.ndarray:
    w, x, y, z = q
    vx, vy, vz = v
    qv = np.array([x, y, z], dtype=np.float32)
    vec = np.array([vx, vy, vz], dtype=np.float32)
    t = 2.0 * np.cross(qv, vec)
    return vec + w * t + np.cross(qv, t)


# ---------------------------------------------------------------------------
# combined visualisation
# ---------------------------------------------------------------------------

def run_combined(
    drone_data: dict,
    gt_data: dict,       # timestamps already shifted to drone clock
    offset_s: float,
) -> None:
    try:
        import rerun as rr
    except ImportError:
        print("rerun-sdk not found.  Install with:  pip install rerun-sdk")
        return

    rr.init("vio_compare", spawn=True)

    # ── drone data unpacking ─────────────────────────────────────────────────
    ts       = drone_data["timestamps"]
    st       = drone_data["states"]
    qs       = drone_data["quaternions"]
    Pd       = drone_data["cov_diagonals"]
    feat     = drone_data["feature_snaps"]
    n        = len(ts)
    inp_feat = drone_data.get("input_feature_snaps", [[] for _ in range(n)])
    cov_feat = drone_data.get("feature_cov_snaps",   [[] for _ in range(n)])

    state_keys = ("x", "y", "z", "vx", "vy", "vz", "d0", "d1", "d2")
    cov_keys   = ("P_x", "P_y", "P_z",
                  "P_vx", "P_vy", "P_vz",
                  "P_d0", "P_d1", "P_d2")

    feature_depths: dict[int, list] = {}
    drone_traj: list = []

    # ── GT data unpacking ──────────────────────────────────────────
    gt_ts   = gt_data["timestamps"]      # already offset-shifted seconds
    gt_pos  = gt_data["positions"]
    gt_qs   = gt_data["quaternions"]
    gt_mk_pos  = gt_data.get("marker_positions")
    gt_mk_nm   = gt_data.get("marker_names")
    gt_unlbl   = gt_data.get("unlabeled_marker_positions")
    gt_traj: list = []

    axis_len   = 0.1
    axis_color = np.array([[255, 0, 0, 255],
                            [0, 255, 0, 255],
                            [0, 0, 255, 255]], dtype=np.uint8)

    # static world-frame decorations logged once
    _log_world_decorations(rr, axis_color)

    # ── drone loop ───────────────────────────────────────────────────────────
    print("  Logging drone data …")
    for i in range(n):
        t_s = float(ts[i]) / 1000.0
        rr.set_time_seconds("time", t_s)

        if TRAJECTORY_DISPLAY:
            x, y, z = float(st[i, 0]), float(st[i, 1]), float(st[i, 2])
            rr.log("drone/trajectory", rr.Points3D([[x, y, z]], radii=0.01))

            drone_traj.append([x, y, z])
            if len(drone_traj) >= 2:
                arr = np.array(drone_traj, dtype=np.float32)
                rr.log("drone/trajectory_line", rr.LineStrips3D([arr]))
                rr.log("drone/trajectory_xz",   rr.LineStrips2D([arr[:, [0, 2]]]))
                rr.log("drone/trajectory_xy",   rr.LineStrips2D([arr[:, [0, 1]]]))

            w, qx, qy, qz = (float(qs[i, k]) for k in range(4))
            q = (w, qx, qy, qz)
            rr.log("drone/pose", rr.Transform3D(
                translation=[x, y, z],
                rotation=rr.Quaternion(xyzw=[qx, qy, qz, w]),
            ))
            origins = np.tile([x, y, z], (3, 1)).astype(np.float32)
            axes = np.vstack([
                _rotate_vec(q, (axis_len, 0.0, 0.0)),
                _rotate_vec(q, (0.0, axis_len, 0.0)),
                _rotate_vec(q, (0.0, 0.0, axis_len)),
            ]).astype(np.float32)
            rr.log("drone/pose_axes", rr.Arrows3D(
                vectors=axes, origins=origins, colors=axis_color, radii=0.01))

        if POSITION_DISPLAY:
            rr.log("state/px", rr.Scalars(float(st[i, 0])))
            rr.log("state/py", rr.Scalars(float(st[i, 1])))
            rr.log("state/pz", rr.Scalars(float(st[i, 2])))

        if VELOCITY_DISPLAY:
            # drone velocities are in body frame — transform to world frame
            vx_body, vy_body, vz_body = float(st[i, 3]), float(st[i, 4]), float(st[i, 5])
            w, qx, qy, qz = (float(qs[i, k]) for k in range(4))
            q = (w, qx, qy, qz)
            # rotate body velocity to world frame
            v_world = _rotate_vec(q, (vx_body, vy_body, vz_body))
            rr.log("velocity/drone_world_vx", rr.Scalars(float(v_world[0])))
            rr.log("velocity/drone_world_vy", rr.Scalars(float(v_world[1])))
            rr.log("velocity/drone_world_vz", rr.Scalars(float(v_world[2])))
            # also log body-frame velocities for comparison
            rr.log("velocity/drone_body_vx", rr.Scalars(vx_body))
            rr.log("velocity/drone_body_vy", rr.Scalars(vy_body))
            rr.log("velocity/drone_body_vz", rr.Scalars(vz_body))

        if STATE_DISPLAY:
            for j, key in enumerate(state_keys):
                rr.log(f"state/{key}", rr.Scalars(float(st[i, j])))

        if COV_DIAG_DISPLAY:
            for j, key in enumerate(cov_keys):
                val = float(Pd[i, j])
                rr.log(f"covariance/{key}", rr.Scalars(val if val > 0 else 1e-9))

        snap = feat[i]
        if FEATURE_CLOUD_DISPLAY and snap:
            pts = np.array([[f[4], f[5], f[6]] for f in snap], dtype=np.float32)
            ids = [f[0] for f in snap]
            rr.log("features/cloud",
                   rr.Points3D(pts, radii=0.02, labels=[str(fid) for fid in ids]))
            pts_img = np.array([[float(f[1]), float(f[2])] for f in snap], dtype=np.float32)
            try:
                rr.log("features/image",
                       rr.Points2D(pts_img, radii=0.03, labels=[str(fid) for fid in ids]))
            except AttributeError:
                rr.log("features/image",
                       rr.Points3D(
                           np.column_stack((pts_img, np.zeros((len(pts_img), 1), dtype=np.float32))),
                           radii=0.03, labels=[str(fid) for fid in ids]))

        if FEATURE_DEPTH_DISPLAY and snap:
            for f in snap:
                fid = f[0]
                rr.log(f"features/depth_f{fid}", rr.Scalars(float(f[6])))
                feature_depths.setdefault(fid, []).append((t_s, float(f[6])))

        if INPUT_FEATURES_DISPLAY and inp_feat[i]:
            raw = inp_feat[i]
            pts_raw = np.array([[float(r[1]), float(r[2])] for r in raw], dtype=np.float32)
            labels_raw = [str(r[0]) for r in raw]
            try:
                rr.log("features/raw_pixels",
                       rr.Points2D(pts_raw, radii=0.03, labels=labels_raw))
            except AttributeError:
                rr.log("features/raw_pixels",
                       rr.Points3D(
                           np.column_stack((pts_raw, np.zeros((len(pts_raw), 1), dtype=np.float32))),
                           radii=0.03, labels=labels_raw))

        if FEATURE_COV_DISPLAY and cov_feat[i]:
            rel_vals = []
            for entry in cov_feat[i]:
                fid, sigma_bx, sigma_by, sigma_rho, rel = entry
                rr.log(f"features/uncertainty/rel_depth/f{fid}", rr.Scalars(float(rel)))
                rr.log(f"features/uncertainty/sigma_rho/f{fid}", rr.Scalars(float(sigma_rho)))
                rr.log(f"features/uncertainty/sigma_bearing/f{fid}",
                       rr.Scalars(float((sigma_bx + sigma_by) * 0.5)))
                rel_vals.append(rel)
            if rel_vals:
                rr.log("features/uncertainty/median_rel_depth",
                       rr.Scalars(float(np.median(rel_vals))))

    # ── GT loop ────────────────────────────────────────────────────
    print("  Logging ground-truth data …")
    for i in range(len(gt_ts)):
        t_s = float(gt_ts[i])          # already on drone clock (offset applied)
        rr.set_time_seconds("time", t_s)

        if not GT_DISPLAY:
            continue

        x, y, z = gt_pos[i]
        w, qx, qy, qz = gt_qs[i]

        rr.log("ground_truth/trajectory", rr.Points3D([[x, y, z]], radii=0.01))

        gt_traj.append([x, y, z])
        if len(gt_traj) >= 2:
            arr = np.array(gt_traj, dtype=np.float32)
            rr.log("ground_truth/trajectory_line", rr.LineStrips3D([arr]))
            rr.log("ground_truth/trajectory_xz",   rr.LineStrips2D([arr[:, [0, 2]]]))
            rr.log("ground_truth/trajectory_xy",   rr.LineStrips2D([arr[:, [0, 1]]]))

        rr.log("ground_truth/pose", rr.Transform3D(
            translation=[x, y, z],
            rotation=rr.Quaternion(xyzw=[qx, qy, qz, w]),
        ))
        q = (float(w), float(qx), float(qy), float(qz))
        origins = np.tile([x, y, z], (3, 1)).astype(np.float32)
        axes = np.vstack([
            rotate_vector_by_quaternion(q, (axis_len, 0.0, 0.0)),
            rotate_vector_by_quaternion(q, (0.0, axis_len, 0.0)),
            rotate_vector_by_quaternion(q, (0.0, 0.0, axis_len)),
        ]).astype(np.float32)
        rr.log("ground_truth/pose_axes", rr.Arrows3D(
            vectors=axes, origins=origins, colors=axis_color, radii=0.01))

        # GT velocity from position differences (world frame)
        if VELOCITY_DISPLAY and i > 0:
            dt = float(gt_ts[i] - gt_ts[i-1])
            if dt > 1e-6:  # avoid division by zero
                dx = gt_pos[i][0] - gt_pos[i-1][0]
                dy = gt_pos[i][1] - gt_pos[i-1][1]
                dz = gt_pos[i][2] - gt_pos[i-1][2]
                gt_vx = dx / dt
                gt_vy = dy / dt
                gt_vz = dz / dt
                rr.log("velocity/gt_world_vx", rr.Scalars(float(gt_vx)))
                rr.log("velocity/gt_world_vy", rr.Scalars(float(gt_vy)))
                rr.log("velocity/gt_world_vz", rr.Scalars(float(gt_vz)))

        if gt_mk_pos is not None and gt_mk_pos.shape[0] > i:
            labels = [str(n) for n in gt_mk_nm] if gt_mk_nm is not None else None
            rr.log("ground_truth/markers",
                   rr.Points3D(gt_mk_pos[i], radii=0.02, labels=labels))

        if gt_unlbl is not None and len(gt_unlbl) > i and gt_unlbl[i]:
            up = np.array(gt_unlbl[i], dtype=np.float32)
            rr.log("ground_truth/unlabeled_markers",
                   rr.Points3D(up, radii=0.02,
                               colors=np.tile([[255, 0, 0, 255]], (len(up), 1))))

    print(f"Done.  Drone: {n} steps, {len(feature_depths)} features.  "
          f"GT: {len(gt_ts)} poses.  Offset applied: {offset_s:+.4f} s")


def _log_world_decorations(rr, axis_color: np.ndarray) -> None:
    world_axis_length = 0.2
    world_origins = np.tile(np.array([0.0, 0.0, 0.0], dtype=np.float32), (3, 1))
    world_axes = np.array([
        [world_axis_length, 0.0, 0.0],
        [0.0, world_axis_length, 0.0],
        [0.0, 0.0, world_axis_length],
    ], dtype=np.float32)
    rr.log("world/axes", rr.Arrows3D(
        vectors=world_axes, origins=world_origins, colors=axis_color, radii=0.01))

    plane_size, plane_steps = 0.5, 5
    pts = [[xi, yi, 0.0]
           for xi in np.linspace(-plane_size, plane_size, plane_steps)
           for yi in np.linspace(-plane_size, plane_size, plane_steps)]
    rr.log("world/ground_plane", rr.Points3D(
        np.array(pts, dtype=np.float32),
        radii=0.005,
        colors=np.tile(np.array([[200, 200, 200, 100]], dtype=np.uint8), (len(pts), 1)),
    ))


# ---------------------------------------------------------------------------
# trim helpers
# ---------------------------------------------------------------------------

def _trim_drone_data(drone_data: dict, cutoff_s: float) -> dict:
    """keep only drone data where timestamp (drone clock, seconds) <= cutoff_s."""
    ts = drone_data["timestamps"]
    mask = ts / 1000.0 <= cutoff_s
    result = dict(drone_data)
    for key in ("timestamps", "states", "quaternions", "cov_diagonals"):
        if key in drone_data:
            result[key] = drone_data[key][mask]
    for key in ("feature_snaps", "input_feature_snaps", "feature_cov_snaps"):
        if key in drone_data:
            orig = drone_data[key]
            result[key] = [orig[i] for i in range(len(orig)) if i < len(mask) and mask[i]]
    return result


def _trim_gt_data(gt_data: dict, cutoff_s: float) -> dict:
    """keep only GT data where timestamp (drone clock, seconds) <= cutoff_s."""
    ts = np.array(gt_data["timestamps"])
    mask = ts <= cutoff_s
    result = dict(gt_data)
    result["timestamps"] = ts[mask]
    result["positions"]  = gt_data["positions"][mask]
    result["quaternions"] = gt_data["quaternions"][mask]
    if gt_data.get("marker_positions") is not None:
        result["marker_positions"] = gt_data["marker_positions"][mask]
    if gt_data.get("unlabeled_marker_positions") is not None:
        orig = gt_data["unlabeled_marker_positions"]
        result["unlabeled_marker_positions"] = [
            orig[i] for i in range(len(orig)) if i < len(mask) and mask[i]
        ]
    return result


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Visualise EKF replay + OptiTrack ground truth on a shared time axis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--offset_yaml", type=str, default=None,
                   help="YAML file produced by sync_times.py  "
                        "(supplies drone_csv, optitrack_csv, offset_s)")
    p.add_argument("--drone_csv",     type=str, default=None,
                   help="Override drone CSV path from the YAML")
    p.add_argument("--optitrack_csv", type=str, default=None,
                   help="Override OptiTrack CSV path from the YAML")
    p.add_argument("--offset_s",  type=float, default=None,
                   help="Override offset_s from the YAML  "
                        "(t_ot_aligned = t_ot_raw + offset_s)")
    p.add_argument("--start_s", type=float, default=None,
                   help="Trim GT: discard rows with raw OT timestamp < start_s "
                        "(defaults to start_gt_s from the offset YAML if available)")
    p.add_argument("--end_s",   type=float, default=None,
                   help="Trim GT: discard rows with raw OT timestamp > end_s "
                        "(defaults to end_gt_s from the offset YAML if available)")
    p.add_argument("--anchor_delay_s", type=float, default=2.0,
                   help="Seconds after IMU-detected liftoff to use as the pose-alignment "
                        "anchor when zeroing the GT trajectory to the drone frame (default: 2.0)")
    p.add_argument("--no_anchor", action="store_true",
                   help="Skip GT pose anchoring — show raw aligned trajectories.")
    p.add_argument("--show_s", type=float, default=None,
                   help="Only visualise the first X seconds after liftoff "
                        "(defaults to show_s from the offset YAML if set; null/omitted = no trim)")
    p.add_argument("--config", type=str, default=None,
                   help="EKF config YAML for the replay (default: kalman_replay_config.yaml)")
    return p


def main() -> None:
    args = build_parser().parse_args()

    # ── resolve paths and offset from YAML ──────────────────────────────────
    yaml_cfg: dict = {}
    if args.offset_yaml:
        yaml_cfg = load_offset_yaml(args.offset_yaml)
        print(f"Loaded offset YAML: {args.offset_yaml}")

    drone_csv     = args.drone_csv     or yaml_cfg.get("drone_csv")
    optitrack_csv = args.optitrack_csv or yaml_cfg.get("optitrack_csv")
    offset_s      = args.offset_s      if args.offset_s is not None else float(yaml_cfg.get("offset_s", 0.0))
    start_s       = args.start_s       if args.start_s is not None else yaml_cfg.get("start_gt_s")
    end_s         = args.end_s         if args.end_s is not None else yaml_cfg.get("end_gt_s")
    _yaml_show    = yaml_cfg.get("show_s")
    show_s        = args.show_s if args.show_s is not None else (float(_yaml_show) if _yaml_show is not None else None)

    def _parse_rotation(key: str, default: np.ndarray) -> np.ndarray:
        """parse a 3x3 rotation matrix from yaml_cfg[key]; return default on missing/error."""
        raw = yaml_cfg.get(key)
        if raw is None:
            return default
        try:
            arr = np.array(raw, dtype=np.float32)
            if arr.shape == (9,):
                arr = arr.reshape(3, 3)
            if arr.shape == (3, 3):
                print(f"  {key} from YAML: {arr.tolist()}")
                return arr
            print(f"  WARNING: {key} has unexpected shape {arr.shape}, using default.")
        except Exception as e:
            print(f"  WARNING: could not parse {key} from YAML: {e}")
        return default

    _identity = np.eye(3, dtype=np.float32)
    ot_rotation   = _parse_rotation("optitrack_rotation", _identity)
    body_rotation = _parse_rotation("gt_body_rotation",   None)  # type: ignore[arg-type]

    if not drone_csv:
        print("ERROR: drone CSV not specified.  Use --drone_csv or --offset_yaml.")
        sys.exit(1)
    if not optitrack_csv:
        print("ERROR: OptiTrack CSV not specified.  Use --optitrack_csv or --offset_yaml.")
        sys.exit(1)

    # resolve relative paths from the script's working directory
    if not os.path.isabs(drone_csv):
        drone_csv = os.path.join(os.getcwd(), drone_csv)
    if not os.path.isabs(optitrack_csv):
        optitrack_csv = os.path.join(os.getcwd(), optitrack_csv)

    print(f"Drone CSV     : {drone_csv}")
    print(f"OptiTrack CSV : {optitrack_csv}")
    print(f"Offset        : {offset_s:+.4f} s")

    # ── load drone data ──────────────────────────────────────────────────────
    print("\nRunning EKF replay …")
    replay_parser = _replay.build_parser()
    replay_args   = replay_parser.parse_args([drone_csv])
    if args.config is not None:
        replay_args.config = args.config
        print(f"  EKF config: {args.config}")
    if "flying_delay_s" in yaml_cfg:
        replay_args.flying_delay_s = float(yaml_cfg["flying_delay_s"])
        print(f"  flying_delay_s overridden from offset YAML: {replay_args.flying_delay_s} s")
    if "fallback_rho" in yaml_cfg:
        replay_args.fallback_rho = float(yaml_cfg["fallback_rho"])
        print(f"  fallback_rho overridden from offset YAML: {replay_args.fallback_rho}")
    drone_data    = _replay.replay(replay_args)

    # ── load GT (time_offset shifts OT clock to drone clock) ───────
    print("Loading ground truth …")
    gt_data = _load_optitrack_gt(
        optitrack_csv,
        ot_rotation,
        time_offset=offset_s,
        cut_before=start_s,
        cut_after=end_s,
        body_rotation=body_rotation if isinstance(body_rotation, np.ndarray) else None,
    )
    print(f"  GT: {len(gt_data['timestamps'])} poses  "
          f"[{gt_data['timestamps'][0]:.1f} – {gt_data['timestamps'][-1]:.1f} s aligned]")

    # ── liftoff detection (needed for anchoring and/or show_s trim) ────────
    liftoff_imu_s = None
    need_liftoff = (not args.no_anchor) or (show_s is not None)
    if need_liftoff:
        print("\nFinding liftoff …")
        imu_data = load_drone_imu(drone_csv)
        liftoff_imu_s = detect_liftoff_imu(imu_data["t_s"], imu_data["accel_mag"])
        if liftoff_imu_s is None:
            liftoff_imu_s = float(drone_data["timestamps"][0]) / 1000.0
            print(f"  IMU liftoff not detected — falling back to recording start "
                  f"({liftoff_imu_s:.2f} s)")
        else:
            print(f"  IMU liftoff detected at {liftoff_imu_s:.3f} s (drone clock)")

    # ── trim to show_s seconds after liftoff ────────────────────────────────
    if show_s is not None and liftoff_imu_s is not None:
        cutoff_s = liftoff_imu_s + show_s
        print(f"  Trimming to first {show_s:.1f} s after liftoff (cutoff: {cutoff_s:.2f} s) …")
        drone_data = _trim_drone_data(drone_data, cutoff_s)
        gt_data    = _trim_gt_data(gt_data, cutoff_s)
        print(f"  After trim — drone: {len(drone_data['timestamps'])} steps, "
              f"GT: {len(gt_data['timestamps'])} poses")

    # ── pose alignment: GT → drone frame ────────────────────────────────────
    # default: 4-DOF yaw+translation Umeyama; single-instant anchor only when the
    # offset YAML carries a value for anchor_s
    _anchor_s = yaml_cfg.get("anchor_s", None)
    _use_4dof = (_anchor_s is None) or (
        isinstance(_anchor_s, str) and _anchor_s.strip().lower() == "4dof")
    if args.no_anchor:
        print("\nPose anchoring disabled (--no_anchor).")
    elif _use_4dof:
        from helpers import anchor_ground_truth_4dof
        _dt = drone_data["timestamps"].astype(np.float64) / 1000.0
        _dp = np.asarray(drone_data["states"])[:, :3]
        _nz = ~((_dp[:, 0] == 0) & (_dp[:, 1] == 0) & (_dp[:, 2] == 0))
        gt_data = anchor_ground_truth_4dof(gt_data, _dt[_nz], _dp[_nz])
        print("  GT trajectory 4-DOF (yaw+translation) aligned to drone frame.")
    else:
        anchor_delay_s = float(_anchor_s) if isinstance(_anchor_s, (int, float)) \
            else (float(yaml_cfg["anchor_delay_s"]) if "anchor_delay_s" in yaml_cfg
                  else args.anchor_delay_s)
        anchor_t_s = liftoff_imu_s + anchor_delay_s
        print(f"  Anchor time : {anchor_t_s:.3f} s  (liftoff + {anchor_delay_s:.1f} s, legacy single-instant)")
        result = _find_anchor_poses(drone_data, gt_data, anchor_t_s)
        if result is not None:
            drone_pos, drone_quat, gt_pos, gt_quat = result
            gt_data = anchor_ground_truth(gt_data, drone_pos, drone_quat, gt_pos, gt_quat)
            print("  GT trajectory anchored to drone frame.")
        else:
            print("  Anchoring skipped — GT displayed in its own aligned frame.")

    # ── combined rerun visualisation ─────────────────────────────────────────
    print("\nStarting rerun …")
    run_combined(drone_data, gt_data, offset_s)


if __name__ == "__main__":
    main()
