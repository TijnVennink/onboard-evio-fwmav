#!/usr/bin/env python3
"""
replay.py — offline EKF replay from a CSV recording, for sweeping/tuning.

Usage:
    python replay.py rec_005.csv [options]

    --flying              Enable in-flight predict path (default: off)
    --acc_xy FLOAT        procNoiseAcc_xy (default: 0.5)
    --acc_z  FLOAT        procNoiseAcc_z  (default: 1.0)
    --gyro_rp FLOAT       measNoiseGyro_rollpitch (default: 0.1)
    --gyro_yaw FLOAT      measNoiseGyro_yaw       (default: 0.1)
    --att_rev FLOAT       attitude_reversion       (default: 0.001)
    --meas_noise FLOAT    feature measurement noise variance (default: 0.01)
    --rho_init FLOAT      initial inverse-depth for new features (default: 0.5)
    --min_updates INT     only add features with n_updates >= this (default: 1)
    --lost_frames INT     consecutive missing cycles before feature removal (default: 10)
    --save PATH           save collected data to .npz file
    --rerun               stream to rerun.io viewer (requires: pip install rerun-sdk)
    --plot                show static matplotlib plots after replay

the script mirrors the on-board logic for sweeping:
  1. predict()           — IMU propagation
  2. feature_update()    — for each active track
  3. add_process_noise() — with actual elapsed dt
  4. finalize()          — fold attitude error, rebuild R
"""

import argparse
import sys
import os
import math

import numpy as np
import pandas as pd

try:
    import yaml
except ImportError:
    yaml = None

# allow running from openmv_offline/ directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kalman_core_py as kc
from helpers import detect_liftoff_imu as _detect_liftoff_imu

# ---------------------------------------------------------------------------
# EKF parameters (defaults mirror methodA.py)
# ---------------------------------------------------------------------------
CAM_FX, CAM_FY = 184.46, 182.85
CAM_CX, CAM_CY = 160.37, 153.25
CAM_K1, CAM_K2 = 0.083, -0.171
CAM_P1, CAM_P2 = -0.0017, -0.0024
CAM_K3          = 0.201
UPDATE_HZ       = 500.0

DEFAULT_PROCESS_NOISE = {
    "acc_xy": 0.5,
    "acc_z": 1.0,
    "vel": 0.0,
    "pos": 0.0,
    "att": 0.0,
    "gyro_rp": 0.1,
    "gyro_yaw": 0.1,
    "att_reversion": 0.001,
}
DEFAULT_FEATURE_MEAS_NOISE = 0.01
DEFAULT_FALLBACK_RHO = 0.33
DEFAULT_INIT_STDDEV_IDEPTH = 0.001
DEFAULT_MAX_DEPTH_UNCERTAINTY_RATIO = 0.5
DEFAULT_INNOVATION_GATE    = 0.0   # disabled; set to e.g. 5.991 (chi²(2,0.95)) to enable
DEFAULT_GATE_WARMUP_S      = 5.0   # seconds after phase-2 start with gate disabled
DEFAULT_FLOW_MEAS_VAR      = 0.05  # NOT USED event-flow (vx,vy) meas variance; not in paper (only used when flow_update on)
DEFAULT_AGE_TRUST_POW      = 0.0   # USED  per-feature R weight: meas_var *= (n_ref/n_updates)^pow; 0=off (uniform R)
DEFAULT_AGE_TRUST_FLOOR    = 0.1   # USED  min R-scale factor (cap max trust on very long-lived tracks)
DEFAULT_VEL_MAX            = 0   # USED  physical velocity clamp [m/s] in predict; 0=disabled
DEFAULT_DEPTH_TYPE         = "inverse"  # feature depth parametrization; only inverse in paper (log/hyperbolic WIP)
# initial covariance std-devs at phase-2 reset
DEFAULT_INIT_STDDEV_POS_XY  = 0.1    # [m]
DEFAULT_INIT_STDDEV_POS_Z   = 0.1    # [m]
DEFAULT_INIT_STDDEV_VEL     = 0.2    # [m/s]
DEFAULT_INIT_STDDEV_ATT_RP  = 0.05   # [rad] ~2.9°
DEFAULT_INIT_STDDEV_ATT_YAW = 0.05   # [rad] ~2.9°
DEFAULT_GYRO_LPF_CUTOFF_HZ = 3.0   # gyro low-pass cutoff [Hz] (honest Nyquist); ~3Hz optimal on this FWMAV; 0 = disabled
DEFAULT_ACC_LPF_CUTOFF_HZ  = 0.0   # acc_z low-pass cutoff [Hz]; damps wing-beat lift in v_z; 0 = disabled (regression-exact)
DEFAULT_RHO_INIT = 2.0
DEFAULT_MIN_UPDATES = 5
DEFAULT_LOST_FRAMES = 10
DEFAULT_IMU_TO_BODY_ROTATION = [1, 0, 0, 0, 1, 0, 0, 0, 1]
DEFAULT_FLYING = True
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "kalman_replay_config.yaml")
# above values are not tuned accordingly

# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------

def load_csv(path: str) -> pd.DataFrame:
    """load the interleaved I/T CSV produced by record_features_imu.py."""
    df = pd.read_csv(path, comment="#")
    # column names: type, t_ms, f1, f2, f3, f4, f5, f6
    df["t_ms"] = pd.to_numeric(df["t_ms"], errors="coerce")
    df["f1"]   = pd.to_numeric(df["f1"],   errors="coerce")
    df["f2"]   = pd.to_numeric(df["f2"],   errors="coerce")
    df["f3"]   = pd.to_numeric(df["f3"],   errors="coerce")
    df["f4"]   = pd.to_numeric(df["f4"],   errors="coerce")
    df["f5"]   = pd.to_numeric(df["f5"],   errors="coerce")
    df["f6"]   = pd.to_numeric(df["f6"],   errors="coerce")
    return df


def _parse_config_value(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    text = str(value).strip()
    lowered = text.lower()
    if lowered in ("true", "yes", "on"):  # pragma: no cover
        return True
    if lowered in ("false", "no", "off"):  # pragma: no cover
        return False
    if text.startswith("[") and text.endswith("]"):
        items = [item.strip() for item in text[1:-1].split(",") if item.strip()]
        return [_parse_config_value(item) for item in items]
    if "," in text:
        items = [item.strip() for item in text.split(",") if item.strip()]
        return [_parse_config_value(item) for item in items]
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    return text


def _set_nested_config_value(config, key, value):
    parts = key.replace(" ", "").split(".")
    target = config
    for part in parts[:-1]:
        if part not in target or not isinstance(target[part], dict):
            target[part] = {}
        target = target[part]
    target[parts[-1]] = value


def _load_text_config(path: str) -> dict:
    config = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            if ":" in line:
                key, value = line.split(":", 1)
            elif "=" in line:
                key, value = line.split("=", 1)
            else:
                continue
            _set_nested_config_value(config, key.strip(), _parse_config_value(value.strip()))
    return config


def _load_config(path: str) -> dict:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".yaml", ".yml"):
        if yaml is None:
            raise RuntimeError("PyYAML is required to load YAML config files")
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cfg if cfg is not None else {}
    if ext in (".txt", ".cfg", ".ini"):
        return _load_text_config(path)
    if yaml is not None:
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if isinstance(cfg, dict):
            return cfg
    return _load_text_config(path)


def _get_nested_setting(settings: dict, path: str, default):
    if not isinstance(settings, dict):
        return default
    node = settings
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def _get_setting(settings: dict, arg_value, path: str, default):
    if arg_value is not None:
        return arg_value
    return _get_nested_setting(settings, path, default)


def _get_default_config_path() -> str | None:
    if os.path.isfile(DEFAULT_CONFIG_PATH):
        return DEFAULT_CONFIG_PATH
    return None


def _apply_config_settings(settings: dict, args):
    update_rate = _get_setting(settings, args.update_rate, "update_rate", UPDATE_HZ)
    kc.set_update_rate(update_rate)

    imu_to_body_rotation = _get_nested_setting(settings, "imu_to_body_rotation", DEFAULT_IMU_TO_BODY_ROTATION)
    if isinstance(imu_to_body_rotation, (list, tuple)) and len(imu_to_body_rotation) == 9:
        kc.set_imu_to_body_rotation(*[float(v) for v in imu_to_body_rotation])

    # camera→body rotation (fixed extrinsic); accept a 3×3 nested matrix or flat 9-list. absent/identity ⇒ optical axis ≡ body z.
    cam_to_body = _get_nested_setting(settings, "camera_to_body_rotation", None)
    if cam_to_body is not None:
        flat = []
        for row in cam_to_body:
            if isinstance(row, (list, tuple)):
                flat.extend(float(v) for v in row)
            else:
                flat.append(float(row))
        if len(flat) == 9:
            kc.set_camera_to_body(*flat)

    cam = _get_nested_setting(settings, "camera_intrinsics", None)
    if cam is not None:
        if isinstance(cam, dict):
            fx = _get_nested_setting(cam, "fx", CAM_FX)
            fy = _get_nested_setting(cam, "fy", CAM_FY)
            cx = _get_nested_setting(cam, "cx", CAM_CX)
            cy = _get_nested_setting(cam, "cy", CAM_CY)
            k1 = _get_nested_setting(cam, "k1", CAM_K1)
            k2 = _get_nested_setting(cam, "k2", CAM_K2)
            p1 = _get_nested_setting(cam, "p1", CAM_P1)
            p2 = _get_nested_setting(cam, "p2", CAM_P2)
            k3 = _get_nested_setting(cam, "k3", CAM_K3)
            kc.set_camera_intrinsics(fx, fy, cx, cy, k1, k2, p1, p2, k3)
        elif isinstance(cam, (list, tuple)):
            if len(cam) == 4:
                kc.set_camera_intrinsics(*[float(v) for v in cam])
            elif len(cam) == 8:
                kc.set_camera_intrinsics(*[float(v) for v in cam], 0.0)
            elif len(cam) == 9:
                kc.set_camera_intrinsics(*[float(v) for v in cam])

    process_noise_values = {
        "acc_xy": _get_setting(settings, args.acc_xy, "process_noise.acc_xy", DEFAULT_PROCESS_NOISE["acc_xy"]),
        "acc_z": _get_setting(settings, args.acc_z, "process_noise.acc_z", DEFAULT_PROCESS_NOISE["acc_z"]),
        "vel": _get_setting(settings, args.vel, "process_noise.vel", DEFAULT_PROCESS_NOISE["vel"]),
        "pos": _get_setting(settings, args.pos, "process_noise.pos", DEFAULT_PROCESS_NOISE["pos"]),
        "att": _get_setting(settings, args.att, "process_noise.att", DEFAULT_PROCESS_NOISE["att"]),
        "gyro_rp": _get_setting(settings, args.gyro_rp, "process_noise.gyro_rp", DEFAULT_PROCESS_NOISE["gyro_rp"]),
        "gyro_yaw": _get_setting(settings, args.gyro_yaw, "process_noise.gyro_yaw", DEFAULT_PROCESS_NOISE["gyro_yaw"]),
        "att_reversion": _get_setting(settings, args.att_rev, "process_noise.att_reversion", DEFAULT_PROCESS_NOISE["att_reversion"]),
    }
    kc.set_process_noise(
        process_noise_values["acc_xy"],
        process_noise_values["acc_z"],
        process_noise_values["vel"],
        process_noise_values["pos"],
        process_noise_values["att"],
        process_noise_values["gyro_rp"],
        process_noise_values["gyro_yaw"],
        process_noise_values["att_reversion"],
    )

    flying = _get_setting(settings, args.flying, "flying", DEFAULT_FLYING)
    kc.set_flying(bool(flying))

    feature_meas_noise = _get_setting(settings, args.meas_noise, "feature_meas_noise", DEFAULT_FEATURE_MEAS_NOISE)
    kc.set_feature_meas_noise(float(feature_meas_noise))

    fallback_rho = _get_setting(settings, getattr(args, "fallback_rho", None), "fallback_rho", DEFAULT_FALLBACK_RHO)
    kc.set_fallback_rho(float(fallback_rho))

    init_stddev_idepth = _get_setting(settings, getattr(args, "init_stddev_idepth", None), "init_stddev_idepth", DEFAULT_INIT_STDDEV_IDEPTH)
    kc.set_init_stddev_idepth(float(init_stddev_idepth))

    max_depth_uncertainty_ratio = _get_setting(settings, getattr(args, "max_depth_uncertainty_ratio", None), "max_depth_uncertainty_ratio", DEFAULT_MAX_DEPTH_UNCERTAINTY_RATIO)
    kc.set_max_depth_uncertainty_ratio(float(max_depth_uncertainty_ratio))

    innovation_gate = _get_setting(settings, getattr(args, "innovation_gate", None), "innovation_gate", DEFAULT_INNOVATION_GATE)
    kc.set_innovation_gate(float(innovation_gate))

    vel_max = _get_setting(settings, getattr(args, "vel_max", None), "vel_max", DEFAULT_VEL_MAX)
    kc.set_vel_max(float(vel_max))

    depth_type = _get_setting(settings, getattr(args, "depth_type", None), "depth_type", DEFAULT_DEPTH_TYPE)
    kc.set_depth_type(depth_type)

    # gyro-bias estimation; not in paper (0/0 = disabled, reproduces 9-state behaviour)
    kc.set_init_stddev_gyro_bias(float(_get_setting(settings, getattr(args, "init_stddev_gyro_bias", None), "init_stddev_gyro_bias", 0.0)))
    kc.set_gyro_bias_noise(float(_get_setting(settings, getattr(args, "gyro_bias_noise", None), "gyro_bias_noise", 0.0)))

    # robocentric feature model; not in paper (default off = world-anchored)
    kc.set_robocentric(bool(_get_setting(settings, getattr(args, "robocentric", None), "robocentric", False)))
    # P_rf seeding; not in paper (shelved, default off, opt-in via `prf_seed: true`)
    kc.set_prf_seed(bool(_get_setting(settings, getattr(args, "prf_seed", None), "prf_seed", False)))
    # First-Estimates Jacobians (FEJ); not in paper (default off, opt-in via `fej: true`)
    kc.set_fej(bool(_get_setting(settings, getattr(args, "fej", None), "fej", False)))
    # event-flow velocity update; not in paper (default off, opt-in via `flow_update: true`).
    _flow_on = bool(_get_setting(settings, getattr(args, "flow_update", None), "flow_update", False))
    kc.set_flow_update(_flow_on)
    # no-thrust vertical model (default off, opt-in via `no_accel_thrust: true`). thrust assumed = gravity, so a_z does not drive v_z.
    _no_thrust = bool(_get_setting(settings, getattr(args, "no_accel_thrust", None), "no_accel_thrust", False))
    kc.set_no_accel_thrust(_no_thrust)
    # Huber robust feature down-weighting; not in paper (default 0=off, opt-in via `huber_delta`)
    kc.set_huber_delta(float(_get_setting(settings, getattr(args, "huber_delta", None), "huber_delta", 0.0)))
    kc.set_feat_process_noise(
        float(_get_setting(settings, getattr(args, "feat_bearing_noise", None), "feat_bearing_noise", 0.0)),
        float(_get_setting(settings, getattr(args, "feat_depth_noise", None), "feat_depth_noise", 0.0)))

    # initial covariance std-devs, set before kca_reset so they are stored in the struct. kca_reset reads them when rebuilding P.
    _init_pos_xy  = float(_get_setting(settings, getattr(args, "init_stddev_pos_xy",  None), "init_stddev_pos_xy",  DEFAULT_INIT_STDDEV_POS_XY))
    _init_pos_z   = float(_get_setting(settings, getattr(args, "init_stddev_pos_z",   None), "init_stddev_pos_z",   DEFAULT_INIT_STDDEV_POS_Z))
    _init_vel     = float(_get_setting(settings, getattr(args, "init_stddev_vel",     None), "init_stddev_vel",     DEFAULT_INIT_STDDEV_VEL))
    _init_att_rp  = float(_get_setting(settings, getattr(args, "init_stddev_att_rp",  None), "init_stddev_att_rp",  DEFAULT_INIT_STDDEV_ATT_RP))
    _init_att_yaw = float(_get_setting(settings, getattr(args, "init_stddev_att_yaw", None), "init_stddev_att_yaw", DEFAULT_INIT_STDDEV_ATT_YAW))
    kc.set_init_stddev(_init_pos_xy, _init_pos_z, _init_vel, _init_att_rp, _init_att_yaw)

    return {
        "rho_init": _get_setting(settings, args.rho_init, "rho_init", DEFAULT_RHO_INIT),
        "min_updates": _get_setting(settings, args.min_updates, "min_updates", DEFAULT_MIN_UPDATES),
        "lost_frames": _get_setting(settings, args.lost_frames, "lost_frames", DEFAULT_LOST_FRAMES),
        "flying_delay_s": _get_setting(settings, getattr(args, "flying_delay_s", None), "flying_delay_s", 0.0),
        "innovation_gate":    _get_setting(settings, getattr(args, "innovation_gate", None),    "innovation_gate",    DEFAULT_INNOVATION_GATE),
        "gate_warmup_s":      _get_setting(settings, getattr(args, "gate_warmup_s", None),      "gate_warmup_s",      DEFAULT_GATE_WARMUP_S),
        # 0 when flow off -> _manage_tracks skips the call entirely (regression-exact)
        "flow_meas_var":      (_get_setting(settings, getattr(args, "flow_meas_var", None), "flow_meas_var", DEFAULT_FLOW_MEAS_VAR) if _flow_on else 0.0),
        # per-feature age trust: meas_var *= (n_ref/n_updates)^pow, capped at floor. pow=0 -> off (regression-exact)
        "feature_meas_noise": float(feature_meas_noise),
        "age_trust_pow":      float(_get_setting(settings, getattr(args, "age_trust_pow", None),   "age_trust_pow",   DEFAULT_AGE_TRUST_POW)),
        "age_trust_floor":    float(_get_setting(settings, getattr(args, "age_trust_floor", None), "age_trust_floor", DEFAULT_AGE_TRUST_FLOOR)),
    }


def group_cycles(df: pd.DataFrame):
    """iterate over (i_row, [t_rows]) pairs in order. T-rows are grouped under the I-row that precedes them."""
    current_i = None
    current_t = []
    for row in df.itertuples(index=False):
        if row.type == "I":
            if current_i is not None:
                yield current_i, current_t
            current_i = row
            current_t = []
        elif row.type == "T":
            current_t.append(row)
    if current_i is not None:
        yield current_i, current_t


# ---------------------------------------------------------------------------
# feature track management — mirrors manage_feature_tracks() in methodA.py
# ---------------------------------------------------------------------------

def _manage_tracks(tracks, track_to_feature, track_missing,
                   min_updates: int, rho_init: float, lost_frames: int,
                   t_ms: float = 0.0, nis_out: list | None = None,
                   flow_meas_var: float = 0.0, gyro_xyz=(0.0, 0.0, 0.0),
                   base_meas_noise: float = 0.0, age_pow: float = 0.0, age_floor: float = 0.1) -> int:
    """returns count of observations gated by the Mahalanobis gate. if nis_out is given, appends (t_ms, d2) NIS per valid feature update for consistency plots."""
    active_ids = set()
    n_gated = 0

    gx, gy, gz = gyro_xyz
    for (track_id, x, y, _vx, _vy, n_updates) in tracks:
        active_ids.add(track_id)
        track_missing.pop(track_id, None)

        if n_updates < min_updates:
            continue  # skip not-yet-confirmed tracks

        if track_id not in track_to_feature:
            try:
                fid = kc.add_feature(x, y, rho_init)
                track_to_feature[track_id] = fid
            except Exception:
                continue

        fid = track_to_feature[track_id]
        # per-feature age trust: long-lived tracks (high n_updates) are cleaner
        # detections -> smaller R (more trust). 0.0 -> use global R (regression-exact).
        meas_var_eff = 0.0
        if age_pow > 0.0 and base_meas_noise > 0.0 and n_updates > min_updates:
            f = (float(min_updates) / float(n_updates)) ** age_pow
            if f < age_floor: f = age_floor
            meas_var_eff = base_meas_noise * f
        try:
            _rx, _ry, gated, d2 = kc.feature_update(fid, x, y, meas_var_eff)
            if gated:
                n_gated += 1
            if nis_out is not None and d2 >= 0.0:
                nis_out.append((t_ms, d2))
            # event-flow velocity update; not in paper (no-op unless flow_update on)
            if flow_meas_var > 0.0:
                kc.flow_update(fid, _vx, _vy, gx, gy, gz, flow_meas_var)
        except Exception:
            track_to_feature.pop(track_id, None)
            track_missing.pop(track_id, None)

    # stale track cleanup
    for tid in list(track_to_feature.keys()):
        if tid not in active_ids:
            track_missing[tid] = track_missing.get(tid, 0) + 1
            if track_missing[tid] >= lost_frames:
                fid = track_to_feature.pop(tid)
                track_missing.pop(tid, None)
                try:
                    kc.remove_feature(fid)
                except Exception:
                    pass

    return n_gated


# ---------------------------------------------------------------------------
# Main replay loop
# ---------------------------------------------------------------------------

def replay(args) -> dict:
    print(f"Loading {args.csv} …")
    df = load_csv(args.csv)

    n_i = (df["type"] == "I").sum()
    n_t = (df["type"] == "T").sum()
    print(f"  {n_i} IMU rows, {n_t} track rows")

    # ---- EKF setup --------------------------------------------------------
    kc.init()
    config_path = args.config if args.config else _get_default_config_path()
    settings = _load_config(config_path) if config_path else {}
    if config_path is not None:
        print(f"Using config: {config_path}")
    cfg = _apply_config_settings(settings, args)
    # recording already stores body-frame data → identity rotation
    if _get_nested_setting(settings, "imu_to_body_rotation", None) is None:
        kc.set_imu_to_body_rotation(1, 0, 0,  0, 1, 0,  0, 0, 1)
    if _get_nested_setting(settings, "camera_to_body_rotation", None) is None:
        kc.set_camera_to_body(1, 0, 0,  0, 1, 0,  0, 0, 1)
    if _get_nested_setting(settings, "camera_intrinsics", None) is None:
        kc.set_camera_intrinsics(CAM_FX, CAM_FY, CAM_CX, CAM_CY,
                                  CAM_K1, CAM_K2, CAM_P1, CAM_P2, CAM_K3)

    # raw gyro snapshot before LPF, for the static gyro-bias stillness check; not in paper.
    _imu_rows = df["type"] == "I"
    _gyro_raw_t = df.loc[_imu_rows, "t_ms"].values.astype(float)
    _gyro_raw   = df.loc[_imu_rows, ["f4", "f5", "f6"]].values.astype(float)

    # ---- gyro low-pass filter (suppress wing-beat vibrations) ---------------
    # Nyquist is derived from the actual IMU sample rate (median dt of the I-rows),
    _gyro_lpf_cutoff = float(_get_setting(settings, getattr(args, "gyro_lpf_cutoff_hz", None), "gyro_lpf_cutoff_hz", DEFAULT_GYRO_LPF_CUTOFF_HZ))
    if _gyro_lpf_cutoff > 0.0:
        try:
            from scipy.signal import butter, filtfilt as _filtfilt
            _imu_idx = df.index[df["type"] == "I"]
            _imu_t_ms = df.loc[_imu_idx, "t_ms"].values.astype(float)
            _dt_med_s = float(np.median(np.diff(_imu_t_ms))) / 1000.0 if len(_imu_t_ms) > 1 else (1.0 / UPDATE_HZ)
            _fs = 1.0 / _dt_med_s if _dt_med_s > 0 else UPDATE_HZ  # actual IMU rate [Hz]
            _nyq = _fs / 2.0
            _wn = _gyro_lpf_cutoff / _nyq
            if 0.0 < _wn < 1.0 and len(_imu_idx) > 12:
                _b, _a = butter(4, _wn, btype="low")
                for _col in ("f4", "f5", "f6"):
                    _raw = df.loc[_imu_idx, _col].values.astype(float)
                    df.loc[_imu_idx, _col] = _filtfilt(_b, _a, _raw)
                print(f"  Gyro LPF: {_gyro_lpf_cutoff:.1f} Hz Butterworth (4th order, zero-phase) "
                      f"[IMU fs={_fs:.1f} Hz, Nyquist={_nyq:.1f} Hz, Wn={_wn:.3f}]")
            elif _wn >= 1.0:
                print(f"  Gyro LPF: cutoff {_gyro_lpf_cutoff:.1f} Hz >= Nyquist {_nyq:.1f} Hz (fs={_fs:.1f}) — skipped")
        except ImportError:
            print("  WARNING: scipy not available — gyro LPF skipped")

    # ---- acc_z low-pass filter (damp wing-beat lift in v_z; z-only) ----------
    # mirrors the gyro LPF but only on az (f3); ax/ay and gyro untouched. 0 = disabled (regression-exact).
    _acc_lpf_cutoff = float(_get_setting(settings, getattr(args, "acc_lpf_cutoff_hz", None), "acc_lpf_cutoff_hz", DEFAULT_ACC_LPF_CUTOFF_HZ))
    if _acc_lpf_cutoff > 0.0:
        try:
            from scipy.signal import butter, filtfilt as _filtfilt
            _imu_idx = df.index[df["type"] == "I"]
            _imu_t_ms = df.loc[_imu_idx, "t_ms"].values.astype(float)
            _dt_med_s = float(np.median(np.diff(_imu_t_ms))) / 1000.0 if len(_imu_t_ms) > 1 else (1.0 / UPDATE_HZ)
            _fs = 1.0 / _dt_med_s if _dt_med_s > 0 else UPDATE_HZ  # actual IMU rate [Hz]
            _nyq = _fs / 2.0
            _wn = _acc_lpf_cutoff / _nyq
            if 0.0 < _wn < 1.0 and len(_imu_idx) > 12:
                _b, _a = butter(4, _wn, btype="low")
                _raw = df.loc[_imu_idx, "f3"].values.astype(float)
                df.loc[_imu_idx, "f3"] = _filtfilt(_b, _a, _raw)
                print(f"  Acc_z LPF: {_acc_lpf_cutoff:.1f} Hz Butterworth (4th order, zero-phase) "
                      f"[IMU fs={_fs:.1f} Hz, Nyquist={_nyq:.1f} Hz, Wn={_wn:.3f}]")
            elif _wn >= 1.0:
                print(f"  Acc_z LPF: cutoff {_acc_lpf_cutoff:.1f} Hz >= Nyquist {_nyq:.1f} Hz (fs={_fs:.1f}) — skipped")
        except ImportError:
            print("  WARNING: scipy not available — acc_z LPF skipped")

    # ---- liftoff detection for optional delayed EKF start -----------------
    flying_delay_s = cfg["flying_delay_s"]
    liftoff_s = None
    cutoff_ms = 0.0  # 0 = start feature updates immediately (phase 1 skipped)
    if flying_delay_s > 0.0:
        i_df      = df[df["type"] == "I"]
        imu_t_s   = i_df["t_ms"].values.astype(float) / 1000.0
        accel_mg  = i_df[["f1", "f2", "f3"]].values.astype(float)
        accel_ms2 = accel_mg * (9.81 / 1000.0)
        accel_mag = np.linalg.norm(accel_ms2, axis=1)
        liftoff_s = _detect_liftoff_imu(imu_t_s, accel_mag)
        if liftoff_s is not None:
            cutoff_ms = (liftoff_s + flying_delay_s) * 1000.0
            print(f"  IMU liftoff detected at {liftoff_s:.2f} s  "
                  f"→ feature updates begin at {liftoff_s + flying_delay_s:.2f} s "
                  f"(+{flying_delay_s:.1f} s delay)")
        else:
            print("  IMU liftoff not detected — feature updates begin immediately.")

    # ---- collected state --------------------------------------------------
    timestamps      = []   # float, ms
    states          = []   # list of 9-float lists [x,y,z,vx,vy,vz,d0,d1,d2]
    quaternions     = []   # list of 4-float tuples (w,x,y,z)
    cov_diagonals   = []   # list of 9-float lists (P diagonal)
    feature_snaps   = []   # list of lists of (id,bx,by,rho,Xw,Yw,Zw) tuples
    input_feature_snaps = []   # list of N lists of (track_id, px, py, n_updates)
    feature_cov_snaps   = []   # list of N lists of (id, sigma_bx, sigma_by, sigma_rho, rel_depth_unc)
    nis_records         = []   # (t_ms, d2) per feature update, for NIS consistency plots
    cov_full            = []   # list of N 9x9 P_rr matrices (only when log_full_cov)
    log_full_cov        = bool(getattr(args, "log_full_cov", False))  # full P_rr for NEES/correlation
    _end_s              = getattr(args, "end_s", None)                 # cap replay at this time [s]
    _replay_end_ms      = float(_end_s) * 1000.0 if _end_s else None

    # ---- feature management state ----------------------------------------
    track_to_feature: dict = {}
    track_missing: dict    = {}
    total_gated: int       = 0  # cumulative gate-rejected observations

    prev_t_ms = None
    cycle_idx = 0
    in_phase2 = cutoff_ms <= 0.0  # true immediately if no delay configured
    phase2_start_t_ms = 0.0 if cutoff_ms <= 0.0 else None  
    # gravity reference: collect data from [liftoff_ms - 4000, liftoff_ms - 500].
    _liftoff_ms        = liftoff_s * 1000.0 if liftoff_s is not None else cutoff_ms
    _GRAV_REF_END_MS   = _liftoff_ms - 500.0   # 500 ms safety margin before liftoff
    _GRAV_REF_START_MS = _liftoff_ms - 4000.0  # up to 4 s before liftoff
    grav_ref_buf: list = []       # list of (ax_mg, ay_mg, az_mg)
    gyro_bias = (0.0, 0.0, 0.0)   # subtracted from gyro in predict; set at phase-2 if calib valid
    _first_t_ms = float(df[df["type"] == "I"]["t_ms"].iloc[0])

    # gravity/accel attitude reference during flight (0 meas-var = disabled)
    _grav_meas_var = float(_get_setting(settings, getattr(args, "gravity_meas_var", None), "gravity_meas_var", 0.0))
    _grav_mag_tol  = float(_get_setting(settings, getattr(args, "gravity_mag_tol",  None), "gravity_mag_tol",  0.0))
    # static gyro-bias calib; not in paper (measure bias from the pre-liftoff still window)
    _gyro_bias_calib = bool(_get_setting(settings, getattr(args, "gyro_bias_calib", None), "gyro_bias_calib", False))

    for i_row, t_rows in group_cycles(df):
        t_ms = float(i_row.t_ms)
        if _replay_end_ms is not None and t_ms > _replay_end_ms:
            break   # stop at the eval-window end (post-window data has sensor blackouts)
        dt_ms = (t_ms - prev_t_ms) if prev_t_ms is not None else (1000.0 / UPDATE_HZ)
        prev_t_ms = t_ms

        ax  = float(i_row.f1)  # ax_mg  (body frame)
        ay  = float(i_row.f2)  # ay_mg
        az  = float(i_row.f3)  # az_mg
        gx  = float(i_row.f4)  # gx_dps (body frame)
        gy  = float(i_row.f5)  # gy_dps
        gz  = float(i_row.f6)  # gz_dps

        # accumulate accel in the window before phase-2 start (motors off, drone on ground)
        if _GRAV_REF_START_MS <= t_ms <= _GRAV_REF_END_MS:
            grav_ref_buf.append((ax, ay, az))

        # 1. IMU predict — phase 2 only. phase 1 (takeoff) has unstable IMU/features that would corrupt the state.
        if in_phase2:
            _DT_MAX_S = 0.050
            dt_s = min(dt_ms / 1000.0, _DT_MAX_S)
            if dt_ms / 1000.0 > _DT_MAX_S:
                print(f"  WARNING: large dt {dt_ms:.1f} ms at t={t_ms/1000:.3f} s — clamped to {_DT_MAX_S*1000:.0f} ms")
            kc.predict_dt(ax, ay, az, gx - gyro_bias[0], gy - gyro_bias[1], gz - gyro_bias[2], dt_s)
            # 1b. gravity/accel attitude reference (roll/pitch), if enabled
            if _grav_meas_var > 0.0:
                kc.gravity_update(ax, ay, az, _grav_meas_var, _grav_mag_tol)

        # 2. feature measurement updates
        tracks = []
        for t in t_rows:
            tracks.append((
                int(t.f1),    # track_id
                float(t.f2),  # px
                float(t.f3),  # py
                float(t.f4),  # vx
                float(t.f5),  # vy
                int(t.f6),    # n_updates
            ))

        # ---- phase 1 → phase 2 transition --------------------------------
        if not in_phase2 and t_ms >= cutoff_ms:
            in_phase2 = True
            phase2_start_t_ms = t_ms

            # -- gravity-aligned attitude initialisation --
            # sanity check: magnitude must be near 9.81; otherwise fall back to identity attitude (roll=pitch=0).
            _GRAV_MAG_TOL = 1.0
            yaw = 0.0
            use_gravity_init = False
            grav_mag = 0.0
            if len(grav_ref_buf) >= 5:
                mean_ax = sum(v[0] for v in grav_ref_buf) / len(grav_ref_buf)
                mean_ay = sum(v[1] for v in grav_ref_buf) / len(grav_ref_buf)
                mean_az = sum(v[2] for v in grav_ref_buf) / len(grav_ref_buf)
                g_x = mean_ax * (9.81 / 1000.0)
                g_y = mean_ay * (9.81 / 1000.0)
                g_z = mean_az * (9.81 / 1000.0)
                grav_mag = math.sqrt(g_x**2 + g_y**2 + g_z**2)
                if abs(grav_mag - 9.81) <= _GRAV_MAG_TOL:
                    roll_grav  = math.atan2(g_y, g_z)
                    pitch_grav = math.atan2(-g_x, math.sqrt(g_y**2 + g_z**2))
                    use_gravity_init = True

            if use_gravity_init:
                roll_deg  = math.degrees(roll_grav)
                pitch_deg = math.degrees(pitch_grav)
                print(f"  Phase 2 start at t={t_ms/1000:.2f} s  "
                      f"(roll={roll_deg:+.1f}deg  pitch={pitch_deg:+.1f}deg  yaw=+0.0deg)"
                      f"  [grav-ref n={len(grav_ref_buf)}, mag={grav_mag:.2f} m/s2]")
                cr, sr = math.cos(roll_grav/2),  math.sin(roll_grav/2)
                cp, sp = math.cos(pitch_grav/2), math.sin(pitch_grav/2)
                init_qw = cp*cr
                init_qx = cp*sr
                init_qy = sp*cr
                init_qz = -sp*sr
            else:
                n_str  = str(len(grav_ref_buf))
                mag_str = f", mag={grav_mag:.2f} m/s2" if len(grav_ref_buf) >= 5 else ""
                print(f"  Phase 2 start at t={t_ms/1000:.2f} s  "
                      f"(roll=+0.0deg  pitch=+0.0deg  yaw=+0.0deg)"
                      f"  [identity fallback: n={n_str}{mag_str}]")
                init_qw, init_qx, init_qy, init_qz = 1.0, 0.0, 0.0, 0.0

            # static gyro-bias; not in paper: mean of raw gyro over the pre-liftoff still window. rejected if window short, gyro not still, or accel mag off.
            if _gyro_bias_calib:
                _wmask = (_gyro_raw_t >= _GRAV_REF_START_MS) & (_gyro_raw_t <= _GRAV_REF_END_MS)
                _gw = _gyro_raw[_wmask]
                _gb_n = len(_gw)
                if _gb_n >= 30:
                    _bias = _gw.mean(axis=0)
                    _gstd = float(_gw.std(axis=0).max())
                    if _gstd < 8.0 and 9.2 <= grav_mag <= 10.4:
                        gyro_bias = (float(_bias[0]), float(_bias[1]), float(_bias[2]))
                        print(f"  gyro-bias calib: ({_bias[0]:+.2f}, {_bias[1]:+.2f}, {_bias[2]:+.2f}) dps "
                              f"[n={_gb_n}, max_std={_gstd:.1f}, accel={grav_mag:.2f}]")
                    else:
                        print(f"  WARNING: gyro-bias calib SKIPPED "
                              f"(max_std={_gstd:.1f}dps accel={grav_mag:.2f}m/s2 n={_gb_n})")
                else:
                    print(f"  WARNING: gyro-bias calib SKIPPED (window too short n={_gb_n})")

            kc.reset(0.0, 0.0, 0.0, yaw)
            kc.set_attitude_quat(init_qw, init_qx, init_qy, init_qz)
            # kc.reset() calls set_default_params() internally, wiping all custom noise settings. re-apply them now.
            _apply_config_settings(settings, args)
            track_to_feature.clear()
            track_missing.clear()

        if in_phase2:
            # gate warmup: suppress innovation gate for the first gate_warmup_s seconds of phase 2 so the filter can converge first.
            _gate_warmup_s = float(cfg.get("gate_warmup_s", DEFAULT_GATE_WARMUP_S))
            _in_warmup = (t_ms - phase2_start_t_ms) < _gate_warmup_s * 1000.0
            if _in_warmup:
                kc.set_innovation_gate(0.0)
            else:
                kc.set_innovation_gate(float(cfg.get("innovation_gate", DEFAULT_INNOVATION_GATE)))

            n_gated = _manage_tracks(tracks, track_to_feature, track_missing,
                                     cfg["min_updates"], cfg["rho_init"], cfg["lost_frames"],
                                     t_ms=t_ms, nis_out=nis_records,
                                     flow_meas_var=cfg["flow_meas_var"],
                                     gyro_xyz=(gx - gyro_bias[0], gy - gyro_bias[1], gz - gyro_bias[2]),
                                     base_meas_noise=cfg["feature_meas_noise"],
                                     age_pow=cfg["age_trust_pow"], age_floor=cfg["age_trust_floor"])
            total_gated += n_gated

        # 3. process noise — phase 2 only (consistent with predict)
        if in_phase2:
            kc.add_process_noise(dt_ms)

        # 4. finalise
        kc.finalize()

        # ---- collect snapshot -------------------------------------------
        timestamps.append(t_ms)
        s = kc.get_state()       # always called (used by progress print below)
        quaternions.append(kc.get_quaternion())
        if in_phase2:
            states.append([s["x"], s["y"], s["z"], s["vx"], s["vy"], s["vz"],
                            s["d0"], s["d1"], s["d2"]])
            P = kc.get_full_covariance()
            cov_diagonals.append([P[i][i] for i in range(9)])
            if log_full_cov:
                cov_full.append([row[:] for row in P])   # 9x9 robot covariance
            feature_snaps.append(kc.get_all_features())
        else:
            # phase 1: don't trust pos/vel yet — log zeros; attitude is real
            states.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            cov_diagonals.append([0.0] * 9)
            if log_full_cov:
                cov_full.append([[0.0] * 9 for _ in range(9)])
            feature_snaps.append([])
        # raw input pixels always logged so the 2D camera panel works in phase 1
        input_feature_snaps.append(
            [(t[0], t[1], t[2], t[5]) for t in tracks]
        )
        cov_snap = []
        for f in feature_snaps[-1]:
            fid, rho = int(f[0]), float(f[3])
            try:
                P_bx, P_by, P_rho = kc.get_feature_cov_diag(fid)
                sigma_rho = P_rho ** 0.5
                rel_depth_unc = sigma_rho / max(abs(rho), 1e-6)
                cov_snap.append((fid, P_bx**0.5, P_by**0.5, sigma_rho, rel_depth_unc))
            except Exception:
                pass
        feature_cov_snaps.append(cov_snap)

        cycle_idx += 1
        if cycle_idx % 200 == 0:
            n_feat = kc.get_active_feature_count()
            x, y, z = s["x"], s["y"], s["z"]
            print(f"  cycle {cycle_idx:5d}  t={t_ms/1000:.2f}s  "
                  f"pos=({x:.3f},{y:.3f},{z:.3f})  n_feat={n_feat}"
                  + (f"  gated={total_gated}" if total_gated > 0 else ""))

    print(f"Replay complete: {cycle_idx} cycles processed."
          + (f"  Innovation-gated observations: {total_gated}" if total_gated > 0 else ""))

    data = {
        "timestamps":    np.array(timestamps),
        "states":        np.array(states),         # (N, 9)
        "quaternions":   np.array(quaternions),    # (N, 4)
        "cov_diagonals": np.array(cov_diagonals),  # (N, 9)
        "feature_snaps": feature_snaps,            # list of lists
        "input_feature_snaps": input_feature_snaps, # list of lists
        "feature_cov_snaps":   feature_cov_snaps,   # list of lists
        "nis": np.array(nis_records, dtype=np.float64) if nis_records
               else np.empty((0, 2), dtype=np.float64),  # (M, 2) = (t_ms, d2)
    }
    if log_full_cov:
        data["cov_full"] = np.array(cov_full)      # (N, 9, 9)
    return data


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", help="path to rec_NNN.csv recording")
    p.add_argument("--config", type=str, default=None,
                   help=f"path to a YAML or TXT config file with Kalman settings (default: {DEFAULT_CONFIG_PATH})")
    p.add_argument("--log_full_cov", action="store_true",
                   help="log the full 9x9 robot covariance per step (for NEES / "
                        "pitch-lateral correlation diagnostics); off by default")
    p.add_argument("--end_s", type=float, default=None,
                   help="stop the replay at this time [s] (drone clock); used to cap "
                        "evaluation at the eval-window end, ignoring post-window sensor "
                        "blackouts. Default: replay the full recording.")
    p.add_argument("--update_rate", type=float, default=None,
                   help="EKF update rate in Hz")
    p.add_argument("--flying", action=argparse.BooleanOptionalAction,
                   help="enable or disable in-flight predict path")
    p.add_argument("--acc_xy",      type=float, default=None,
                   help="procNoiseAcc_xy")
    p.add_argument("--acc_z",       type=float, default=None,
                   help="procNoiseAcc_z")
    p.add_argument("--gyro_rp",     type=float, default=None,
                   help="measNoiseGyro_rollpitch")
    p.add_argument("--gyro_yaw",    type=float, default=None,
                   help="measNoiseGyro_yaw")
    p.add_argument("--att_rev",     type=float, default=None,
                   help="attitude_reversion")
    p.add_argument("--vel",         type=float, default=None,
                   help="procNoiseVel")
    p.add_argument("--pos",         type=float, default=None,
                   help="procNoisePos")
    p.add_argument("--att",         type=float, default=None,
                   help="procNoiseAtt")
    p.add_argument("--meas_noise",  type=float, default=None,
                   help="feature measurement noise variance")
    p.add_argument("--fallback_rho", type=float, default=None,
                   help="fallback inverse depth used when no features exist yet [1/m]")
    p.add_argument("--init_stddev_idepth", type=float, default=None,
                   help="initial relative std-dev for inverse-depth covariance (dimensionless, default 0.001)")
    p.add_argument("--max_depth_uncertainty_ratio", type=float, default=None,
                   help="sigma_rho/rho threshold for adaptive median-rho (default 0.5)")
    p.add_argument("--innovation_gate", type=float, default=None,
                   help="Mahalanobis distance² gate for feature updates; "
                        "observations above this are skipped (0=disabled, "
                        "chi²(2,0.95)=5.991, chi²(2,0.99)=9.210)")
    p.add_argument("--vel_max", type=float, default=None,
                   help="physical velocity clamp [m/s] applied in predict; "
                        "bounds gravity-leakage runaway (0=disabled)")
    p.add_argument("--gyro_lpf_cutoff_hz", type=float, default=None,
                   help="gyro low-pass cutoff [Hz] (Butterworth 4th-order, zero-phase); "
                        "0=disabled (overrides gyro_lpf_cutoff_hz from config YAML)")
    p.add_argument("--acc_lpf_cutoff_hz", type=float, default=None,
                   help="acc_z low-pass cutoff [Hz] (Butterworth 4th-order, zero-phase); "
                        "damps wing-beat lift in v_z; 0=disabled (overrides config YAML)")
    p.add_argument("--gyro_bias_calib", action=argparse.BooleanOptionalAction, default=None,
                   help="subtract a static gyro bias measured from the pre-liftoff still "
                        "window (off by default; skipped if the window is not still/level)")
    p.add_argument("--depth_type", type=str, default=None,
                   help="feature depth parametrization: regular/inverse/log/hyperbolic "
                        "(default inverse = original behaviour)")
    p.add_argument("--gate_warmup_s", type=float, default=None,
                   help="seconds after phase-2 start during which the gate is disabled "
                        "so the filter can converge before gating is enforced (default 5.0)")
    p.add_argument("--init_stddev_pos_xy",  type=float, default=None,
                   help="initial position XY std-dev at phase-2 reset [m] (default 0.1)")
    p.add_argument("--init_stddev_pos_z",   type=float, default=None,
                   help="initial position Z std-dev at phase-2 reset [m] (default 0.1)")
    p.add_argument("--init_stddev_vel",     type=float, default=None,
                   help="initial velocity std-dev at phase-2 reset [m/s] (default 0.2)")
    p.add_argument("--init_stddev_att_rp",  type=float, default=None,
                   help="initial roll/pitch std-dev at phase-2 reset [rad] (default 0.05)")
    p.add_argument("--init_stddev_att_yaw", type=float, default=None,
                   help="initial yaw std-dev at phase-2 reset [rad] (default 0.05)")
    p.add_argument("--rho_init",    type=float, default=None,
                   help="initial inverse depth for new features [1/m]")
    p.add_argument("--min_updates", type=int,   default=None,
                   help="minimum n_updates to accept a track")
    p.add_argument("--lost_frames", type=int,   default=None,
                   help="missing cycles before feature is removed")
    p.add_argument("--flying_delay_s", type=float, default=None,
                   help="seconds after IMU-detected liftoff before feature updates begin "
                        "(0.0 = disabled; overrides flying_delay_s from config YAML)")
    p.add_argument("--save",        type=str,   default=None,
                   help="save result to .npz file (e.g. result.npz)")
    p.add_argument("--rerun",       action="store_true",
                   help="stream to rerun.io (requires rerun-sdk)")
    p.add_argument("--plot",        action="store_true",
                   help="show static matplotlib plots after replay")
    return p


def main():
    args = build_parser().parse_args()

    data = replay(args)

    if args.save:
        # save numpy arrays; feature_snaps is a ragged list so save separately
        _RAGGED = {"feature_snaps", "input_feature_snaps", "feature_cov_snaps"}
        out = {k: v for k, v in data.items() if k not in _RAGGED}
        np.savez_compressed(args.save, **out)
        print(f"Saved numeric data to {args.save}")

    if args.rerun:
        try:
            import visualize
            visualize.run(data)
        except ImportError as e:
            print(f"Cannot import visualize: {e}")

    if args.plot:
        try:
            import plot_static
            plot_static.run(data)
        except ImportError as e:
            print(f"Cannot import plot_static: {e}")

    if not args.rerun and not args.plot:
        # print a brief summary
        st = data["states"]
        print("\nFinal state:")
        keys = ["x", "y", "z", "vx", "vy", "vz", "d0", "d1", "d2"]
        for k, v in zip(keys, st[-1]):
            print(f"  {k:4s} = {v:+.4f}")
        t = data["timestamps"]
        print(f"\nDuration: {(t[-1]-t[0])/1000:.2f} s  ({len(t)} cycles)")


if __name__ == "__main__":
    main()
