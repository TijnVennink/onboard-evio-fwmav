"""
EKF parameter sweep with nevergrad CMA, or DE.

loads a sync-offset YAML for the CSV paths and ground truth alignment, then tunes EKF noise parameters by minimising position + orientation RMSE against OptiTrack.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import math
import os
import sys

import numpy as np
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _OPTUNA_AVAILABLE = True
except ImportError:
    _OPTUNA_AVAILABLE = False

import replay as _replay
from helpers import (
    load_optitrack_ground_truth as _load_ot_gt,
    anchor_ground_truth,
    detect_liftoff_imu as _detect_liftoff_imu,
    load_drone_imu as _load_drone_imu,
)

# CMA optimizer package (optimizer/ subdirectory)
from optimizer.optimize       import (
    CMAOptimizer, NevergradOptimizer, build_eval_config, write_best_config, PARAM_TO_YAML,
)
from optimizer.parameter_space import ParameterSpace

# frame rotation is loaded per-recording from the offset YAML.


# ---------------------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------------------

def _load_offset_yaml(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_sweep_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ---------------------------------------------------------------------------
# metric helpers
# ---------------------------------------------------------------------------

def _find_closest_indices(sorted_ts: np.ndarray, query_ts: np.ndarray) -> np.ndarray:
    """for each value in query_ts return the index of the nearest element in sorted_ts."""
    idx = np.searchsorted(sorted_ts, query_ts)
    idx = np.clip(idx, 0, len(sorted_ts) - 1)
    left = np.clip(idx - 1, 0, len(sorted_ts) - 1)
    take_left = np.abs(sorted_ts[left] - query_ts) <= np.abs(sorted_ts[idx] - query_ts)
    return np.where(take_left, left, idx)


def _compute_metrics(
    drone_t_s:   np.ndarray,   # (N,) seconds
    drone_pos:   np.ndarray,   # (N, 3) metres
    drone_q:     np.ndarray,   # (N, 4) (w, x, y, z)
    gt_ts:       np.ndarray,   # (M,) seconds (offset-aligned, anchored)
    gt_pos:      np.ndarray,   # (M, 3) metres
    gt_q:        np.ndarray,   # (M, 4) (w, x, y, z)
    eval_start_s: float,
    pos_weight:   float,
    ori_weight:   float,
    eval_end_s:   float = np.inf,
) -> float:
    """
    returns pos_weight * pos_RMSE + ori_weight * ori_RMSE_rad.
    returns 1e6 if there are fewer than 10 usable samples.
    """
    mask = (drone_t_s >= eval_start_s) & (drone_t_s <= eval_end_s)
    if mask.sum() < 10:
        return 1e6

    d_t = drone_t_s[mask]
    d_p = drone_pos[mask]
    d_q = drone_q[mask]

    # match each drone timestamp to the nearest GT timestamp
    gi = _find_closest_indices(gt_ts, d_t)
    time_gap = np.abs(gt_ts[gi] - d_t)
    valid = time_gap < 0.5         # discard pairs > 0.5 s apart

    if valid.sum() < 10:
        return 1e6

    d_p = d_p[valid];  d_q = d_q[valid]
    g_p = gt_pos[gi[valid]]
    g_q = gt_q[gi[valid]]

    # position RMSE [m]
    pos_err  = np.linalg.norm(d_p - g_p, axis=1)
    pos_rmse = float(np.sqrt(np.mean(pos_err ** 2)))

    # orientation geodesic error [rad]:  2·arccos(|q1·q2|)  handles double-cover
    dot      = np.clip(np.abs(np.einsum("ij,ij->i", d_q, g_q)), 0.0, 1.0)
    ori_err  = 2.0 * np.arccos(dot)
    ori_rmse = float(np.sqrt(np.mean(ori_err ** 2)))

    return pos_weight * pos_rmse + ori_weight * ori_rmse


# ---------------------------------------------------------------------------
# anchor helper (mirrors visualize_combined.py logic)
# ---------------------------------------------------------------------------

def _find_anchor_poses(
    drone_t_s:    np.ndarray,
    drone_states: np.ndarray,
    drone_qs:     np.ndarray,
    gt_ts:        np.ndarray,
    gt_pos:       np.ndarray,
    gt_qs:        np.ndarray,
    anchor_t_s:   float,
) -> tuple | None:
    """
    find nearest samples at anchor_t_s in both datasets.
    returns (drone_pos, drone_quat, gt_pos, gt_quat) or None if the anchor is more than 2 s away from real data in either dataset.
    """
    di = int(np.argmin(np.abs(drone_t_s - anchor_t_s)))
    gi = int(np.argmin(np.abs(gt_ts     - anchor_t_s)))

    if abs(drone_t_s[di] - anchor_t_s) > 2.0:
        return None
    if abs(gt_ts[gi] - anchor_t_s) > 2.0:
        return None

    d_pos  = np.array(drone_states[di, :3], dtype=np.float32)
    d_quat = tuple(float(drone_qs[di, k]) for k in range(4))
    g_pos  = np.array(gt_pos[gi], dtype=np.float32)
    g_quat = tuple(float(gt_qs[gi, k]) for k in range(4))
    return d_pos, d_quat, g_pos, g_quat


# ---------------------------------------------------------------------------
# replay args builder
# ---------------------------------------------------------------------------

# maps sweep param names (dot-notation) → argparse dest name in replay.py
_PARAM_TO_ARG: dict[str, str] = {
    "process_noise.acc_xy":        "acc_xy",
    "process_noise.acc_z":         "acc_z",
    "process_noise.gyro_rp":       "gyro_rp",
    "process_noise.gyro_yaw":      "gyro_yaw",
    "process_noise.att_reversion": "att_rev",
    "process_noise.vel":           "vel",
    "process_noise.pos":           "pos",
    "process_noise.att":           "att",
    "feature_meas_noise":          "meas_noise",
    "rho_init":                    "rho_init",
    "init_stddev_idepth":          "init_stddev_idepth",
    "max_depth_uncertainty_ratio": "max_depth_uncertainty_ratio",
    "min_updates":                 "min_updates",
    "lost_frames":                 "lost_frames",
    "flying_delay_s":              "flying_delay_s",
    "fallback_rho":                "fallback_rho",
    "innovation_gate":             "innovation_gate",
    "gate_warmup_s":               "gate_warmup_s",
    "init_stddev_pos_xy":          "init_stddev_pos_xy",
    "init_stddev_pos_z":           "init_stddev_pos_z",
    "init_stddev_vel":             "init_stddev_vel",
    "init_stddev_att_rp":          "init_stddev_att_rp",
    "init_stddev_att_yaw":         "init_stddev_att_yaw",
    "gravity_meas_var":            "gravity_meas_var",
    "gravity_mag_tol":             "gravity_mag_tol",
    "gyro_bias_noise":             "gyro_bias_noise",
    "init_stddev_gyro_bias":       "init_stddev_gyro_bias",
    "depth_type":                  "depth_type",
    "flow_meas_var":               "flow_meas_var",
    "huber_delta":                 "huber_delta",
    "age_trust_pow":               "age_trust_pow",
}


def _build_replay_args(
    drone_csv: str,
    base_config_path: str,
    param_values: dict,
) -> argparse.Namespace:
    """create an args Namespace for replay() with sampled params overriding YAML defaults."""
    parser = _replay.build_parser()
    ns = parser.parse_args([drone_csv])
    ns.config = base_config_path
    # suppress rerun/plot/save inside sweep
    ns.rerun  = False
    ns.plot   = False
    ns.save   = None
    for pname, value in param_values.items():
        arg_dest = _PARAM_TO_ARG.get(pname)
        if arg_dest is not None:
            setattr(ns, arg_dest, value)
    return ns


# ---------------------------------------------------------------------------
# optuna objective factory (legacy, not in paper)
# ---------------------------------------------------------------------------

def _build_objective(
    drone_csv:        str,
    base_config_path: str,
    gt_cached:        dict,   # raw GT (NOT anchored) cached before the study
    liftoff_s:        float,
    flying_delay_s_base: float,
    eval_start_s:     float,
    sweep_params_cfg: dict,
    pos_weight:       float,
    ori_weight:       float,
    eval_end_s:       float = np.inf,
) -> callable:

    gt_ts_cached  = gt_cached["timestamps"].astype(np.float64)
    gt_pos_cached = gt_cached["positions"].astype(np.float64)
    gt_qs_cached  = gt_cached["quaternions"].astype(np.float64)

    def objective(trial: optuna.Trial) -> float:

        # ── 1. sample parameters ─────────────────────────────────────────────
        param_values: dict = {}
        for pname, pcfg in sweep_params_cfg.items():
            if not pcfg.get("enabled", True):
                continue
            ptype   = pcfg.get("type", "float")
            low     = float(pcfg["low"])
            high    = float(pcfg["high"])
            use_log = bool(pcfg.get("log", False))
            if ptype == "int":
                param_values[pname] = trial.suggest_int(pname, int(low), int(high))
            else:
                param_values[pname] = trial.suggest_float(pname, low, high, log=use_log)

        # ── 2. run replay (stdout silenced to keep progress bar clean) ───────
        ns = _build_replay_args(drone_csv, base_config_path, param_values)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                drone_data = _replay.replay(ns)
        except Exception as e:
            print(f"\n  [trial {trial.number}] replay failed: {e}")
            return 1e6

        drone_t_s = drone_data["timestamps"].astype(np.float64) / 1000.0
        drone_pos = drone_data["states"][:, :3].astype(np.float64)
        drone_qs  = drone_data["quaternions"].astype(np.float64)

        # ── 3. anchor GT per-trial using sampled anchor_delay_s (if present) ──
        # anchor delay: sampled value or fall back to base flying delay
        anchor_delay = param_values.get("anchor_delay_s", float(flying_delay_s_base))
        anchor_t_s = float(liftoff_s) + float(anchor_delay)

        # find nearest samples in drone and cached GT for anchoring
        di = int(np.argmin(np.abs(drone_t_s - anchor_t_s)))
        gi = int(np.argmin(np.abs(gt_ts_cached - anchor_t_s)))

        if abs(drone_t_s[di] - anchor_t_s) > 2.0 or abs(gt_ts_cached[gi] - anchor_t_s) > 2.0:
            # anchoring failed for this trial — penalize heavily
            print(f"  [trial {trial.number}] anchoring failed: nearest drone gap={abs(drone_t_s[di]-anchor_t_s):.2f}s, gt gap={abs(gt_ts_cached[gi]-anchor_t_s):.2f}s")
            return 1e6

        drone_anchor_pos = np.array(drone_pos[di], dtype=np.float32)
        drone_anchor_quat = tuple(float(drone_qs[di, k]) for k in range(4))

        gt_anchor_pos = np.array(gt_pos_cached[gi], dtype=np.float32)
        gt_anchor_quat = tuple(float(gt_qs_cached[gi, k]) for k in range(4))

        # build an anchored GT for this trial
        gt_trial = anchor_ground_truth(
            {"timestamps": gt_ts_cached, "positions": gt_pos_cached, "quaternions": gt_qs_cached},
            drone_anchor_pos, drone_anchor_quat,
            gt_anchor_pos, gt_anchor_quat,
        )

        gt_ts = gt_trial["timestamps"].astype(np.float64)
        gt_pos = gt_trial["positions"].astype(np.float64)
        gt_qs  = gt_trial["quaternions"].astype(np.float64)

        # ── 4. compute objective using the trial-anchored GT ────────────────
        score = _compute_metrics(
            drone_t_s, drone_pos, drone_qs,
            gt_ts, gt_pos, gt_qs,
            eval_start_s, pos_weight, ori_weight, eval_end_s,
        )

        return score

    return objective


# ---------------------------------------------------------------------------
# write best config
# ---------------------------------------------------------------------------

# maps sweep param name → path into the YAML config dict
_PARAM_TO_YAML: dict[str, tuple] = {
    "process_noise.acc_xy":        ("process_noise", "acc_xy"),
    "process_noise.acc_z":         ("process_noise", "acc_z"),
    "process_noise.gyro_rp":       ("process_noise", "gyro_rp"),
    "process_noise.gyro_yaw":      ("process_noise", "gyro_yaw"),
    "process_noise.att_reversion": ("process_noise", "att_reversion"),
    "process_noise.vel":           ("process_noise", "vel"),
    "process_noise.pos":           ("process_noise", "pos"),
    "process_noise.att":           ("process_noise", "att"),
    "feature_meas_noise":          ("feature_meas_noise",),
    "rho_init":                    ("rho_init",),
    "init_stddev_idepth":          ("init_stddev_idepth",),
    "max_depth_uncertainty_ratio": ("max_depth_uncertainty_ratio",),
    "min_updates":                 ("min_updates",),
    "lost_frames":                 ("lost_frames",),
    "flying_delay_s":              ("flying_delay_s",),
    "fallback_rho":                ("fallback_rho",),
    "anchor_delay_s":              ("anchor_delay_s",),
    "innovation_gate":             ("innovation_gate",),
    "gate_warmup_s":               ("gate_warmup_s",),
    "init_stddev_pos_xy":          ("init_stddev_pos_xy",),
    "init_stddev_pos_z":           ("init_stddev_pos_z",),
    "init_stddev_vel":             ("init_stddev_vel",),
    "init_stddev_att_rp":          ("init_stddev_att_rp",),
    "init_stddev_att_yaw":         ("init_stddev_att_yaw",),
    "gravity_meas_var":            ("gravity_meas_var",),
    "gravity_mag_tol":             ("gravity_mag_tol",),
    "gyro_bias_noise":             ("gyro_bias_noise",),
    "init_stddev_gyro_bias":       ("init_stddev_gyro_bias",),
    "depth_type":                  ("depth_type",),
    "flow_meas_var":               ("flow_meas_var",),
    "huber_delta":                 ("huber_delta",),
    "age_trust_pow":               ("age_trust_pow",),
}


def _write_best_config(
    base_config_path: str,
    best_params: dict,
    output_path: str,
) -> None:
    with open(base_config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    for pname, value in best_params.items():
        yaml_path = _PARAM_TO_YAML.get(pname)
        if yaml_path is None:
            continue
        if len(yaml_path) == 2:
            if yaml_path[0] not in cfg or not isinstance(cfg[yaml_path[0]], dict):
                cfg[yaml_path[0]] = {}
            cfg[yaml_path[0]][yaml_path[1]] = round(float(value), 8)
        else:
            cfg[yaml_path[0]] = round(float(value), 8)

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    print(f"Best config written → {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "EKF noise parameter sweep (CMA-ES or Optuna TPE), minimising "
            "position + orientation RMSE against OptiTrack ground truth."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--offset_yaml",      required=True,
                   help="Sync offset YAML produced by sync_times.py")
    p.add_argument("--sweep_config",     default="kalman_sweep_config.yaml",
                   help="Sweep parameter config YAML")
    p.add_argument("--base_config",      default="kalman_replay_config.yaml",
                   help="Base EKF config (unmodified; sweep params override it per trial)")
    p.add_argument("--n_trials",         type=int,   default=None,
                   help="Override n_trials from sweep config")
    p.add_argument("--runtime_s",        type=float, default=20.0,
                   help="Only evaluate the first N seconds of trajectory after liftoff")
    p.add_argument("--output_config",    default="kalman_replay_config_best.yaml",
                   help="Output path for the best config YAML")
    # ── optimizer selection ──────────────────────────────────────────────────
    p.add_argument("--optimizer",        choices=["cma", "multicma", "bipop", "ngopt", "de", "tpe"], default="cma",
                   help="Optimizer backend: cma/multicma/bipop/ngopt/de = Nevergrad "
                        "(bipop requires: pip install pymoo), tpe = Optuna TPE (legacy)")
    # ── CMA specific ─────────────────────────────────────────────────────────
    p.add_argument("--n_workers",        type=int,   default=None,
                   help="[CMA] Parallel evaluation workers (1 = sequential)")
    p.add_argument("--sigma0",           type=float, default=None,
                   help="[CMA] Initial CMA step size in normalised parameter space")
    p.add_argument("--population_size",  type=int,   default=None,
                   help="[CMA] Population size per generation (None = auto)")
    p.add_argument("--checkpoint_every", type=int,   default=None,
                   help="[CMA] Save optimizer checkpoint every N trials")
    p.add_argument("--output_dir",       type=str,   default=None,
                   help="[CMA] Directory for history.csv, checkpoints (default: optimizer_output/)")
    p.add_argument("--resume",           type=str,   default=None,
                   help="[CMA] Path to a checkpoint .pkl file to resume from")
    p.add_argument("--warm_start",       action="store_true",
                   help="[CMA] Seed CMA initial mean from base_config values")
    p.add_argument("--preview",           action="store_true",
                   help="Launch rerun to visualise the base run + anchored GT before the sweep "
                        "starts; press Enter in the terminal to proceed")
    return p


def main() -> None:
    args = build_parser().parse_args()

    wdir = os.getcwd()

    # ── load configs ────────────────────────────────────────────────────────
    offset_cfg = _load_offset_yaml(args.offset_yaml)
    sweep_cfg  = _load_sweep_config(
        args.sweep_config if os.path.isabs(args.sweep_config)
        else os.path.join(_HERE, args.sweep_config)
    )

    base_config_path = (
        args.base_config if os.path.isabs(args.base_config)
        else os.path.join(_HERE, args.base_config)
    )

    sweep_params_cfg = sweep_cfg.get("parameters", {})
    obj_cfg          = sweep_cfg.get("objective",  {})
    opt_cfg          = sweep_cfg.get("optimizer",  {})

    n_trials   = args.n_trials if args.n_trials is not None else int(opt_cfg.get("n_trials", 100))
    output_config_path = (
        args.output_config if os.path.isabs(args.output_config)
        else os.path.join(_HERE, args.output_config)
    )

    # ── per-recording rotations from offset YAML ─────────────────────────────
    def _parse_rotation(key: str, default: np.ndarray) -> np.ndarray:
        raw = offset_cfg.get(key)
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

    _identity     = np.eye(3, dtype=np.float32)
    ot_rotation   = _parse_rotation("optitrack_rotation", _identity)
    body_rotation = _parse_rotation("gt_body_rotation",   None)  # type: ignore[arg-type]

    # ── nevergrad path (CMA, NGOpt, DE) ─────────────────────────────────────────
    if args.optimizer in ("cma", "multicma", "bipop", "ngopt", "de"):
        n_workers  = args.n_workers        if args.n_workers       is not None else int(opt_cfg.get("n_workers",        1))
        seed       = int(opt_cfg.get("seed", 42))
        sigma0     = args.sigma0           if args.sigma0          is not None else float(opt_cfg.get("sigma0",           0.5))
        pop_size   = args.population_size  if args.population_size is not None else opt_cfg.get("population_size")
        ckpt_every = args.checkpoint_every if args.checkpoint_every is not None else int(opt_cfg.get("checkpoint_every",  10))
        output_dir = args.output_dir       if args.output_dir      is not None else str(opt_cfg.get("output_dir",  "optimizer_output"))
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(_HERE, output_dir)

        # build parameter space
        param_space = ParameterSpace(sweep_params_cfg)

        # optional warm-start from base config
        warm_start_values: dict | None = None
        if args.warm_start:
            try:
                with open(base_config_path, encoding="utf-8") as f:
                    _bcfg = yaml.safe_load(f) or {}
                warm_start_values = {}
                for k in param_space.enabled_names():
                    if "." in k:
                        top, sub = k.split(".", 1)
                        val = (_bcfg.get(top) or {}).get(sub)
                    else:
                        val = _bcfg.get(k)
                    if val is not None:
                        warm_start_values[k] = float(val)
                print(f"  Warm-start values found for {len(warm_start_values)} params")
            except Exception as e:
                print(f"  Warning: warm-start failed: {e}")

        # build eval config (loads GT, detects liftoff, anchors GT).
        # runtime_s comes from the sweep config "objective", else the CLI --runtime_s default.
        runtime_s = float(obj_cfg.get("runtime_s", args.runtime_s))

        eval_cfg, eval_start_s = build_eval_config(
                offset_cfg, base_config_path, obj_cfg, sweep_cfg, runtime_s, wdir,
                ot_rotation=ot_rotation,
                body_rotation=body_rotation if isinstance(body_rotation, np.ndarray) else None,
            )

        # ── optional preview: visualise base run + anchored GT in rerun ─────────
        if args.preview:
            print("\nPreview — launching rerun with base run + anchored GT …")
            from visualize_combined import (
                run_combined as _run_combined,
                _trim_drone_data as _trim_drone,
                _trim_gt_data    as _trim_gt,
            )
            from optimizer.replay_runner import build_replay_args as _bra
            _prev_ns = _bra(eval_cfg.drone_csv, eval_cfg.base_config_path, {})
            if "flying_delay_s" in offset_cfg:
                _prev_ns.flying_delay_s = float(offset_cfg["flying_delay_s"])
            if "fallback_rho" in offset_cfg:
                _prev_ns.fallback_rho = float(offset_cfg["fallback_rho"])
            print("  Running base EKF replay …")
            _prev_drone = _replay.replay(_prev_ns)
            _prev_gt = {
                "timestamps":                  eval_cfg.gt_ts,
                "positions":                   eval_cfg.gt_pos,
                "quaternions":                 eval_cfg.gt_qs,
                "marker_positions":            None,
                "marker_names":                None,
                "unlabeled_marker_positions":  None,
            }
            # trim both streams to runtime_s seconds (same window the sweep evaluates)
            _cutoff = float(eval_cfg.eval_end_s)
            print(f"  Trimming preview to t ≤ {_cutoff:.2f} s (runtime_s = {runtime_s:.1f} s after liftoff) …")
            _prev_drone = _trim_drone(_prev_drone, _cutoff)
            _prev_gt    = _trim_gt(_prev_gt, _cutoff)
            print(f"  Preview — drone: {len(_prev_drone['timestamps'])} steps, "
                  f"GT: {len(_prev_gt['timestamps'])} poses")
            _run_combined(_prev_drone, _prev_gt, float(offset_cfg.get("offset_s", 0.0)))
            input("\n  Inspect rerun, then press Enter here to start the sweep … ")

        # print all params
        col_w = 44
        print(f"\nSweeping {param_space.n_dims} parameters over {n_trials} trials")
        print(f"  {'Parameter':<{col_w}} {'Low':>8}  {'High':>8}  {'Log'}")
        print(f"  {'-'*col_w}  {'--------'}  {'--------'}  {'---'}")
        for spec in param_space._specs:
            print(f"  {spec.name:<{col_w}} {spec.low:>8.4g}  {spec.high:>8.4g}  {spec.log}")
        print()

        # run optimizer
        optimizer = NevergradOptimizer(
            eval_config       = eval_cfg,
            param_space       = param_space,
            n_trials          = n_trials,
            n_workers         = n_workers,
            seed              = seed,
            sigma0            = sigma0,
            population_size   = pop_size,
            checkpoint_every  = ckpt_every,
            output_dir        = output_dir,
            warm_start_values = warm_start_values,
            optimizer_name    = args.optimizer,
        )
        result = optimizer.run(resume_path=args.resume)

        # print results
        col_w = 44
        print(f"\n{'='*62}")
        print(f"  Best trial #{result.best_trial_id}   score = {result.best_score:.5f}")
        print(f"{'='*62}")
        for k in sorted(result.best_params):
            print(f"  {k:<{col_w}} {result.best_params[k]:>12.6g}")
        print()

        write_best_config(base_config_path, result.best_params, output_config_path)
        return

    # ── TPE (optuna legacy, not in paper) path ───────────────────────────────
    if not _OPTUNA_AVAILABLE:
        raise SystemExit(
            "Optuna is not installed. Install with:  pip install optuna\n"
            "Or switch to a Nevergrad optimizer:  --optimizer cma"
        )

    drone_csv     = offset_cfg.get("drone_csv", "")
    optitrack_csv = offset_cfg.get("optitrack_csv", "")
    offset_s      = float(offset_cfg.get("offset_s", 0.0))
    start_gt_s    = offset_cfg.get("start_gt_s")
    end_gt_s      = offset_cfg.get("end_gt_s")

    if not os.path.isabs(drone_csv):
        drone_csv = os.path.join(wdir, drone_csv)
    if not os.path.isabs(optitrack_csv):
        optitrack_csv = os.path.join(wdir, optitrack_csv)

    pos_weight     = float(obj_cfg.get("position_weight",    1.0))
    ori_weight     = float(obj_cfg.get("orientation_weight", 0.5))
    settling_s     = float(obj_cfg.get("settling_s",         3.0))
    study_name     = str(opt_cfg.get("study_name", "ekf_sweep"))

    # ── load GT once (cached for entire sweep) ──────────────────────────────
    print("Loading OptiTrack GT …")
    print(f"  {optitrack_csv}")
    gt_data_cached = _load_ot_gt(
        optitrack_csv, ot_rotation,
        time_offset=offset_s,
        cut_before=start_gt_s,
        cut_after=end_gt_s,
        body_rotation=body_rotation if isinstance(body_rotation, np.ndarray) else None,
    )
    print(f"  {len(gt_data_cached['timestamps'])} poses  "
          f"[{gt_data_cached['timestamps'][0]:.2f} – {gt_data_cached['timestamps'][-1]:.2f} s aligned]")

    # ── detect liftoff once ─────────────────────────────────────────────────
    print("\nDetecting liftoff from IMU …")
    imu_data  = _load_drone_imu(drone_csv)
    liftoff_s = _detect_liftoff_imu(imu_data["t_s"], imu_data["accel_mag"])
    if liftoff_s is None:
        df_tmp    = _replay.load_csv(drone_csv)
        liftoff_s = float(df_tmp[df_tmp["type"] == "I"]["t_ms"].iloc[0]) / 1000.0
        print(f"  No liftoff detected — falling back to recording start ({liftoff_s:.2f} s)")
    else:
        print(f"  Liftoff at {liftoff_s:.2f} s (drone clock)")

    try:
        with open(base_config_path, encoding="utf-8") as f:
            _bcfg = yaml.safe_load(f) or {}
        flying_delay_s_base = float(_bcfg.get("flying_delay_s", 0.0))
    except Exception:
        flying_delay_s_base = 0.0

    phase2_start_s = liftoff_s + flying_delay_s_base
    eval_start_s   = phase2_start_s + settling_s
    eval_end_s     = liftoff_s + runtime_s
    print(f"  phase2_start_s = {phase2_start_s:.2f} s  (drone pos reset to 0,0,0 here)")
    print(f"  eval_start_s   = {eval_start_s:.2f} s")
    print(f"  eval_end_s     = {eval_end_s:.2f} s  (liftoff + {runtime_s:.1f} s)")

    print(f"\nAnchoring GT once at phase2_start_s = {phase2_start_s:.2f} s …")
    _base_ns = _build_replay_args(drone_csv, base_config_path, {})
    with contextlib.redirect_stdout(io.StringIO()):
        _base_data = _replay.replay(_base_ns)
    _base_t_s = _base_data["timestamps"].astype(np.float64) / 1000.0
    _base_qs  = _base_data["quaternions"].astype(np.float64)

    _p2_idx = int(np.argmin(np.abs(_base_t_s - phase2_start_s)))
    drone_anchor_pos  = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    drone_anchor_quat = tuple(float(_base_qs[_p2_idx, k]) for k in range(4))

    _gt_ts  = gt_data_cached["timestamps"].astype(np.float64)
    _gt_pos = gt_data_cached["positions"]
    _gt_qs  = gt_data_cached["quaternions"]

    _gi = int(np.argmin(np.abs(_gt_ts - phase2_start_s)))
    if abs(_gt_ts[_gi] - phase2_start_s) > 2.0:
        raise SystemExit(f"GT has no data near phase2_start_s = {phase2_start_s:.2f} s")

    gt_anchor_pos  = np.array(_gt_pos[_gi], dtype=np.float32)
    gt_anchor_quat = tuple(float(_gt_qs[_gi, k]) for k in range(4))

    gt_fixed = anchor_ground_truth(
        gt_data_cached,
        drone_anchor_pos, drone_anchor_quat,
        gt_anchor_pos, gt_anchor_quat,
    )
    print(f"  GT anchored: {len(gt_fixed['timestamps'])} poses  "
          f"[{gt_fixed['timestamps'][0]:.2f} – {gt_fixed['timestamps'][-1]:.2f} s]")

    # ── print sweep summary ──────────────────────────────────────────────────
    enabled_params = {k: v for k, v in sweep_params_cfg.items() if v.get("enabled", True)}
    col_w = 44
    print(f"\nSweeping {len(enabled_params)} parameters over {n_trials} trials")
    print(f"Objective: {pos_weight}×pos_RMSE[m] + {ori_weight}×ori_RMSE[rad]  "
          f"(eval window: t ≥ {eval_start_s:.1f} s)\n")
    print(f"  {'Parameter':<{col_w}} {'Low':>8}  {'High':>8}  {'Log'}")
    print(f"  {'-'*col_w}  {'--------'}  {'--------'}  {'---'}")
    for pname, pcfg in enabled_params.items():
        print(f"  {pname:<{col_w}} {pcfg['low']:>8.4g}  {pcfg['high']:>8.4g}  "
              f"{str(pcfg.get('log', False))}")
    print()

    # ── run optuna study ─────────────────────────────────────────────────────
    # pass raw cached GT and liftoff info so the objective can anchor per-trial
    objective = _build_objective(
        drone_csv, base_config_path, gt_data_cached, liftoff_s, flying_delay_s_base,
        eval_start_s, sweep_params_cfg, pos_weight, ori_weight, eval_end_s,
    )

    study = optuna.create_study(direction="minimize", study_name=study_name)
    study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=True)

    # ── results ──────────────────────────────────────────────────────────────
    best = study.best_trial
    print(f"\n{'='*62}")
    print(f"  Best trial #{best.number}   objective = {best.value:.5f}")
    print(f"{'='*62}")
    print(f"  {'Parameter':<{col_w}} {'Best value':>12}")
    print(f"  {'-'*col_w}  {'----------'}")
    for k in sorted(best.params):
        print(f"  {k:<{col_w}} {best.params[k]:>12.6g}")
    print()

    # top-5 table
    all_t = sorted([t for t in study.trials if t.value is not None], key=lambda t: t.value)
    param_keys = sorted(best.params.keys())
    header_vals = "  ".join(f"{k.split('.')[-1][:11]:>11}" for k in param_keys)
    print(f"  Top {min(5, len(all_t))} trials:")
    print(f"  {'#':>4}  {'Objective':>10}  {header_vals}")
    for t in all_t[:5]:
        vals = "  ".join(f"{t.params.get(k, float('nan')):>11.4g}" for k in param_keys)
        print(f"  {t.number:>4}  {t.value:>10.5f}  {vals}")
    print()

    _write_best_config(base_config_path, best.params, output_config_path)


if __name__ == "__main__":
    main()
