#!/usr/bin/env python3
"""
attitude plot: yaw, pitch, roll from the EKF estimate vs OptiTrack GT.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from plot_helpers import load_offset_evaluated, plt


def quaternion_to_euler(q):
    """
    convert quaternion (w, x, y, z) to euler angles (roll, pitch, yaw) in radians.
    """
    if q.ndim == 1:
        # single quaternion, normalise to guard against floating-point drift
        q = q / (np.linalg.norm(q) + 1e-12)
        w, x, y, z = q

        # roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        sinp = np.clip(sinp, -1.0, 1.0)
        pitch = np.arcsin(sinp)
        
        # yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return np.array([roll, pitch, yaw])
    else:
        # array of quaternions, normalise each row to guard against drift
        q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
        w = q[:, 0]
        x = q[:, 1]
        y = q[:, 2]
        z = q[:, 3]

        # roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        sinp = np.clip(sinp, -1.0, 1.0)
        pitch = np.arcsin(sinp)
        
        # yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return np.column_stack([roll, pitch, yaw])


def main():
    # point at any offset YAML; all other values (rotations, flying_delay_s,
    # fallback_rho, anchor_delay_s, show_s window) are loaded per-file.
    file_path = "/home/tvennink/Desktop/thesis_code/openmv_offline/data_record/6_may/offsets_vanilla/may1800_001.yaml"
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)
    clip = (float(sys.argv[3]), float(sys.argv[4])) if len(sys.argv) > 4 else None

    # replay config: vanilla no-thrust plus onboard drag best
    _pdir = os.path.dirname(os.path.abspath(__file__))
    replay_config = os.path.join(_pdir, "..", "kalman_replay_config_vanilla_nothrust_best.yaml")
    # sweep config supplies the eval-window/objective block
    sweep_config = os.path.join(_pdir, "..", "kalman_sweep_config_vanilla_nothrust.yaml")

    # GT comes back already anchored into the EKF frame and trimmed to the
    # sweep's show_s eval window, so euler axes line up with the EKF (no remap).
    drone, gt, (plot_start_s, plot_end_s) = load_offset_evaluated(
        file_path, replay_config=replay_config, sweep_config=sweep_config, clip=clip)

    # time arrays (already in seconds, drone clock)
    d_t  = drone["t_s"]
    gt_t = gt["t_s"]

    # convert quaternions to euler angles (in degrees)
    d_quat = drone["quaternions"]
    d_euler = quaternion_to_euler(d_quat)
    d_roll = np.rad2deg(d_euler[:, 0])
    d_pitch = np.rad2deg(d_euler[:, 1])
    d_yaw = np.rad2deg(d_euler[:, 2])

    gt_quat = gt["quaternions"]
    gt_euler = quaternion_to_euler(gt_quat)
    # GT is anchored into the EKF frame, so each euler channel already matches
    # its EKF counterpart, plot directly with no remap.
    gt_roll  = np.rad2deg(gt_euler[:, 0])
    gt_pitch = np.rad2deg(gt_euler[:, 1])
    gt_yaw   = np.rad2deg(gt_euler[:, 2])

    # debug output
    print("\nDebug info:")
    print(f"  Time range: drone=[{d_t[0]:.2f}, {d_t[-1]:.2f}]s, GT=[{gt_t[0]:.2f}, {gt_t[-1]:.2f}]s")
    print(f"  Sample count: drone={len(d_t)}, GT={len(gt_t)}")
    print(f"  Drone attitude range: roll=[{d_roll.min():.1f}, {d_roll.max():.1f}]°, "
          f"pitch=[{d_pitch.min():.1f}, {d_pitch.max():.1f}]°, yaw=[{d_yaw.min():.1f}, {d_yaw.max():.1f}]°")
    print(f"  GT attitude range: roll=[{gt_roll.min():.1f}, {gt_roll.max():.1f}]°, "
          f"pitch=[{gt_pitch.min():.1f}, {gt_pitch.max():.1f}]°, yaw=[{gt_yaw.min():.1f}, {gt_yaw.max():.1f}]°")
    
    # figure/style mirrors velocity.py: same size, one legend on the top subplot.
    fig, axes = plt.subplots(3, 1, figsize=(6, 5), sharex=True)

    # roll subplot
    axes[0].plot(gt_t, gt_roll, color="tab:blue", linewidth=1.2, label="Motion capture")
    axes[0].plot(d_t, d_roll, color="tab:red", linewidth=1.2, label="Estimate")
    axes[0].set_ylabel('Roll [°]')
    axes[0].grid(True, linestyle=":", alpha=0.7)
    # extra y-headroom so the legend sits in clear space above the data
    y0, y1 = axes[0].get_ylim()
    axes[0].set_ylim(y0, y1 + 0.35 * (y1 - y0))
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].set_title("Attitude (body w.r.t. world)")

    # pitch subplot
    axes[1].plot(gt_t, gt_pitch, color="tab:blue", linewidth=1.2)
    axes[1].plot(d_t, d_pitch, color="tab:red", linewidth=1.2)
    axes[1].set_ylabel('Pitch [°]')
    axes[1].grid(True, linestyle=":", alpha=0.7)

    # yaw subplot
    axes[2].plot(gt_t, gt_yaw, color="tab:blue", linewidth=1.2)
    axes[2].plot(d_t, d_yaw, color="tab:red", linewidth=1.2)
    axes[2].set_ylabel('Yaw [°]')
    axes[2].set_xlabel("t [s]")
    axes[2].grid(True, linestyle=":", alpha=0.7)

    plt.tight_layout()

    # save figure
    output_path = os.path.join(out_dir, "attitude.pdf")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    
    plt.show()


if __name__ == "__main__":
    main()
