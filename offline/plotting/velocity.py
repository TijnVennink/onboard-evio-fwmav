"""world-frame velocity: estimate (red) vs OptiTrack GT (blue), both from
differentiated positions.

run from openmv_offline/: python "plotting/velocity.py".
"""
from __future__ import annotations
import sys
import os

# ---------------------------------------------------------------------------
# load  any offset YAML
# ---------------------------------------------------------------------------
file_path = "/home/tvennink/Desktop/thesis_code/openmv_offline/data_record/6_may/offsets_vanilla/may1800_001.yaml"
if len(sys.argv) > 1:
    file_path = sys.argv[1]

# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
out_dir = sys.argv[2] if len(sys.argv) > 2 else _HERE
clip = (float(sys.argv[3]), float(sys.argv[4])) if len(sys.argv) > 4 else None

# replay config: vanilla no-thrust plus onboard drag best
replay_config = os.path.join(_HERE, "..", "kalman_replay_config_vanilla_nothrust_best.yaml")
# sweep config supplies the eval-window/objective block
sweep_config = os.path.join(_HERE, "..", "kalman_sweep_config_vanilla_nothrust.yaml")

from plot_helpers import load_offset_evaluated, plt
import numpy as np


def compute_velocity_from_positions(timestamps: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """velocity by numerical differentiation of positions, timestamps (N,) in
    seconds and positions (N, 3), returns velocities (N, 3)."""
    # central differences for interior points, forward/backward for endpoints
    velocities = np.zeros_like(positions)
    dt = np.diff(timestamps)

    # forward difference for first point
    velocities[0] = (positions[1] - positions[0]) / dt[0]

    # central differences for interior points
    for i in range(1, len(positions) - 1):
        dt_total = timestamps[i + 1] - timestamps[i - 1]
        velocities[i] = (positions[i + 1] - positions[i - 1]) / dt_total

    # backward difference for last point
    velocities[-1] = (positions[-1] - positions[-2]) / dt[-1]
    
    return velocities


def main() -> None:
    # GT comes back already anchored into the EKF frame and trimmed to the
    # sweep's show_s eval window, on the drone clock (seconds).
    drone, gt, (plot_start_s, plot_end_s) = load_offset_evaluated(
        file_path, replay_config=replay_config, sweep_config=sweep_config, clip=clip)

    # estimate world velocity: differentiate world positions (states[:, 0:3])
    d_t = drone["t_s"]               # already seconds (drone clock)
    d_pos = drone["states"][:, 0:3]  # world positions [x, y, z]
    d_vel = compute_velocity_from_positions(d_t, d_pos)
    d_vx = d_vel[:, 0]
    d_vy = d_vel[:, 1]
    d_vz = d_vel[:, 2]

    # GT world velocity: differentiate anchored positions directly
    gt_t = gt["t_s"]
    gt_pos = gt["positions"]
    gt_vel = compute_velocity_from_positions(gt_t, gt_pos)
    
    # debug: print some sample values
    print(f"\nDebug info:")
    print(f"  Time range: drone=[{d_t.min():.2f}, {d_t.max():.2f}]s, GT=[{gt_t.min():.2f}, {gt_t.max():.2f}]s")
    print(f"  Sample count: drone={len(d_t)}, GT={len(gt_t)}")
    print(f"  Drone world velocity range: vx=[{d_vx.min():.3f}, {d_vx.max():.3f}], vy=[{d_vy.min():.3f}, {d_vy.max():.3f}], vz=[{d_vz.min():.3f}, {d_vz.max():.3f}]")
    print(f"  GT world velocity range: vx=[{gt_vel[:,0].min():.3f}, {gt_vel[:,0].max():.3f}], vy=[{gt_vel[:,1].min():.3f}, {gt_vel[:,1].max():.3f}], vz=[{gt_vel[:,2].min():.3f}, {gt_vel[:,2].max():.3f}]")
    
    g_vx = gt_vel[:, 0]
    g_vy = gt_vel[:, 1]
    g_vz = gt_vel[:, 2]

    # ---------------------------------------------------------------------------
    # world-frame velocity onto display axes. EKF world is Z-up (OptiTrack Y maps
    # to EKF Z): forward = EKF x, lateral = EKF y, up = EKF z, not body-frame.
    # ---------------------------------------------------------------------------
    plot_d = {"fwd": d_vx, "lat": d_vy, "up": d_vz}
    plot_g = {"fwd": g_vx, "lat": g_vy, "up": g_vz}

    # 3 vertically stacked subplots
    fig, axes = plt.subplots(3, 1, figsize=(6, 5), sharex=True)

    # forward subplot
    axes[0].plot(gt_t, plot_g["fwd"], color="tab:blue", linewidth=1.2, label="Motion capture")
    axes[0].plot(d_t, plot_d["fwd"], color="tab:red", linewidth=1.2, label="Estimate")
    axes[0].set_ylabel("$v_x$ [m/s]")
    axes[0].grid(True, linestyle=":", alpha=0.7)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].set_title("World-frame velocity")

    # lateral subplot
    axes[1].plot(gt_t, plot_g["lat"], color="tab:blue", linewidth=1.2)
    axes[1].plot(d_t, plot_d["lat"], color="tab:red", linewidth=1.2)
    axes[1].set_ylabel("$v_y$ [m/s]")
    axes[1].grid(True, linestyle=":", alpha=0.7)

    # up subplot
    axes[2].plot(gt_t, plot_g["up"], color="tab:blue", linewidth=1.2)
    axes[2].plot(d_t, plot_d["up"], color="tab:red", linewidth=1.2)
    axes[2].set_ylabel("$v_z$ [m/s]")
    axes[2].set_xlabel("t [s]")
    axes[2].grid(True, linestyle=":", alpha=0.7)

    plt.tight_layout()
    
    # save first, then show (so debug output appears even if window is closed quickly)
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, "velocity.pdf")
    plt.savefig(output_path, bbox_inches="tight")
    print(f"\nPlot saved to: {output_path}")
    
    plt.show()


if __name__ == "__main__":
    main()