"""3D trajectory comparison: EVIO estimate (red) vs OptiTrack GT (blue).

run from openmv_offline/: python "plotting/tracked trajectories_3D.py".
"""
from __future__ import annotations
import sys
import os

# ---------------------------------------------------------------------------
# point at any offset yaml from dataset; all other values (rotations, flying_delay_s,
# fallback_rho, anchor_delay_s, show_s window) are loaded per-file.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

file_path = "/home/tvennink/Desktop/thesis_code/openmv_offline/data_record/6_may/offsets_vanilla/may1800_000.yaml"
if len(sys.argv) > 1:
    file_path = sys.argv[1]
out_dir   = sys.argv[2] if len(sys.argv) > 2 else _HERE
clip      = (float(sys.argv[3]), float(sys.argv[4])) if len(sys.argv) > 4 else None
save_name = os.path.splitext(os.path.basename(file_path))[0]  # offset stem

# replay config: vanilla no-thrust plus onboard drag best
replay_config = os.path.join(_HERE, "..", "kalman_replay_config_vanilla_nothrust_best.yaml")
# sweep config supplies the eval-window/objective block
sweep_config = os.path.join(_HERE, "..", "kalman_sweep_config_vanilla_nothrust.yaml")

plot_2Dxy = True  # if True, also plot 2D top-down view (x vs y)
plot_2Dxz = True  # also plot 2D side view (x z)
plot_2Dzy = True  # also plot 2D side view (z y)

MARK_EVERY_S = float(os.environ.get("TRAJ_MARK_S", 10.0))

# ---------------------------------------------------------------------------
from plot_helpers import load_offset_evaluated, plt
import numpy as np


def _add_time_marks(ax, d_t, d_dims, g_t, g_dims, label_first=True):
    """matched time markers plus connectors, d_dims/g_dims are tuples of coord
    arrays (2 for 2D axes, 3 for 3D axes)."""
    t0 = max(d_t[0], g_t[0])
    t1 = min(d_t[-1], g_t[-1])
    marks = np.arange(np.ceil(t0 / MARK_EVERY_S) * MARK_EVERY_S, t1, MARK_EVERY_S)
    for k, tm in enumerate(marks):
        di = int(np.argmin(np.abs(d_t - tm)))
        gi = int(np.argmin(np.abs(g_t - tm)))
        dp = [c[di] for c in d_dims]
        gp = [c[gi] for c in g_dims]
        ax.plot(*zip(dp, gp), color="0.35", lw=0.7, zorder=3)
        ax.plot(*[[v] for v in gp], "o", color="tab:blue", ms=4,
                mec="white", mew=0.5, zorder=4)
        ax.plot(*[[v] for v in dp], "o", color="tab:red", ms=4,
                mec="white", mew=0.5, zorder=4)
        if label_first:
            ax.text(*gp, f" {tm:.0f}s", fontsize=6, color="0.25", zorder=5)


def main() -> None:
    # GT comes back anchored into the EKF frame and trimmed to the sweep's
    # show_s eval window. EKF world is Z-up: forward=x, lateral=y, up=z.
    drone, gt, (plot_start_s, plot_end_s) = load_offset_evaluated(
        file_path, replay_config=replay_config, sweep_config=sweep_config, clip=clip)
    os.makedirs(out_dir, exist_ok=True)

    # estimated position: states[:, 0:3] = x, y, z
    d_x = drone["states"][:, 0]
    d_y = drone["states"][:, 1]
    d_z = drone["states"][:, 2]

    # GT position
    g_x = gt["positions"][:, 0]
    g_y = gt["positions"][:, 1]
    g_z = gt["positions"][:, 2]

    fig = plt.figure(figsize=(5, 4))
    ax  = fig.add_subplot(111, projection="3d")

    d_t, gt_t = drone["t_s"], gt["t_s"]

    ax.plot(g_x, g_y, g_z, color="tab:blue", linewidth=1.2, label="Ground truth", alpha=0.85)
    ax.plot(d_x, d_y, d_z, color="tab:red",  linewidth=1.2, label="Estimated", alpha=0.85)
    _add_time_marks(ax, d_t, (d_x, d_y, d_z), gt_t, (g_x, g_y, g_z))

    ax.set_xlabel("forward [m]")
    ax.set_ylabel("lateral [m]")
    ax.set_zlabel("up [m]")
    ax.set_title("Robot position (world frame)")

    # style pane walls (light gray with white grid)
    pane_color = (0.92, 0.92, 0.92, 1.0)
    ax.xaxis.pane.fill = True
    ax.yaxis.pane.fill = True
    ax.zaxis.pane.fill = True
    ax.xaxis.pane.set_facecolor(pane_color)
    ax.yaxis.pane.set_facecolor(pane_color)
    ax.zaxis.pane.set_facecolor(pane_color)
    ax.xaxis.pane.set_edgecolor("white")
    ax.yaxis.pane.set_edgecolor("white")
    ax.zaxis.pane.set_edgecolor("white")
    ax.grid(True, color="white", linewidth=0.8)

    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"tracked_trajectories_3D_{save_name}.pdf"), bbox_inches="tight")
    plt.show()

    # ------------------------------------------------------------------
    # 2D plots
    # ------------------------------------------------------------------
    def _style_2d_ax(ax2: "plt.Axes") -> None:
        ax2.set_facecolor((0.92, 0.92, 0.92, 1.0))
        ax2.grid(True, color="white", linewidth=0.8)
        for spine in ax2.spines.values():
            spine.set_edgecolor("white")

    if plot_2Dxy:
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        ax2.plot(g_x, g_y, color="tab:blue", linewidth=1.2, label="Ground truth", alpha=0.85)
        ax2.plot(d_x, d_y, color="tab:red",  linewidth=1.2, label="Estimated", alpha=0.85)
        _add_time_marks(ax2, d_t, (d_x, d_y), gt_t, (g_x, g_y))
        ax2.set_xlabel("forward [m]")
        ax2.set_ylabel("lateral [m]")
        ax2.set_title("Robot position (top-down: forward–lateral, world frame)")
        _style_2d_ax(ax2)
        ax2.legend(loc="upper right", fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"tracked_trajectories_2Dxy_{save_name}.pdf"), bbox_inches="tight")
        plt.show()

    if plot_2Dxz:
        fig3, ax3 = plt.subplots(figsize=(5, 4))
        ax3.plot(g_x, g_z, color="tab:blue", linewidth=1.2, label="Ground truth", alpha=0.85)
        ax3.plot(d_x, d_z, color="tab:red",  linewidth=1.2, label="Estimated", alpha=0.85)
        _add_time_marks(ax3, d_t, (d_x, d_z), gt_t, (g_x, g_z))
        ax3.set_xlabel("forward [m]")
        ax3.set_ylabel("up [m]")
        ax3.set_title("Robot position (side: forward–up, world frame)")
        _style_2d_ax(ax3)
        ax3.legend(loc="upper right", fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"tracked_trajectories_2Dxz_{save_name}.pdf"), bbox_inches="tight")
        plt.show()

    if plot_2Dzy:
        fig4, ax4 = plt.subplots(figsize=(5, 4))
        ax4.plot(g_z, g_y, color="tab:blue", linewidth=1.2, label="Ground truth", alpha=0.85)
        ax4.plot(d_z, d_y, color="tab:red",  linewidth=1.2, label="Estimated", alpha=0.85)
        _add_time_marks(ax4, d_t, (d_z, d_y), gt_t, (g_z, g_y))
        ax4.set_xlabel("up [m]")
        ax4.set_ylabel("lateral [m]")
        ax4.set_title("Robot position (side: up–lateral, world frame)")
        _style_2d_ax(ax4)
        ax4.legend(loc="upper right", fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"tracked_trajectories_2Dzy_{save_name}.pdf"), bbox_inches="tight")
        plt.show()


if __name__ == "__main__":
    main()