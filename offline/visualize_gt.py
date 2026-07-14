from __future__ import annotations
import sys
import argparse
import numpy as np

from helpers import load_optitrack_ground_truth as _load_optitrack_ground_truth, rotate_vector_by_quaternion

optitrack_csv = "/home/tvennink/Desktop/thesis_code/openmv_offline/data_record/6_may/optitrack/cftijn_may6_1715_cut.csv"

DEFAULT_OPTITRACK_TO_FLAPPER_WORLD = np.array(
    [0, 0, 1,
     1, 0, 0,
     0, 1, 0], dtype=np.float32).reshape(3, 3)

#optitrack x -> flapper y -> for 
#optitrack y -> flapper z
#optrack z -> flapper x


def load_optitrack_ground_truth(path: str, cut_before: float | None = None, cut_after: float | None = None) -> dict[str, np.ndarray]:
    return _load_optitrack_ground_truth(path, DEFAULT_OPTITRACK_TO_FLAPPER_WORLD, cut_before=cut_before, cut_after=cut_after)


def plot_ground_truth_rerun(gt_data: dict[str, np.ndarray]) -> None:
    try:
        import rerun as rr
    except ImportError:
        print("rerun-sdk not found. Install with: pip install rerun-sdk")
        return

    rr.init("optitrack_ground_truth", spawn=True)

    timestamps = gt_data["timestamps"]
    positions = gt_data["positions"]
    quaternions = gt_data["quaternions"]
    marker_positions = gt_data.get("marker_positions")
    marker_names = gt_data.get("marker_names")
    unlabeled_marker_positions = gt_data.get("unlabeled_marker_positions")
    unlabeled_marker_names = gt_data.get("unlabeled_marker_names")
    axis_length = 0.1
    axis_color = np.array([[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255]], dtype=np.uint8)

    # world axes at the origin and a small ground plane for reference
    world_axis_length = 0.2
    world_origins = np.tile(np.array([0.0, 0.0, 0.0], dtype=np.float32), (3, 1))
    world_axes = np.array([
        [world_axis_length, 0.0, 0.0],
        [0.0, world_axis_length, 0.0],
        [0.0, 0.0, world_axis_length],
    ], dtype=np.float32)
    rr.log("world/axes", rr.Arrows3D(
        vectors=world_axes,
        origins=world_origins,
        colors=axis_color,
        radii=0.01,
    ))

    plane_points = []
    plane_size = 0.5
    plane_steps = 5
    for xi in np.linspace(-plane_size, plane_size, plane_steps):
        for yi in np.linspace(-plane_size, plane_size, plane_steps):
            plane_points.append([xi, yi, 0.0])
    rr.log("world/ground_plane", rr.Points3D(
        np.array(plane_points, dtype=np.float32),
        radii=0.005,
        colors=np.tile(np.array([[200, 200, 200, 100]], dtype=np.uint8), (len(plane_points), 1)),
    ))

    trajectory_pts: list = []   # growing list of [x,y,z] for line strips

    for i in range(len(timestamps)):
        t_s = float(timestamps[i])
        rr.set_time_seconds("time", t_s)

        x, y, z = positions[i]
        w, qx, qy, qz = quaternions[i]

        rr.log("ground_truth/trajectory", rr.Points3D([[x, y, z]], radii=0.01))

        trajectory_pts.append([x, y, z])
        if len(trajectory_pts) >= 2:
            traj_arr = np.array(trajectory_pts, dtype=np.float32)
            rr.log("ground_truth/trajectory_line",
                   rr.LineStrips3D([traj_arr]))
            rr.log("ground_truth/trajectory_xz",
                   rr.LineStrips2D([traj_arr[:, [0, 2]]]))
            rr.log("ground_truth/trajectory_xy",
                   rr.LineStrips2D([traj_arr[:, [0, 1]]]))
        rr.log("ground_truth/pose", rr.Transform3D(
            translation=[x, y, z],
            rotation=rr.Quaternion(xyzw=[qx, qy, qz, w]),
        ))

        origins = np.tile([x, y, z], (3, 1)).astype(np.float32)
        axes = np.vstack([
            rotate_vector_by_quaternion((w, qx, qy, qz), (axis_length, 0.0, 0.0)),
            rotate_vector_by_quaternion((w, qx, qy, qz), (0.0, axis_length, 0.0)),
            rotate_vector_by_quaternion((w, qx, qy, qz), (0.0, 0.0, axis_length)),
        ]).astype(np.float32)

        rr.log("ground_truth/pose_axes", rr.Arrows3D(
            vectors=axes,
            origins=origins,
            colors=axis_color,
            radii=0.01,
        ))

        if marker_positions is not None and marker_positions.shape[0] > i:
            pts = marker_positions[i]
            labels = [str(name) for name in marker_names] if marker_names is not None else None
            rr.log("ground_truth/markers", rr.Points3D(
                pts,
                radii=0.02,
                labels=labels,
            ))

        if unlabeled_marker_positions is not None and len(unlabeled_marker_positions) > i:
            pts = unlabeled_marker_positions[i]
            if pts:
                pts = np.array(pts, dtype=np.float32)
                colors = np.tile(np.array([[255, 0, 0, 255]], dtype=np.uint8), (len(pts), 1))
                rr.log("ground_truth/unlabeled_markers", rr.Points3D(
                    pts,
                    radii=0.02,
                    colors=colors,
                ))

    print(f"Logged {len(timestamps)} ground truth poses to rerun.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualise OptiTrack ground truth in rerun.")
    parser.add_argument("--csv", type=str, default=optitrack_csv,
                        help="Path to the OptiTrack CSV file")
    parser.add_argument("--start_s", type=float, default=None,
                        help="Trim data: discard rows with timestamp < start_s")
    parser.add_argument("--end_s", type=float, default=None,
                        help="Trim data: discard rows with timestamp > end_s")
    args = parser.parse_args()

    if not args.csv:
        print("Please set optitrack_csv or pass --csv <path>.")
        sys.exit(1)

    gt = load_optitrack_ground_truth(
        args.csv,
        cut_before=args.start_s,
        cut_after=args.end_s,
    )
    plot_ground_truth_rerun(gt)