from __future__ import annotations
import csv
import numpy as np


def quaternion_from_rotation_matrix(matrix: np.ndarray) -> tuple[float, float, float, float]:
    m = matrix
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    else:
        if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s
    return (float(w), float(x), float(y), float(z))


def quaternion_multiply(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def rotate_vector_by_quaternion(quaternion: tuple[float, float, float, float], vector: tuple[float, float, float]) -> np.ndarray:
    w, x, y, z = quaternion
    vx, vy, vz = vector
    q_vec = np.array([x, y, z], dtype=np.float32)
    v = np.array([vx, vy, vz], dtype=np.float32)
    t = 2.0 * np.cross(q_vec, v)
    return v + w * t + np.cross(q_vec, t)


def quaternion_conjugate(quaternion: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    w, x, y, z = quaternion
    return (float(w), float(-x), float(-y), float(-z))


def _extract_optitrack_marker_columns(header_rows: list[list[str]], data_header: list[str], named_prefix: str = "cftijn:Marker", unlabeled_prefix: str = "Unlabeled") -> tuple[list[str], list[list[int]], list[str], list[list[int]]]:
    name_row = next(
        (
            row for row in header_rows
            if len(row) >= 2 and (row[0].strip().lower() == "name" or row[1].strip().lower() == "name")
        ),
        None,
    )
    if name_row is None:
        return [], [], [], []

    named_indices: dict[str, list[int]] = {}
    unlabeled_indices: dict[str, list[int]] = {}
    for idx, name in enumerate(name_row):
        if not name:
            continue
        if idx >= len(data_header):
            continue
        axis = data_header[idx].strip().upper()
        if axis not in {"X", "Y", "Z"}:
            continue
        if name.startswith(named_prefix):
            named_indices.setdefault(name, []).append(idx)
        elif name.startswith(unlabeled_prefix):
            unlabeled_indices.setdefault(name, []).append(idx)

    def build_marker_groups(indices_dict: dict[str, list[int]]) -> tuple[list[str], list[list[int]]]:
        names: list[str] = []
        positions: list[list[int]] = []
        for name, indices in sorted(indices_dict.items(), key=lambda item: min(item[1])):
            indices = sorted(indices)
            if len(indices) < 3:
                continue
            selected = indices[:3]
            if selected[1] != selected[0] + 1 or selected[2] != selected[1] + 1:
                for start in range(len(indices) - 2):
                    candidate = indices[start:start + 3]
                    if candidate[1] == candidate[0] + 1 and candidate[2] == candidate[1] + 1:
                        selected = candidate
                        break
                else:
                    continue
            names.append(name)
            positions.append(selected)
        return names, positions

    named_names, named_positions = build_marker_groups(named_indices)
    unlabeled_names, unlabeled_positions = build_marker_groups(unlabeled_indices)
    return named_names, named_positions, unlabeled_names, unlabeled_positions


def quaternion_normalize(quaternion: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    q = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm <= 0.0:
        raise ValueError("Cannot normalize a zero-length quaternion")
    q /= norm
    return (float(q[0]), float(q[1]), float(q[2]), float(q[3]))


def anchor_ground_truth(gt_data: dict[str, np.ndarray], state_position: np.ndarray, state_quaternion: tuple[float, float, float, float], gt_anchor_position: np.ndarray, gt_anchor_quaternion: tuple[float, float, float, float]) -> dict[str, np.ndarray]:
    normalized_state_quat = quaternion_normalize(state_quaternion)
    normalized_gt_quat = quaternion_normalize(gt_anchor_quaternion)
    rel_quat = quaternion_multiply(normalized_state_quat, quaternion_conjugate(normalized_gt_quat))

    anchored_positions = []
    gt_anchor_rotated = rotate_vector_by_quaternion(rel_quat, tuple(gt_anchor_position))
    translation = np.asarray(state_position, dtype=np.float32) - gt_anchor_rotated

    for position in gt_data["positions"]:
        anchored_position = rotate_vector_by_quaternion(rel_quat, tuple(position)) + translation
        anchored_positions.append(anchored_position)

    anchored_quaternions = [quaternion_multiply(rel_quat, tuple(q)) for q in gt_data["quaternions"]]

    anchored_data = {
        "timestamps": gt_data["timestamps"],
        "positions": np.asarray(anchored_positions, dtype=np.float32),
        "quaternions": np.asarray(anchored_quaternions, dtype=np.float32),
    }

    if "marker_positions" in gt_data:
        anchored_marker_positions = []
        for row in gt_data["marker_positions"]:
            anchored_marker_positions.append([
                rotate_vector_by_quaternion(rel_quat, tuple(pos)) + translation
                for pos in row
            ])
        anchored_data["marker_positions"] = np.asarray(anchored_marker_positions, dtype=np.float32)

    if "marker_names" in gt_data:
        anchored_data["marker_names"] = gt_data["marker_names"]

    if "unlabeled_marker_positions" in gt_data:
        anchored_unlabeled_marker_positions = []
        for row in gt_data["unlabeled_marker_positions"]:
            anchored_unlabeled_marker_positions.append([
                rotate_vector_by_quaternion(rel_quat, tuple(pos)) + translation
                for pos in row
            ])
        anchored_data["unlabeled_marker_positions"] = anchored_unlabeled_marker_positions

    if "unlabeled_marker_names" in gt_data:
        anchored_data["unlabeled_marker_names"] = gt_data["unlabeled_marker_names"]

    return anchored_data


def fit_yaw_translation(src_pos: np.ndarray, dst_pos: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """least-squares 4-DOF alignment: yaw about world-z plus translation, src to dst."""
    src = np.asarray(src_pos, dtype=np.float64)
    dst = np.asarray(dst_pos, dtype=np.float64)
    mx, my = src.mean(0), dst.mean(0)
    Xc, Yc = src - mx, dst - my
    num = float(np.sum(Xc[:, 0] * Yc[:, 1] - Xc[:, 1] * Yc[:, 0]))
    den = float(np.sum(Xc[:, 0] * Yc[:, 0] + Xc[:, 1] * Yc[:, 1]))
    psi = float(np.arctan2(num, den))
    c, s = np.cos(psi), np.sin(psi)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    t = my - R @ mx
    return R, t, psi


def yaw_quaternion(psi: float) -> tuple[float, float, float, float]:
    """quaternion (w,x,y,z) for a rotation by ψ about world-z."""
    return (float(np.cos(psi / 2.0)), 0.0, 0.0, float(np.sin(psi / 2.0)))


def anchor_ground_truth_4dof(gt_data: dict[str, np.ndarray], ref_t_s: np.ndarray,
                             ref_pos: np.ndarray) -> dict[str, np.ndarray]:
    """trajectory-wide 4-DOF (yaw+translation) alignment of GT onto a reference."""
    gt_t = np.asarray(gt_data["timestamps"], dtype=np.float64)
    gt_p = np.asarray(gt_data["positions"], dtype=np.float64)
    gt_q = np.asarray(gt_data["quaternions"], dtype=np.float64)
    ref_t = np.asarray(ref_t_s, dtype=np.float64)
    ref_p = np.asarray(ref_pos, dtype=np.float64)

    lo, hi = max(gt_t[0], ref_t[0]), min(gt_t[-1], ref_t[-1])
    m = (gt_t >= lo) & (gt_t <= hi)
    if int(m.sum()) < 5:
        return dict(gt_data)   # not enough overlap, leave GT untouched
    ref_on_gt = np.column_stack([np.interp(gt_t[m], ref_t, ref_p[:, k]) for k in range(3)])
    R, t, psi = fit_yaw_translation(gt_p[m], ref_on_gt)   # gt to ref

    qy = yaw_quaternion(psi)
    out = dict(gt_data)
    out["positions"]   = ((R @ gt_p.T).T + t).astype(np.float32)
    out["quaternions"] = np.array([quaternion_multiply(qy, tuple(q)) for q in gt_q],
                                  dtype=np.float32)
    if gt_data.get("marker_positions") is not None:
        mp = np.asarray(gt_data["marker_positions"], dtype=np.float64)   # shape (N, M, 3)
        out["marker_positions"] = (((mp.reshape(-1, 3) @ R.T) + t)
                                   .reshape(mp.shape).astype(np.float32))
    if gt_data.get("unlabeled_marker_positions") is not None:
        out["unlabeled_marker_positions"] = [
            [list((R @ np.asarray(p, dtype=np.float64)) + t) for p in row]
            for row in gt_data["unlabeled_marker_positions"]
        ]
    return out


def transform_ground_truth_frame(gt_data: dict[str, np.ndarray], frame_rotation: np.ndarray) -> dict[str, np.ndarray]:
    q_rot = quaternion_from_rotation_matrix(frame_rotation)

    transformed_positions = []
    for position in gt_data["positions"]:
        transformed_pos = frame_rotation @ np.asarray(position, dtype=np.float32)
        transformed_positions.append(transformed_pos)

    transformed_quaternions = [quaternion_multiply(q_rot, tuple(q)) for q in gt_data["quaternions"]]

    transformed_data = {
        "timestamps": gt_data["timestamps"],
        "positions": np.asarray(transformed_positions, dtype=np.float32),
        "quaternions": np.asarray(transformed_quaternions, dtype=np.float32),
    }

    if "marker_positions" in gt_data:
        transformed_marker_positions = []
        for row in gt_data["marker_positions"]:
            transformed_marker_positions.append([
                frame_rotation @ np.asarray(pos, dtype=np.float32)
                for pos in row
            ])
        transformed_data["marker_positions"] = np.asarray(transformed_marker_positions, dtype=np.float32)

    if "marker_names" in gt_data:
        transformed_data["marker_names"] = gt_data["marker_names"]

    if "unlabeled_marker_positions" in gt_data:
        transformed_unlabeled_marker_positions = []
        for row in gt_data["unlabeled_marker_positions"]:
            transformed_unlabeled_marker_positions.append([
                frame_rotation @ np.asarray(pos, dtype=np.float32)
                for pos in row
            ])
        transformed_data["unlabeled_marker_positions"] = transformed_unlabeled_marker_positions

    if "unlabeled_marker_names" in gt_data:
        transformed_data["unlabeled_marker_names"] = gt_data["unlabeled_marker_names"]

    return transformed_data


def load_optitrack_ground_truth(path: str, optitrack_rotation: np.ndarray, time_offset: float = 0.0, cut_before: float | None = None, cut_after: float | None = None, body_rotation: np.ndarray | None = None) -> dict[str, np.ndarray]:
    timestamps: list[float] = []
    positions: list[tuple[float, float, float]] = []
    quaternions: list[tuple[float, float, float, float]] = []
    marker_positions: list[list[tuple[float, float, float]]] = []
    unlabeled_marker_positions: list[list[tuple[float, float, float]]] = []

    q_rot = quaternion_from_rotation_matrix(optitrack_rotation)
    q_body = quaternion_from_rotation_matrix(body_rotation) if body_rotation is not None else None

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = None
        header_rows: list[list[str]] = []

        for row in reader:
            if not row:
                continue
            stripped = [c.strip() for c in row]
            if len(stripped) >= 2 and (stripped[0].lower() == "frame" or stripped[1].lower() == "frame"):
                header = stripped
                break
            header_rows.append(stripped)

        if header is None:
            raise ValueError(f"Could not find header row in ground truth CSV: {path}")

        time_idx = next((i for i, value in enumerate(header) if value.lower() == "time (seconds)"), None)
        if time_idx is None:
            raise ValueError(f"Could not find Time (Seconds) column in ground truth CSV: {path}")

        marker_names, marker_columns, unlabeled_marker_names, unlabeled_marker_columns = _extract_optitrack_marker_columns(header_rows, header)
        all_marker_idxs = [idx for marker in marker_columns + unlabeled_marker_columns for idx in marker]
        max_marker_idx = max(all_marker_idxs, default=-1)
        pose_start = time_idx + 1

        for row in reader:
            if not row:
                continue
            columns = [c.strip() for c in row]
            if len(columns) <= max(pose_start + 6, max_marker_idx):
                continue
            try:
                raw_timestamp = float(columns[time_idx])
                if cut_before is not None and raw_timestamp < cut_before:
                    continue
                if cut_after is not None and raw_timestamp > cut_after:
                    continue
                timestamp = raw_timestamp + time_offset
                qx = float(columns[pose_start])
                qy = float(columns[pose_start + 1])
                qz = float(columns[pose_start + 2])
                qw = float(columns[pose_start + 3])
                x = float(columns[pose_start + 4])
                y = float(columns[pose_start + 5])
                z = float(columns[pose_start + 6])

                marker_position_set: list[tuple[float, float, float]] = []
                for marker_cols in marker_columns:
                    mx = float(columns[marker_cols[0]])
                    my = float(columns[marker_cols[1]])
                    mz = float(columns[marker_cols[2]])
                    marker_position_set.append((mx, my, mz))

                unlabeled_marker_position_set: list[tuple[float, float, float]] = []
                for marker_cols in unlabeled_marker_columns:
                    try:
                        mx = columns[marker_cols[0]]
                        my = columns[marker_cols[1]]
                        mz = columns[marker_cols[2]]
                        if mx == "" or my == "" or mz == "":
                            continue
                        unlabeled_marker_position_set.append((float(mx), float(my), float(mz)))
                    except (IndexError, ValueError):
                        continue
            except ValueError:
                continue

            position = optitrack_rotation @ np.array([x, y, z], dtype=np.float32)
            quaternion = quaternion_normalize(quaternion_multiply(q_rot, (qw, qx, qy, qz)))
            if q_body is not None:
                quaternion = quaternion_normalize(quaternion_multiply(quaternion, q_body))
            rotated_marker_position_set: list[tuple[float, float, float]] = []
            for mx, my, mz in marker_position_set:
                marker_pos = optitrack_rotation @ np.array([mx, my, mz], dtype=np.float32)
                rotated_marker_position_set.append((float(marker_pos[0]), float(marker_pos[1]), float(marker_pos[2])))

            rotated_unlabeled_marker_position_set: list[tuple[float, float, float]] = []
            for mx, my, mz in unlabeled_marker_position_set:
                marker_pos = optitrack_rotation @ np.array([mx, my, mz], dtype=np.float32)
                rotated_unlabeled_marker_position_set.append((float(marker_pos[0]), float(marker_pos[1]), float(marker_pos[2])))

            timestamps.append(timestamp)
            quaternions.append(quaternion)
            positions.append((float(position[0]), float(position[1]), float(position[2])))
            marker_positions.append(rotated_marker_position_set)
            unlabeled_marker_positions.append(rotated_unlabeled_marker_position_set)

    if not timestamps:
        raise ValueError(f"No valid ground truth poses found in: {path}")

    return {
        "timestamps": np.array(timestamps, dtype=np.float64),
        "positions": np.array(positions, dtype=np.float32),
        "quaternions": np.array(quaternions, dtype=np.float32),
        "marker_positions": np.asarray(marker_positions, dtype=np.float32) if marker_positions else np.zeros((0, 0, 3), dtype=np.float32),
        "marker_names": np.asarray(marker_names, dtype=object),
        "unlabeled_marker_positions": unlabeled_marker_positions,
        "unlabeled_marker_names": np.asarray(unlabeled_marker_names, dtype=object),
    }


def shift_timestamps(data: dict[str, np.ndarray], offset_s: float) -> dict[str, np.ndarray]:
    return {
        "timestamps": data["timestamps"] + offset_s,
        "positions": data["positions"],
        "quaternions": data["quaternions"],
    }


# ---------------------------------------------------------------------------
# time-synchronisation helpers (used by sync_times.py)
# ---------------------------------------------------------------------------

def load_drone_imu(path: str) -> dict[str, np.ndarray]:
    """load only the IMU rows (type='I') from a drone recording CSV.
    returns a dict of SI-unit arrays on the same time axis (t_s, accel, gyro, and their magnitudes)."""
    _MG_TO_MS2 = 9.81 / 1000.0
    _DPS_TO_RADS = np.pi / 180.0

    t_ms_list: list[float] = []
    ax_list:   list[float] = []
    ay_list:   list[float] = []
    az_list:   list[float] = []
    gx_list:   list[float] = []
    gy_list:   list[float] = []
    gz_list:   list[float] = []

    with open(path, newline="", encoding="utf-8") as f:
        # skip comment lines (start with '#') before passing to DictReader
        lines = [line for line in f if not line.lstrip().startswith("#")]
    reader = csv.DictReader(lines)
    for row in reader:
        if row.get("type", "").strip().upper() != "I":
            continue
        try:
            t_ms_list.append(float(row["t_ms"]))
            ax_list.append(float(row["f1"]))
            ay_list.append(float(row["f2"]))
            az_list.append(float(row["f3"]))
            gx_list.append(float(row["f4"]))
            gy_list.append(float(row["f5"]))
            gz_list.append(float(row["f6"]))
        except (ValueError, KeyError):
            continue

    t_s   = np.array(t_ms_list, dtype=np.float64) / 1000.0
    ax    = np.array(ax_list, dtype=np.float32) * _MG_TO_MS2
    ay    = np.array(ay_list, dtype=np.float32) * _MG_TO_MS2
    az    = np.array(az_list, dtype=np.float32) * _MG_TO_MS2
    gx    = np.array(gx_list, dtype=np.float32) * _DPS_TO_RADS
    gy    = np.array(gy_list, dtype=np.float32) * _DPS_TO_RADS
    gz    = np.array(gz_list, dtype=np.float32) * _DPS_TO_RADS
    accel = np.sqrt(ax**2 + ay**2 + az**2).astype(np.float32)
    gyro  = np.sqrt(gx**2 + gy**2 + gz**2).astype(np.float32)

    return {
        "t_s":       t_s,
        "ax_ms2":    ax,
        "ay_ms2":    ay,
        "az_ms2":    az,
        "gx_rads":   gx,
        "gy_rads":   gy,
        "gz_rads":   gz,
        "accel_mag": accel,
        "gyro_mag":  gyro,
    }


def detect_liftoff_imu(
    t_s: np.ndarray,
    accel_mag: np.ndarray,
    window_s: float = 0.5,
    threshold_ms2: float = 1.5,
    baseline_duration_s: float = 2.0,
    correction_s: float = 1.0,  
) -> float | None:
    """estimate liftoff time from IMU |accel| departing the gravity baseline.
    returns the first time the window mean deviates by more than threshold_ms2, or None if none found."""
    if len(t_s) < 2:
        return None

    # gravity baseline from the first baseline_duration_s of data
    baseline_mask = t_s <= (t_s[0] + baseline_duration_s)
    baseline = float(np.median(accel_mag[baseline_mask])) if baseline_mask.any() else 9.81

    dt_mean = float(np.mean(np.diff(t_s)))
    half_win = max(1, int(round(window_s / dt_mean / 2)))

    for i in range(len(t_s)):
        lo = max(0, i - half_win)
        hi = min(len(accel_mag), i + half_win + 1)
        window_mean = float(np.mean(accel_mag[lo:hi]))
        if abs(window_mean - baseline) > threshold_ms2:
            return float(t_s[i] + correction_s)

    return None


def detect_liftoff_optitrack_z_pos(
    gt_data: dict[str, np.ndarray],
    z_threshold_m: float = 0.5,
) -> float | None:
    """estimate liftoff time from the GT z position first exceeding a threshold."""
    t   = gt_data["timestamps"].astype(np.float64)
    pos = gt_data["positions"]
    if len(t) < 2:
        return None
    for i in range(len(t)):
        if float(pos[i, 2]) >= z_threshold_m:
            return float(t[i])
    return None


def detect_liftoff_optitrack(
    gt_data: dict[str, np.ndarray],
    velocity_threshold_ms: float = 0.05,
    window_s: float = 0.3,
) -> float | None:
    """estimate liftoff time from OptiTrack z-velocity exceeding a threshold."""
    t   = gt_data["timestamps"].astype(np.float64)
    pos = gt_data["positions"]
    if len(t) < 4:
        return None

    z_vel = np.gradient(pos[:, 2].astype(np.float64), t)

    dt_mean = float(np.mean(np.diff(t)))
    half_win = max(1, int(round(window_s / dt_mean / 2)))

    for i in range(len(t)):
        lo = max(0, i - half_win)
        hi = min(len(z_vel), i + half_win + 1)
        if float(np.mean(np.abs(z_vel[lo:hi]))) > velocity_threshold_ms:
            return float(t[i])

    return None


def estimate_time_offset_xcorr(
    imu_t_s: np.ndarray,
    imu_signal: np.ndarray,
    gt_t_s: np.ndarray,
    gt_signal: np.ndarray,
    resample_hz: float = 50.0,
    search_range_s: float | None = None,
) -> tuple[float, float]:
    """estimate the time offset between drone and OptiTrack clocks by cross-correlating two scalar signals.
    returns offset_s (seconds to add to t_ot_raw) and a normalised peak score in [0, 1]."""
    dt = 1.0 / resample_hz

    # build a common-length resampled grid for each signal independently
    imu_t0, imu_t1 = float(imu_t_s[0]), float(imu_t_s[-1])
    gt_t0,  gt_t1  = float(gt_t_s[0]),  float(gt_t_s[-1])
    imu_grid = np.arange(imu_t0, imu_t1, dt)
    gt_grid  = np.arange(gt_t0,  gt_t1,  dt)

    imu_rs = np.interp(imu_grid, imu_t_s.astype(np.float64), imu_signal.astype(np.float64))
    gt_rs  = np.interp(gt_grid,  gt_t_s.astype(np.float64),  gt_signal.astype(np.float64))

    # detrend and normalise
    def _normalise(x: np.ndarray) -> np.ndarray:
        x = x - np.mean(x)
        std = np.std(x)
        return x / std if std > 1e-12 else x

    imu_n = _normalise(imu_rs)
    gt_n  = _normalise(gt_rs)

    # full cross-correlation: positive lag means imu leads gt (gt happened earlier)
    corr = np.correlate(imu_n, gt_n, mode="full")
    lags = np.arange(-(len(gt_n) - 1), len(imu_n)) * dt  # lags in seconds

    # optionally restrict the search window
    if search_range_s is not None:
        mask = np.abs(lags) <= search_range_s
        if mask.any():
            corr = corr[mask]
            lags = lags[mask]

    best_idx = int(np.argmax(corr))
    # lag > 0 means imu_signal is shifted right relative to gt_signal, so gt happened earlier.
    # to align: t_ot_aligned = t_ot_raw + (imu_t0 - gt_t0) - lag
    raw_lag   = float(lags[best_idx])
    offset_s  = (imu_t0 - gt_t0) - raw_lag

    # normalised score (peak over theoretical max)
    max_possible = float(np.sqrt(len(imu_n) * len(gt_n)))
    score = float(corr[best_idx]) / max_possible if max_possible > 0 else 0.0
    score = max(0.0, min(1.0, score))

    return offset_s, score