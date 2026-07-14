"""
ctypes wrapper around kalman_core_a.so, the offline copy of the onboard EKF.
same API as the onboard kalman_core_A module, except predict() takes body-frame
IMU data as arguments so recorded CSV values reproduce the on-device behaviour.
"""

import ctypes
import os

_lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kalman_core_a.so")
_lib = ctypes.CDLL(_lib_path)

# ---- lifecycle -----------------------------------------------------------

_lib.kca_init.restype  = None
_lib.kca_init.argtypes = []

_lib.kca_reset.restype  = None
_lib.kca_reset.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float]

_lib.kca_is_initialized.restype  = ctypes.c_bool
_lib.kca_is_initialized.argtypes = []

# ---- configuration -------------------------------------------------------

_lib.kca_set_imu_to_body_rotation.restype  = None
_lib.kca_set_imu_to_body_rotation.argtypes = [ctypes.c_float] * 9

_lib.kca_set_camera_to_body.restype  = None
_lib.kca_set_camera_to_body.argtypes = [ctypes.c_float] * 9

_lib.kca_set_update_rate.restype  = None
_lib.kca_set_update_rate.argtypes = [ctypes.c_float]

_lib.kca_get_update_period_ms.restype  = ctypes.c_float
_lib.kca_get_update_period_ms.argtypes = []

_lib.kca_set_drag_params.restype  = None
_lib.kca_set_drag_params.argtypes = [ctypes.c_float] * 6

_lib.kca_set_process_noise.restype  = None
_lib.kca_set_process_noise.argtypes = [ctypes.c_float] * 8

_lib.kca_reset_params.restype  = None
_lib.kca_reset_params.argtypes = []

_lib.kca_set_flying.restype  = None
_lib.kca_set_flying.argtypes = [ctypes.c_bool]

_lib.kca_set_camera_intrinsics.restype  = None
_lib.kca_set_camera_intrinsics.argtypes = [ctypes.c_float] * 9

_lib.kca_set_fallback_rho.restype  = None
_lib.kca_set_fallback_rho.argtypes = [ctypes.c_float]

_lib.kca_set_feature_meas_noise.restype  = None
_lib.kca_set_feature_meas_noise.argtypes = [ctypes.c_float]

_lib.kca_set_init_stddev_idepth.restype  = None
_lib.kca_set_init_stddev_idepth.argtypes = [ctypes.c_float]

_lib.kca_set_init_stddev.restype  = None
_lib.kca_set_init_stddev.argtypes = [ctypes.c_float, ctypes.c_float,
                                     ctypes.c_float,
                                     ctypes.c_float, ctypes.c_float]

_lib.kca_set_max_depth_uncertainty_ratio.restype  = None
_lib.kca_set_max_depth_uncertainty_ratio.argtypes = [ctypes.c_float]

_lib.kca_set_innovation_gate.restype  = None
_lib.kca_set_innovation_gate.argtypes = [ctypes.c_float]

_lib.kca_set_vel_max.restype  = None
_lib.kca_set_vel_max.argtypes = [ctypes.c_float]

_lib.kca_set_depth_type.restype  = None
_lib.kca_set_depth_type.argtypes = [ctypes.c_int]

_lib.kca_set_gyro_bias_noise.restype  = None
_lib.kca_set_gyro_bias_noise.argtypes = [ctypes.c_float]
_lib.kca_set_init_stddev_gyro_bias.restype  = None
_lib.kca_set_init_stddev_gyro_bias.argtypes = [ctypes.c_float]
_lib.kca_get_gyro_bias.restype  = None
_lib.kca_get_gyro_bias.argtypes = [ctypes.POINTER(ctypes.c_float)]

_lib.kca_set_robocentric.restype  = None
_lib.kca_set_robocentric.argtypes = [ctypes.c_int]
_lib.kca_set_prf_seed.restype  = None
_lib.kca_set_prf_seed.argtypes = [ctypes.c_int]
_lib.kca_set_fej.restype  = None
_lib.kca_set_fej.argtypes = [ctypes.c_int]
_lib.kca_set_flow_update.restype  = None
_lib.kca_set_flow_update.argtypes = [ctypes.c_int]
_lib.kca_set_no_accel_thrust.restype  = None
_lib.kca_set_no_accel_thrust.argtypes = [ctypes.c_int]
_lib.kca_set_huber_delta.restype  = None
_lib.kca_set_huber_delta.argtypes = [ctypes.c_float]
_lib.kca_flow_update.restype  = ctypes.c_int
_lib.kca_flow_update.argtypes = [ctypes.c_int, ctypes.c_float, ctypes.c_float,
                                 ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float]
_lib.kca_set_feat_process_noise.restype  = None
_lib.kca_set_feat_process_noise.argtypes = [ctypes.c_float, ctypes.c_float]
_lib.kca_gravity_update.restype  = None
_lib.kca_gravity_update.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float]

# ---- EKF steps -----------------------------------------------------------

_lib.kca_predict.restype  = None
_lib.kca_predict.argtypes = [ctypes.c_float] * 6

_lib.kca_predict_dt.restype  = None
_lib.kca_predict_dt.argtypes = [ctypes.c_float] * 7

_lib.kca_finalize.restype  = ctypes.c_bool
_lib.kca_finalize.argtypes = []

_lib.kca_set_attitude_quat.restype  = None
_lib.kca_set_attitude_quat.argtypes = [ctypes.c_float] * 4

_lib.kca_add_process_noise.restype  = None
_lib.kca_add_process_noise.argtypes = [ctypes.c_float]

# ---- state access --------------------------------------------------------

_lib.kca_get_state.restype  = None
_lib.kca_get_state.argtypes = [ctypes.POINTER(ctypes.c_float)]

_lib.kca_get_quaternion.restype  = None
_lib.kca_get_quaternion.argtypes = [ctypes.POINTER(ctypes.c_float)]

_lib.kca_get_rotation_matrix.restype  = None
_lib.kca_get_rotation_matrix.argtypes = [ctypes.POINTER(ctypes.c_float)]

_lib.kca_get_full_covariance.restype  = None
_lib.kca_get_full_covariance.argtypes = [ctypes.POINTER(ctypes.c_float)]

# ---- feature map ---------------------------------------------------------

_lib.kca_add_feature.restype  = ctypes.c_int
_lib.kca_add_feature.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_float]

_lib.kca_feature_update.restype  = ctypes.c_int
_lib.kca_feature_update.argtypes = [
    ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float,
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
]

_lib.kca_remove_feature.restype  = ctypes.c_int
_lib.kca_remove_feature.argtypes = [ctypes.c_int]

_lib.kca_get_active_feature_count.restype  = ctypes.c_int
_lib.kca_get_active_feature_count.argtypes = []

_lib.kca_compute_median_rho.restype  = ctypes.c_float
_lib.kca_compute_median_rho.argtypes = [ctypes.c_float]

_lib.kca_get_feature.restype  = ctypes.c_int
_lib.kca_get_feature.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_float)]

_lib.kca_get_all_features.restype  = ctypes.c_int
_lib.kca_get_all_features.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int]

_lib.kca_get_feature_cov_diag.restype  = ctypes.c_int
_lib.kca_get_feature_cov_diag.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_float)]

# ==========================================================================
# public python API (mirrors the onboard kalman_core_A module)
# ==========================================================================

# ---------- lifecycle -----------------------------------------------------

def init(x: float = 0.0, y: float = 0.0, z: float = 0.0, yaw: float = 0.0) -> None:
    """Init (or re-Init) the EKF; alias for reset()."""
    _lib.kca_init()
    _lib.kca_reset(float(x), float(y), float(z), float(yaw))


def reset(x: float = 0.0, y: float = 0.0, z: float = 0.0, yaw: float = 0.0) -> None:
    """reset EKF state; leaves configuration like update rate or intrinsics unchanged."""
    _lib.kca_init()
    _lib.kca_reset(float(x), float(y), float(z), float(yaw))


def is_initialized() -> bool:
    return bool(_lib.kca_is_initialized())


# ---------- configuration -------------------------------------------------

def set_imu_to_body_rotation(r00, r01, r02, r10, r11, r12, r20, r21, r22) -> None:
    """set the 3x3 IMU-to-body rotation matrix (row-major, 9 scalar args).
    pass identity when the recorded IMU data is already in body frame."""
    _lib.kca_set_imu_to_body_rotation(
        float(r00), float(r01), float(r02),
        float(r10), float(r11), float(r12),
        float(r20), float(r21), float(r22),
    )


def set_camera_to_body(r00, r01, r02, r10, r11, r12, r20, r21, r22) -> None:
    """set the 3x3 R_cb camera-to-body rotation (row-major, 9 scalar args).
    the horizontally-mounted camera maps the optical axis onto the forward body axis."""
    _lib.kca_set_camera_to_body(
        float(r00), float(r01), float(r02),
        float(r10), float(r11), float(r12),
        float(r20), float(r21), float(r22),
    )


def set_update_rate(hz: float) -> None:
    _lib.kca_set_update_rate(float(hz))


def get_update_period_ms() -> float:
    return float(_lib.kca_get_update_period_ms())


def set_drag_params(bx: float = 4.2, by: float = 1.8, bz: float = 0.3,
                    rx: float = 0.0, ry: float = 0.0, rz: float = 0.03) -> None:
    _lib.kca_set_drag_params(float(bx), float(by), float(bz),
                              float(rx), float(ry), float(rz))


def set_process_noise(acc_xy: float = 0.5, acc_z: float = 1.0,
                      vel: float = 0.0,    pos: float = 0.0,
                      att: float = 0.0,    gyro_rp: float = 0.1,
                      gyro_yaw: float = 0.1, att_reversion: float = 0.001) -> None:
    _lib.kca_set_process_noise(float(acc_xy), float(acc_z), float(vel),  float(pos),
                                float(att),    float(gyro_rp), float(gyro_yaw),
                                float(att_reversion))


def reset_params() -> None:
    """restore all noise and drag params to firmware defaults."""
    _lib.kca_reset_params()


def set_flying(flying: bool) -> None:
    _lib.kca_set_flying(bool(flying))


def set_camera_intrinsics(fx: float, fy: float, cx: float, cy: float,
                           k1: float = 0.0, k2: float = 0.0,
                           p1: float = 0.0, p2: float = 0.0,
                           k3: float = 0.0) -> None:
    _lib.kca_set_camera_intrinsics(float(fx), float(fy), float(cx), float(cy),
                                    float(k1), float(k2), float(p1), float(p2), float(k3))


def set_fallback_rho(fallback: float) -> None:
    _lib.kca_set_fallback_rho(float(fallback))


def set_feature_meas_noise(variance: float) -> None:
    _lib.kca_set_feature_meas_noise(float(variance))


def set_init_stddev_idepth(stddev: float) -> None:
    """set the initial relative std-dev for inverse-depth covariance.
    sigma_rho = stddev * rho_init; larger widens initial depth uncertainty."""
    _lib.kca_set_init_stddev_idepth(float(stddev))


def set_init_stddev(pos_xy: float, pos_z: float,
                    vel: float,
                    att_rp: float, att_yaw: float) -> None:
    """Init covariance std-devs applied at each reset; pass 0 to keep current.
    pos_xy/pos_z [m], vel [m/s], att_rp/att_yaw [rad]."""
    _lib.kca_set_init_stddev(float(pos_xy), float(pos_z),
                             float(vel),
                             float(att_rp), float(att_yaw))


def set_max_depth_uncertainty_ratio(ratio: float) -> None:
    """Sigma_rho/rho threshold for the adaptive median-rho computation.
    only features below this ratio contribute to rho_init for new features."""
    _lib.kca_set_max_depth_uncertainty_ratio(float(ratio))


def set_innovation_gate(threshold: float) -> None:
    """Set the Mahalanobis d² gate for feature updates; d² above threshold is skipped.
    pass 0.0 to disable (default)."""
    _lib.kca_set_innovation_gate(float(threshold))


def set_vel_max(vel_max: float) -> None:
    """Test wip: velocity clamp [m/s] applied at the end of each predict so |v| cannot run away.
    pass 0.0 to disable (default 3.0)."""
    _lib.kca_set_vel_max(float(vel_max))


_DEPTH_TYPES = {"regular": 0, "inverse": 1, "log": 2, "hyperbolic": 3}

def set_depth_type(depth_type) -> None:
    """Feat depth parametrization; INVERSE (default)m others expririmental
    log and hyperbolic are not in paper."""
    if isinstance(depth_type, str):
        depth_type = _DEPTH_TYPES.get(depth_type.strip().lower(), 1)
    _lib.kca_set_depth_type(int(depth_type))


def set_gyro_bias_noise(sigma: float) -> None:
    """Random-walk spectral density [rad/s/√Hz], 0 = bias frozen (default). not in paper."""
    _lib.kca_set_gyro_bias_noise(float(sigma))


def set_init_stddev_gyro_bias(stddev: float) -> None:
    """Init gyro-bias std [rad/s], 0 = bias estimation disabled (default). not in paper."""
    _lib.kca_set_init_stddev_gyro_bias(float(stddev))


def get_gyro_bias() -> tuple:
    """return the estimated gyro bias (bgx, bgy, bgz) [rad/s]. not in paper."""
    buf = (ctypes.c_float * 3)()
    _lib.kca_get_gyro_bias(buf)
    return (buf[0], buf[1], buf[2])


def set_robocentric(enable) -> None:
    """enable robocentric feature anchoring; 0/False = world-anchored (default). not in paper."""
    _lib.kca_set_robocentric(1 if enable else 0)


def set_prf_seed(enable) -> None:
    """seed P_rf=P_rr*Gx^T at feature init (default off). WIP, not in paper."""
    _lib.kca_set_prf_seed(1 if enable else 0)


def set_fej(enable) -> None:
    """FEJ first-estimates jacobians (default off): freeze R in the predict blocks to
    keep the yaw+position nullspace honest. not in paper."""
    _lib.kca_set_fej(1 if enable else 0)


def set_flow_update(enable) -> None:
    """event-flow velocity update (default off): use per-track pixel flow (vx,vy) as a
    2D measurement of feature image motion. not in paper."""
    _lib.kca_set_flow_update(1 if enable else 0)


def set_no_accel_thrust(enable) -> None:
    """no-thrust vertical model (default off): assume thrust = gravity so measured a_z does
    not drive v_z, which becomes a process-noise random walk corrected by vision."""
    _lib.kca_set_no_accel_thrust(1 if enable else 0)


def set_huber_delta(delta: float) -> None:
    """huber robust threshold [sigma] for feature updates (default 0=off): observations
    with normalized innovation d>delta get their R inflated instead of hard gating. not in paper."""
    _lib.kca_set_huber_delta(float(delta))


def flow_update(fid: int, vx_px: float, vy_px: float,
                gx_dps: float, gy_dps: float, gz_dps: float, meas_var: float) -> int:
    """DOES not work/wip: front-end flow velocity update for one feature; gx/gy/gz are the body-frame gyro [dps]
    fed to predict. returns 0 ok, 2 gated, -1 skipped. not in paper."""
    return int(_lib.kca_flow_update(int(fid), float(vx_px), float(vy_px),
                                    float(gx_dps), float(gy_dps), float(gz_dps), float(meas_var)))


def set_feat_process_noise(bearing: float, depth: float) -> None:
    """Ror the Robocentric per-step feature process noise (bearing [rad/√Hz], depth-param). not in paper."""
    _lib.kca_set_feat_process_noise(float(bearing), float(depth))


def gravity_update(ax_mg: float, ay_mg: float, az_mg: float,
                   meas_var: float, mag_tol: float = 0.0) -> None:
    """Roll/pitch attitude correction, called once per predict in flight.
    meas_var large = weak/noisy reference; mag_tol rejects |accel| far from g (0=off)."""
    _lib.kca_gravity_update(float(ax_mg), float(ay_mg), float(az_mg),
                            float(meas_var), float(mag_tol))


# ---------- EKF steps -----------------------------------------------------

def predict(ax_mg: float, ay_mg: float, az_mg: float,
            gx_dps: float, gy_dps: float, gz_dps: float) -> None:
    """IMU propagation step using the configured fixed dt (matches on-board)."""
    _lib.kca_predict(float(ax_mg), float(ay_mg), float(az_mg),
                      float(gx_dps), float(gy_dps), float(gz_dps))


def predict_dt(ax_mg: float, ay_mg: float, az_mg: float,
               gx_dps: float, gy_dps: float, gz_dps: float,
               dt_s: float) -> None:
    """IMU propagation step with caller-supplied dt in seconds.
    used for offline replay where CSV row spacing differs from the update rate."""
    _lib.kca_predict_dt(float(ax_mg), float(ay_mg), float(az_mg),
                         float(gx_dps), float(gy_dps), float(gz_dps),
                         float(dt_s))


def finalize() -> bool:
    """Fold the attitude error back into the quaternion and rebuild R.
    returns True if a predict was pending."""
    return bool(_lib.kca_finalize())


def set_attitude_quat(qw: float, qx: float, qy: float, qz: float) -> None:
    """Override the attitude quaternion and rebuild R.
    call after reset() to apply gravity-aligned roll/pitch while keeping the reset state/cov."""
    _lib.kca_set_attitude_quat(float(qw), float(qx), float(qy), float(qz))


def add_process_noise(dt_ms: float) -> None:
    _lib.kca_add_process_noise(float(dt_ms))


# ---------- state access --------------------------------------------------

def get_state() -> dict:
    """returns {'x','y','z','vx','vy','vz','d0','d1','d2'}."""
    buf = (ctypes.c_float * 9)()
    _lib.kca_get_state(buf)
    keys = ('x', 'y', 'z', 'vx', 'vy', 'vz', 'd0', 'd1', 'd2')
    return {k: float(v) for k, v in zip(keys, buf)}


def get_quaternion() -> tuple:
    """returns (w, x, y, z)."""
    buf = (ctypes.c_float * 4)()
    _lib.kca_get_quaternion(buf)
    return tuple(float(v) for v in buf)


def get_rotation_matrix() -> tuple:
    """returns the 3x3 R rotation matrix as a tuple of 3 row-tuples."""
    buf = (ctypes.c_float * 9)()
    _lib.kca_get_rotation_matrix(buf)
    return tuple(tuple(float(buf[i*3+j]) for j in range(3)) for i in range(3))


def get_full_covariance() -> list:
    """returns the 9x9 P covariance as a list of 9 lists."""
    buf = (ctypes.c_float * 81)()
    _lib.kca_get_full_covariance(buf)
    return [[float(buf[i*9+j]) for j in range(9)] for i in range(9)]


# ---------- feature map ---------------------------------------------------

def add_feature(u: float, v: float, rho_init: float = 0.0) -> int:
    """Init a feature from pixel coords; returns the slot id (0-31).
    rho_init=0.0 (default) uses the adaptive median depth of active features."""
    slot = _lib.kca_add_feature(float(u), float(v), float(rho_init))
    if slot < 0:
        if slot == -2:
            raise ValueError("feature pool full (max 32)")
        raise ValueError("add_feature failed (ret=%d)" % slot)
    return slot


def feature_update(fid: int, u: float, v: float, meas_noise: float = 0.0) -> tuple:
    """EKF measurement update for one feature, returning (residual_x, residual_y, gated, nis).
    gated=True means the Mahalanobis gate rejected it, and nis is d² = rᵀS⁻¹r (-1.0 if invalid)."""
    rx = ctypes.c_float(0.0)
    ry = ctypes.c_float(0.0)
    d2 = ctypes.c_float(-1.0)
    ret = _lib.kca_feature_update(int(fid), float(u), float(v), float(meas_noise),
                                   ctypes.byref(rx), ctypes.byref(ry), ctypes.byref(d2))
    if ret == -1:
        raise ValueError("feature_update failed for id %d" % fid)
    gated = (ret == 2)
    return (rx.value, ry.value, gated, d2.value)


def remove_feature(fid: int) -> None:
    ret = _lib.kca_remove_feature(int(fid))
    if ret != 0:
        raise ValueError("remove_feature: invalid or inactive id %d" % fid)


def get_active_feature_count() -> int:
    return int(_lib.kca_get_active_feature_count())


def compute_median_rho(fallback: float = 0.5) -> float:
    """Median rho of active low-uncertainty features; returns fallback if fewer than 3 qualify."""
    return float(_lib.kca_compute_median_rho(float(fallback)))


def get_feature(fid: int) -> tuple:
    """returns (bx, by, rho) for the given feature slot."""
    buf = (ctypes.c_float * 6)()
    ret = _lib.kca_get_feature(int(fid), buf)
    if ret != 0:
        raise ValueError("get_feature: invalid or inactive id %d" % fid)
    return (float(buf[0]), float(buf[1]), float(buf[2]))


def get_all_features() -> list:
    """Get a list of (id, bx, by, rho, Xw, Yw, Zw) for all features."""
    _MAX = 32
    buf = (ctypes.c_float * (_MAX * 7))()
    count = _lib.kca_get_all_features(buf, _MAX)
    result = []
    for i in range(count):
        base = i * 7
        entry = (int(buf[base]),) + tuple(float(buf[base + j]) for j in range(1, 7))
        result.append(entry)
    return result


def get_feature_cov_diag(fid: int) -> tuple:
    """Get (var_bx, var_by, var_rho), the diagonal of P_ff for the feature id."""
    buf = (ctypes.c_float * 3)()
    ret = _lib.kca_get_feature_cov_diag(int(fid), buf)
    if ret != 0:
        raise ValueError("get_feature_cov_diag: invalid or inactive id %d" % fid)
    return (float(buf[0]), float(buf[1]), float(buf[2]))
