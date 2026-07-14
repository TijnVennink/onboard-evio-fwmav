
#include "kalman_core_a_native.h"
#include "physicalConstants.h"
#include <math.h>
#define KC_STATE_DIM 9
#include "mat9.h"
#include "eventCameraConstants.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#define DEG2RAD   ((float)(M_PI / 180.0))
#define MG_TO_MS2 (GRAVITY_MAGNITUDE / 1000.0f)

/* init covariance std-devs, runtime-configurable as well */
#define DEFAULT_INIT_STDDEV_POS_XY  0.1f
#define DEFAULT_INIT_STDDEV_POS_Z   0.1f
#define DEFAULT_INIT_STDDEV_VEL     0.2f
#define DEFAULT_INIT_STDDEV_ATT_RP  0.05f
#define DEFAULT_INIT_STDDEV_ATT_YAW 0.05f

/* feature map standard values, runtime config as well */
#define MAX_FEATURES        70
#define INIT_STDDEV_BEARING 0.003f
#define INIT_STDDEV_IDEPTH  0.001f
#define DEFAULT_MEAS_NOISE  0.01f
#define FALLBACK_RHO        0.33f
#define MAX_DEPTH_UNCERTAINTY_RATIO  0.5f

typedef enum {
    KC_STATE_X, KC_STATE_Y, KC_STATE_Z,
    KC_STATE_PX, KC_STATE_PY, KC_STATE_PZ,
    KC_STATE_D0, KC_STATE_D1, KC_STATE_D2,
    KC_STATE_BGX, KC_STATE_BGY, KC_STATE_BGZ
} kalmanCoreStateIdx_t;

/* depth parametrization, only inverse used in work, others WIP */
typedef enum {
    DEPTH_REGULAR = 0,
    DEPTH_INVERSE = 1,
    DEPTH_LOG = 2,
    DEPTH_HYPERBOLIC = 3
} depth_type_t;

/* per-feature state, bearing stored as unit vector on S² */
typedef struct {
    bool  active;
    float h[3];   /* unit bearing vector */
    float p;      /* depth parameter, INVERSE => 1/||Xw|| */
    float P_ff[3][3];  /* feature covariance */
    float P_rf[KC_STATE_DIM][3];  /* robot-feature cross cov */
} feature_state_t;

/* full filter state, 9-state EKF (pos, vel, attitude error) plus the feature map */
typedef struct {
    bool  initialized;
    bool  is_updated;
    bool  is_flying;

    float update_frequency_hz;
    float update_period_ms;

    float imu_to_body_R[3][3];
    float R_cb[3][3];            /* camera→body rotation */
    float R_fej[3][3];           /* first-estimate rotation for FEJ jacobian */

    float state[KC_STATE_DIM];
    float P[KC_STATE_DIM][KC_STATE_DIM];
    float q[4];
    float R[3][3];               /* body→world rotation */
    float initial_quat[4];

    /* body-frame drag model (similar to crazyfie firmware) */
    float dragBx;
    float dragBy;
    float dragBz;
    float drag_rx;
    float drag_ry;
    float drag_rz;

    /* camera intrinsics + brown distortion */
    float cam_fx, cam_fy;
    float cam_cx, cam_cy;
    float cam_k1, cam_k2, cam_k3;
    float cam_p1, cam_p2;

    /* Process noise, spectral densities so tuned values transfer across rates */
    float procNoiseAcc_xy;
    float procNoiseAcc_z;
    float procNoiseVel;
    float procNoisePos;
    float procNoiseAtt;
    float measNoiseGyro_rollpitch;
    float measNoiseGyro_yaw;
    float procNoiseGyroBias;   /* gyro-bias random walk [rad/s/√Hz], 0=frozen. not in paper, WIP */
    float procNoiseFeatBearing;/* robocentric bearing noise [rad/√Hz]. not in paper WIP */
    float procNoiseFeatDepth;  /* robocentric depth noise [param/√Hz]. not in paper WIP */

    float attitude_reversion;  /* pulls attitude back to init while not flying */

    feature_state_t features[MAX_FEATURES];
    int   active_feature_count;
    float meas_noise_feature;
    float fallback_rho;        /* inverse depth used when the median is not available */

    /* init covariance std-devs, applied on every reset */
    float init_stddev_pos_xy;
    float init_stddev_pos_z;
    float init_stddev_vel;
    float init_stddev_att_rp;
    float init_stddev_att_yaw;
    float init_stddev_gyro_bias;   /* not in paper */
    float init_stddev_idepth;      /* relative depth uncertainty at feature init */
    float max_depth_uncertainty_ratio;

    float innovation_gate;     /* mahalanobis gate on feature update, 0=off */
    float huber_delta;         /* huber down-weighting, 0=off. not in paper WIP */
    float vel_max;             /* physical velocity clamp [m/s], 0=off */

    int   depth_type;          /* only INVERSE used, others WIP */
    int   robocentric;         /* camera-frame features. not in paper WIP */
    int   prf_seed;            /* seed robot<->feature cross-cov at init. shelved */
    int   fej;                 /* first-estimate jacobian on/off */
    int   fej_anchored;        /* R_fej captured at first predict */
    int   flow_update;         /* event-flow velocity update. not in paper WIP */
    int   no_accel_thrust;     /* assume thrust = gravity so a_z does not drive v_z */
} py_kalman_core_t;

static py_kalman_core_t kalman_core;

/* -------------------------------------------------------------------------
 * depth parametrization conversions 
 * d = distance, p = stored parameter. only inverse used, rest is WIP(not in poaper)
 * ------------------------------------------------------------------------- */
static inline float depth_nonzero(float v) {
    if (v >= 0.0f) return (v < 1e-6f) ? 1e-6f : v;
    return (v > -1e-6f) ? -1e-6f : v;
}
/* distance from parameter */
static inline float depth_d_from_p(float p) {
    switch (kalman_core.depth_type) {
        case DEPTH_REGULAR:    return p;
        case DEPTH_LOG:        return expf(p);
        case DEPTH_HYPERBOLIC: return sinhf(p);
        default:               return 1.0f / depth_nonzero(p);  /* INVERSE */
    }
}
/* parameter from distance */
static inline float depth_p_from_d(float d) {
    switch (kalman_core.depth_type) {
        case DEPTH_REGULAR:    return d;
        case DEPTH_LOG:        return logf(depth_nonzero(d));
        case DEPTH_HYPERBOLIC: return asinhf(d);
        default:               return 1.0f / depth_nonzero(d);  /* INVERSE */
    }
}
/* d(distance)/d(parameter), evaluated at parameter p */
static inline float depth_ddist_dp(float p) {
    switch (kalman_core.depth_type) {
        case DEPTH_REGULAR:    return 1.0f;
        case DEPTH_LOG:        return expf(p);
        case DEPTH_HYPERBOLIC: return coshf(p);
        default: { float pp = depth_nonzero(p); return -1.0f / (pp * pp); }  /* INVERSE */
    }
}
/* d(parameter)/d(distance), evaluated at distance d */
static inline float depth_dp_ddist(float d) {
    switch (kalman_core.depth_type) {
        case DEPTH_REGULAR:    return 1.0f;
        case DEPTH_LOG:        return 1.0f / depth_nonzero(d);
        case DEPTH_HYPERBOLIC: return 1.0f / sqrtf(d * d + 1.0f);
        default: { float dd = depth_nonzero(d); return -1.0f / (dd * dd); }  /* INVERSE */
    }
}

/* -------------------------------------------------------------------------
 * robocentric basis (WIP!)
 * ------------------------------------------------------------------------- */

static void tangent_basis(const float h[3], float T1[3], float T2[3]) {
    if (fabsf(h[0]) < 0.9f) {
        float t = sqrtf(h[1]*h[1] + h[2]*h[2]);
        T1[0] = 0.0f;  T1[1] = -h[2]/t;  T1[2] =  h[1]/t;
    } else {
        float t = sqrtf(h[0]*h[0] + h[2]*h[2]);
        T1[0] =  h[2]/t;  T1[1] = 0.0f;  T1[2] = -h[0]/t;
    }
    T2[0] = h[1]*T1[2] - h[2]*T1[1];
    T2[1] = h[2]*T1[0] - h[0]*T1[2];
    T2[2] = h[0]*T1[1] - h[1]*T1[0];
}

/* robocentric features, not used in paper WIP!!! */
static void robo_propagate_feature(const float nor[3], float p,
                                   const float v_b[3], const float omega[3],
                                   float dt, float nor_out[3], float *p_out) {
    float d = depth_d_from_p(p);
    if (!isfinite(d) || d < EPS) d = EPS;
    float nv = nor[0]*v_b[0] + nor[1]*v_b[1] + nor[2]*v_b[2];    
    float invd = 1.0f / d;
    float wxn0 = omega[1]*nor[2] - omega[2]*nor[1];
    float wxn1 = omega[2]*nor[0] - omega[0]*nor[2];
    float wxn2 = omega[0]*nor[1] - omega[1]*nor[0];
    float nd0 = -wxn0 - invd*(v_b[0] - nv*nor[0]);
    float nd1 = -wxn1 - invd*(v_b[1] - nv*nor[1]);
    float nd2 = -wxn2 - invd*(v_b[2] - nv*nor[2]);
    float nn0 = nor[0] + dt*nd0, nn1 = nor[1] + dt*nd1, nn2 = nor[2] + dt*nd2;
    float nm = sqrtf(nn0*nn0 + nn1*nn1 + nn2*nn2) + EPS;
    nor_out[0] = nn0/nm;  nor_out[1] = nn1/nm;  nor_out[2] = nn2/nm;
    *p_out = p + dt * depth_dp_ddist(d) * (-nv);
}

/* -------------------------------------------------------------------------
 * local helpers
 * ------------------------------------------------------------------------- */

static void apply_body_transform(float *x, float *y, float *z) {
    float bx = kalman_core.imu_to_body_R[0][0]*(*x)
             + kalman_core.imu_to_body_R[0][1]*(*y)
             + kalman_core.imu_to_body_R[0][2]*(*z);
    float by = kalman_core.imu_to_body_R[1][0]*(*x)
             + kalman_core.imu_to_body_R[1][1]*(*y)
             + kalman_core.imu_to_body_R[1][2]*(*z);
    float bz = kalman_core.imu_to_body_R[2][0]*(*x)
             + kalman_core.imu_to_body_R[2][1]*(*y)
             + kalman_core.imu_to_body_R[2][2]*(*z);
    *x = bx;  *y = by;  *z = bz;
}

static void pixel_to_normalised(float u, float v, float *xn, float *yn) {
    float x = (u - kalman_core.cam_cx) / kalman_core.cam_fx;
    float y = (v - kalman_core.cam_cy) / kalman_core.cam_fy;
    float k1 = kalman_core.cam_k1, k2 = kalman_core.cam_k2, k3 = kalman_core.cam_k3;
    float p1 = kalman_core.cam_p1, p2 = kalman_core.cam_p2;
    if (fabsf(k1) < EPS && fabsf(k2) < EPS && fabsf(k3) < EPS &&
        fabsf(p1) < EPS && fabsf(p2) < EPS) {
        *xn = x;  *yn = y;  return;
    }
    float xd_obs = x;
    float yd_obs = y;
    for (int i = 0; i < 5; i++) {
        float r2 = x*x + y*y;
        float r4 = r2*r2, r6 = r4*r2;
        float radial = 1.0f + k1*r2 + k2*r4 + k3*r6;
        float tang_x = 2.0f*p1*x*y + p2*(r2 + 2.0f*x*x);
        float tang_y = p1*(r2 + 2.0f*y*y) + 2.0f*p2*x*y;
        float xd = x*radial + tang_x;
        float yd = y*radial + tang_y;
        x -= (xd - xd_obs) / (radial + EPS);
        y -= (yd - yd_obs) / (radial + EPS);
    }
    *xn = x;  *yn = y;
}

static void set_default_params(void) {
    kalman_core.dragBx  = EKF_DRAG_BX;
    kalman_core.dragBy  = EKF_DRAG_BY;
    kalman_core.dragBz  = EKF_DRAG_BZ;
    kalman_core.drag_rx = EKF_DRAG_RX;
    kalman_core.drag_ry = EKF_DRAG_RY;
    kalman_core.drag_rz = EKF_DRAG_RZ;

    kalman_core.procNoiseAcc_xy         = 0.5f;
    kalman_core.procNoiseAcc_z          = 1.0f;
    kalman_core.procNoiseVel            = 0.0f;
    kalman_core.procNoisePos            = 0.0f;
    kalman_core.procNoiseAtt            = 0.0f;
    kalman_core.measNoiseGyro_rollpitch = 0.1f;
    kalman_core.measNoiseGyro_yaw       = 0.1f;
    kalman_core.procNoiseGyroBias       = 0.0f;  /* bias off by default. not in paper */
    kalman_core.procNoiseFeatBearing    = 0.0f;  /* robocentric only, not in paper */
    kalman_core.procNoiseFeatDepth      = 0.0f;  /* robocentric only, not in paper */
    kalman_core.attitude_reversion      = 0.001f;
    kalman_core.fallback_rho                 = FALLBACK_RHO;
    kalman_core.init_stddev_idepth           = INIT_STDDEV_IDEPTH;
    kalman_core.max_depth_uncertainty_ratio  = MAX_DEPTH_UNCERTAINTY_RATIO;

    /* only default these if unset (<=0), else a user set_init_stddev gets wiped.
     * reset() runs reset_state twice so the guard keeps the swept value alive. */
    if (kalman_core.init_stddev_pos_xy  <= 0.0f) kalman_core.init_stddev_pos_xy  = DEFAULT_INIT_STDDEV_POS_XY;
    if (kalman_core.init_stddev_pos_z   <= 0.0f) kalman_core.init_stddev_pos_z   = DEFAULT_INIT_STDDEV_POS_Z;
    if (kalman_core.init_stddev_vel     <= 0.0f) kalman_core.init_stddev_vel     = DEFAULT_INIT_STDDEV_VEL;
    if (kalman_core.init_stddev_att_rp  <= 0.0f) kalman_core.init_stddev_att_rp  = DEFAULT_INIT_STDDEV_ATT_RP;
    if (kalman_core.init_stddev_att_yaw <= 0.0f) kalman_core.init_stddev_att_yaw = DEFAULT_INIT_STDDEV_ATT_YAW;
    kalman_core.init_stddev_gyro_bias = 0.0f;  /* bias off by default. not in paper */
}

static void reset_state(float x, float y, float z, float yaw_rad) {
    kalman_core.state[KC_STATE_X ] = x;
    kalman_core.state[KC_STATE_Y ] = y;
    kalman_core.state[KC_STATE_Z ] = z;
    kalman_core.state[KC_STATE_PX] = 0.0f;
    kalman_core.state[KC_STATE_PY] = 0.0f;
    kalman_core.state[KC_STATE_PZ] = 0.0f;
    kalman_core.state[KC_STATE_D0] = 0.0f;
    kalman_core.state[KC_STATE_D1] = 0.0f;
    kalman_core.state[KC_STATE_D2] = 0.0f;
#if KC_STATE_DIM >= 12
    kalman_core.state[KC_STATE_BGX] = 0.0f;
    kalman_core.state[KC_STATE_BGY] = 0.0f;
    kalman_core.state[KC_STATE_BGZ] = 0.0f;
#endif

    kalman_core.q[0] = cosf(yaw_rad * 0.5f);
    kalman_core.q[1] = 0.0f;
    kalman_core.q[2] = 0.0f;
    kalman_core.q[3] = sinf(yaw_rad * 0.5f);

    kalman_core.initial_quat[0] = kalman_core.q[0];
    kalman_core.initial_quat[1] = kalman_core.q[1];
    kalman_core.initial_quat[2] = kalman_core.q[2];
    kalman_core.initial_quat[3] = kalman_core.q[3];

    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            kalman_core.R[i][j] = (i == j) ? 1.0f : 0.0f;

    float sxy = kalman_core.init_stddev_pos_xy;
    float sz  = kalman_core.init_stddev_pos_z;
    float sv  = kalman_core.init_stddev_vel;
    float srp = kalman_core.init_stddev_att_rp;
    float syaw= kalman_core.init_stddev_att_yaw;
#if KC_STATE_DIM >= 12
    float sbg = kalman_core.init_stddev_gyro_bias;
#endif
    memset(kalman_core.P, 0, sizeof(kalman_core.P));
    kalman_core.P[KC_STATE_X ][KC_STATE_X ] = sxy * sxy;
    kalman_core.P[KC_STATE_Y ][KC_STATE_Y ] = sxy * sxy;
    kalman_core.P[KC_STATE_Z ][KC_STATE_Z ] = sz  * sz;
    kalman_core.P[KC_STATE_PX][KC_STATE_PX] = sv  * sv;
    kalman_core.P[KC_STATE_PY][KC_STATE_PY] = sv  * sv;
    kalman_core.P[KC_STATE_PZ][KC_STATE_PZ] = sv  * sv;
    kalman_core.P[KC_STATE_D0][KC_STATE_D0] = srp * srp;
    kalman_core.P[KC_STATE_D1][KC_STATE_D1] = srp * srp;
    kalman_core.P[KC_STATE_D2][KC_STATE_D2] = syaw* syaw;
#if KC_STATE_DIM >= 12
    kalman_core.P[KC_STATE_BGX][KC_STATE_BGX] = sbg * sbg;
    kalman_core.P[KC_STATE_BGY][KC_STATE_BGY] = sbg * sbg;
    kalman_core.P[KC_STATE_BGZ][KC_STATE_BGZ] = sbg * sbg;
#endif

    set_default_params();

    kalman_core.is_flying  = false;
    kalman_core.is_updated = false;

    memset(kalman_core.features, 0, sizeof(kalman_core.features));
    kalman_core.active_feature_count = 0;
    kalman_core.meas_noise_feature   = DEFAULT_MEAS_NOISE;
    kalman_core.innovation_gate      = 0.0f;  /* off by default */
    kalman_core.huber_delta          = 0.0f;  /* off by default. not in paper */
    kalman_core.vel_max              = 3.0f;  /* m/s clamp, 0=off */
    kalman_core.depth_type           = DEPTH_INVERSE;  /* inverse depth */
    kalman_core.robocentric          = 0;              /* world-anchored. not in paper */
    kalman_core.prf_seed             = 0;              /* cross-cov seed off, shelved */
    kalman_core.fej_anchored         = 0;              /* re-anchor FEJ next predict */
    kalman_core.flow_update          = 0;              /* off by default. not in paper */
    kalman_core.no_accel_thrust      = 0;              /* measured a_z drives v_z */
}

/* =========================================================================
 * Public API
 * ========================================================================= */

void kca_init(void) {
    /* imu-to-body identity by default, defined in config */
    kalman_core.imu_to_body_R[0][0] = 1.0f; kalman_core.imu_to_body_R[0][1] = 0.0f; kalman_core.imu_to_body_R[0][2] = 0.0f;
    kalman_core.imu_to_body_R[1][0] = 0.0f; kalman_core.imu_to_body_R[1][1] = 1.0f; kalman_core.imu_to_body_R[1][2] = 0.0f;
    kalman_core.imu_to_body_R[2][0] = 0.0f; kalman_core.imu_to_body_R[2][1] = 0.0f; kalman_core.imu_to_body_R[2][2] = 1.0f;
    /* camera-to-body identity by default, defined in config */
    kalman_core.R_cb[0][0] = 1.0f; kalman_core.R_cb[0][1] = 0.0f; kalman_core.R_cb[0][2] = 0.0f;
    kalman_core.R_cb[1][0] = 0.0f; kalman_core.R_cb[1][1] = 1.0f; kalman_core.R_cb[1][2] = 0.0f;
    kalman_core.R_cb[2][0] = 0.0f; kalman_core.R_cb[2][1] = 0.0f; kalman_core.R_cb[2][2] = 1.0f;
    kalman_core.update_frequency_hz = 500.0f;
    kalman_core.update_period_ms    = 2.0f;
    kalman_core.cam_fx = CAM_FX;  kalman_core.cam_fy = CAM_FY;
    kalman_core.cam_cx = CAM_CX;  kalman_core.cam_cy = CAM_CY;
    kalman_core.cam_k1 = CAM_K1;  kalman_core.cam_k2 = CAM_K2;
    kalman_core.cam_k3 = CAM_K3;  kalman_core.cam_p1 = CAM_P1;
    kalman_core.cam_p2 = CAM_P2;
    reset_state(0.0f, 0.0f, 0.0f, 0.0f);
    kalman_core.initialized = true;
}

void kca_reset(float x, float y, float z, float yaw_rad) {
    reset_state(x, y, z, yaw_rad);
    kalman_core.initialized = true;
}

bool kca_is_initialized(void) {
    return kalman_core.initialized;
}

void kca_set_imu_to_body_rotation(float r00, float r01, float r02,
                                   float r10, float r11, float r12,
                                   float r20, float r21, float r22) {
    kalman_core.imu_to_body_R[0][0] = r00; kalman_core.imu_to_body_R[0][1] = r01; kalman_core.imu_to_body_R[0][2] = r02;
    kalman_core.imu_to_body_R[1][0] = r10; kalman_core.imu_to_body_R[1][1] = r11; kalman_core.imu_to_body_R[1][2] = r12;
    kalman_core.imu_to_body_R[2][0] = r20; kalman_core.imu_to_body_R[2][1] = r21; kalman_core.imu_to_body_R[2][2] = r22;
}

void kca_set_camera_to_body(float r00, float r01, float r02,
                            float r10, float r11, float r12,
                            float r20, float r21, float r22) {
    kalman_core.R_cb[0][0] = r00; kalman_core.R_cb[0][1] = r01; kalman_core.R_cb[0][2] = r02;
    kalman_core.R_cb[1][0] = r10; kalman_core.R_cb[1][1] = r11; kalman_core.R_cb[1][2] = r12;
    kalman_core.R_cb[2][0] = r20; kalman_core.R_cb[2][1] = r21; kalman_core.R_cb[2][2] = r22;
}

void kca_set_update_rate(float hz) {
    if (hz > 0.0f) {
        kalman_core.update_frequency_hz = hz;
        kalman_core.update_period_ms    = 1000.0f / hz;
    }
}

float kca_get_update_period_ms(void) {
    return kalman_core.update_period_ms;
}

void kca_set_drag_params(float bx, float by, float bz,
                          float rx, float ry, float rz) {
    kalman_core.dragBx   = bx;  kalman_core.dragBy   = by;  kalman_core.dragBz   = bz;
    kalman_core.drag_rx  = rx;  kalman_core.drag_ry  = ry;  kalman_core.drag_rz  = rz;
}

void kca_set_process_noise(float acc_xy, float acc_z, float vel, float pos,
                             float att, float gyro_rp, float gyro_yaw,
                             float att_reversion) {
    kalman_core.procNoiseAcc_xy         = acc_xy;
    kalman_core.procNoiseAcc_z          = acc_z;
    kalman_core.procNoiseVel            = vel;
    kalman_core.procNoisePos            = pos;
    kalman_core.procNoiseAtt            = att;
    kalman_core.measNoiseGyro_rollpitch = gyro_rp;
    kalman_core.measNoiseGyro_yaw       = gyro_yaw;
    kalman_core.attitude_reversion      = att_reversion;
}

void kca_reset_params(void) {
    set_default_params();
}

void kca_set_flying(bool flying) {
    kalman_core.is_flying = flying;
}

void kca_set_camera_intrinsics(float fx, float fy, float cx, float cy,
                                 float k1, float k2, float p1, float p2,
                                 float k3) {
    kalman_core.cam_fx = fx;  kalman_core.cam_fy = fy;
    kalman_core.cam_cx = cx;  kalman_core.cam_cy = cy;
    kalman_core.cam_k1 = k1;  kalman_core.cam_k2 = k2;
    kalman_core.cam_p1 = p1;  kalman_core.cam_p2 = p2;
    kalman_core.cam_k3 = k3;
}

void kca_set_fallback_rho(float fallback) {
    if (fallback > 0.0f) kalman_core.fallback_rho = fallback;
}

void kca_set_init_stddev_idepth(float stddev) {
    if (stddev > 0.0f) kalman_core.init_stddev_idepth = stddev;
}

void kca_set_max_depth_uncertainty_ratio(float ratio) {
    if (ratio > 0.0f) kalman_core.max_depth_uncertainty_ratio = ratio;
}

void kca_set_feature_meas_noise(float variance) {
    if (variance > 0.0f) {
        kalman_core.meas_noise_feature = variance;
    }
}

void kca_set_init_stddev(float pos_xy, float pos_z,
                         float vel,
                         float att_rp, float att_yaw) {
    if (pos_xy > 0.0f) kalman_core.init_stddev_pos_xy  = pos_xy;
    if (pos_z  > 0.0f) kalman_core.init_stddev_pos_z   = pos_z;
    if (vel    > 0.0f) kalman_core.init_stddev_vel      = vel;
    if (att_rp > 0.0f) kalman_core.init_stddev_att_rp  = att_rp;
    if (att_yaw> 0.0f) kalman_core.init_stddev_att_yaw = att_yaw;
}

/* -------------------------------------------------------------------------
 * EKF predict step, IMU propagation..
 * args: body-frame imu in [mg] and [dps], imu_to_body_R is applied.
 * ------------------------------------------------------------------------- */
static void kca_predict_impl(float ax_mg, float ay_mg, float az_mg,
                              float gx_dps, float gy_dps, float gz_dps,
                              float dt) {
    apply_body_transform(&ax_mg, &ay_mg, &az_mg);
    apply_body_transform(&gx_dps, &gy_dps, &gz_dps);

    float ax = ax_mg * MG_TO_MS2;
    float ay = ay_mg * MG_TO_MS2;
    float az = az_mg * MG_TO_MS2;
    /* small experiment state of 12, bias-corrected body rate [rad/s], gyro minus estimated bias. not in paper, WIP */
#if KC_STATE_DIM >= 12
    float gx = gx_dps * DEG2RAD - kalman_core.state[KC_STATE_BGX];
    float gy = gy_dps * DEG2RAD - kalman_core.state[KC_STATE_BGY];
    float gz = gz_dps * DEG2RAD - kalman_core.state[KC_STATE_BGZ];
#else
    /* dim-9: no gyro-bias state, raw rate */
    float gx = gx_dps * DEG2RAD;
    float gy = gy_dps * DEG2RAD;
    float gz = gz_dps * DEG2RAD;
#endif

    /* dt already provided */
    float dt2 = dt * dt;

    float (*Rot)[3] = kalman_core.R;
    float *q        = kalman_core.q;

    float px = kalman_core.state[KC_STATE_PX];
    float py = kalman_core.state[KC_STATE_PY];
    float pz = kalman_core.state[KC_STATE_PZ];

    /* FEJ: evaluate the orientation-dependent A blocks at the first-estimate
     * rotation R_fej, not the live R, to keep the yaw nullspace honest.
     * only the jacobian is frozen, the nominal propagation uses the live Rot. */
    float (*Rj)[3] = Rot;
    if (kalman_core.fej) {
        if (!kalman_core.fej_anchored) {
            memcpy(kalman_core.R_fej, kalman_core.R, sizeof(kalman_core.R_fej));
            kalman_core.fej_anchored = 1;
        }
        Rj = kalman_core.R_fej;
    }

    /* linearised jacobian A */
    static float A[KC_STATE_DIM][KC_STATE_DIM];
    memset(A, 0, sizeof(A));
    for (int i = 0; i < KC_STATE_DIM; i++) { A[i][i] = 1.0f; }

    A[0][3] = Rj[0][0]*dt;  A[0][4] = Rj[0][1]*dt;  A[0][5] = Rj[0][2]*dt;
    A[1][3] = Rj[1][0]*dt;  A[1][4] = Rj[1][1]*dt;  A[1][5] = Rj[1][2]*dt;
    A[2][3] = Rj[2][0]*dt;  A[2][4] = Rj[2][1]*dt;  A[2][5] = Rj[2][2]*dt;

    A[0][6] = ( py*Rj[0][2] - pz*Rj[0][1]) * dt;
    A[1][6] = ( py*Rj[1][2] - pz*Rj[1][1]) * dt;
    A[2][6] = ( py*Rj[2][2] - pz*Rj[2][1]) * dt;

    A[0][7] = (-px*Rj[0][2] + pz*Rj[0][0]) * dt;
    A[1][7] = (-px*Rj[1][2] + pz*Rj[1][0]) * dt;
    A[2][7] = (-px*Rj[2][2] + pz*Rj[2][0]) * dt;

    A[0][8] = ( px*Rj[0][1] - py*Rj[0][0]) * dt;
    A[1][8] = ( px*Rj[1][1] - py*Rj[1][0]) * dt;
    A[2][8] = ( px*Rj[2][1] - py*Rj[2][0]) * dt;

    A[3][3] = 1.0f - dt * kalman_core.dragBx;
    A[4][3] = -gz  * dt;
    A[5][3] =  gy  * dt;

    A[3][4] =  gz  * dt;
    A[4][4] = 1.0f - dt * kalman_core.dragBy;
    A[5][4] = -gx  * dt;

    A[3][5] = -gy  * dt;
    A[4][5] =  gx  * dt;
    A[5][5] = 1.0f - dt * kalman_core.dragBz;

    A[3][6] =  0.0f;
    A[4][6] = -GRAVITY_MAGNITUDE * Rj[2][2] * dt;
    A[5][6] =  GRAVITY_MAGNITUDE * Rj[2][1] * dt;

    A[3][7] =  GRAVITY_MAGNITUDE * Rj[2][2] * dt;
    A[4][7] =  0.0f;
    A[5][7] = -GRAVITY_MAGNITUDE * Rj[2][0] * dt;

    A[3][8] = -GRAVITY_MAGNITUDE * Rj[2][1] * dt;
    A[4][8] =  GRAVITY_MAGNITUDE * Rj[2][0] * dt;
    A[5][8] =  0.0f;

    /* no-thrust model: v_z drive is 0, so no gravity-leak into v_z from attitude */
    /* used for ablation, perhaps for later use when we couple motor input as reference*/
    if (kalman_core.no_accel_thrust) { A[5][6] = 0.0f; A[5][7] = 0.0f; }

    float d0 = gx * dt / 2.0f;
    float d1 = gy * dt / 2.0f;
    float d2 = gz * dt / 2.0f;

    A[6][6] =  1.0f - d1*d1/2.0f - d2*d2/2.0f;
    A[6][7] =  d2 + d0*d1/2.0f;
    A[6][8] = -d1 + d0*d2/2.0f;

    A[7][6] = -d2 + d0*d1/2.0f;
    A[7][7] =  1.0f - d0*d0/2.0f - d2*d2/2.0f;
    A[7][8] =  d0 + d1*d2/2.0f;

    A[8][6] =  d1 + d0*d2/2.0f;
    A[8][7] = -d0 + d1*d2/2.0f;
    A[8][8] =  1.0f - d0*d0/2.0f - d1*d1/2.0f;

#if KC_STATE_DIM >= 12
    /* gyro-bias coupling, not in paper. bias enters the rate as gyro minus b_g. */
    A[KC_STATE_D0][KC_STATE_BGX] = -dt;
    A[KC_STATE_D1][KC_STATE_BGY] = -dt;
    A[KC_STATE_D2][KC_STATE_BGZ] = -dt;
    float drz = kalman_core.drag_rz, dry = kalman_core.drag_ry, drx = kalman_core.drag_rx;
    float dBx = kalman_core.dragBx, dBy = kalman_core.dragBy, dBz = kalman_core.dragBz;
    A[KC_STATE_PX][KC_STATE_BGY] = dt * ( pz - dBx*drz);
    A[KC_STATE_PX][KC_STATE_BGZ] = dt * (-py + dBx*dry);
    A[KC_STATE_PY][KC_STATE_BGX] = dt * (-pz + dBy*drz);
    A[KC_STATE_PY][KC_STATE_BGZ] = dt * ( px - dBy*drx);
    A[KC_STATE_PZ][KC_STATE_BGX] = dt * ( py - dBz*dry);
    A[KC_STATE_PZ][KC_STATE_BGY] = dt * (-px + dBz*drx);
#endif

    /* P_rr <- A * P_rr * A^T. keep A*P_rr (tmp1) for the robocentric cross-terms. */
    static float tmp1[KC_STATE_DIM][KC_STATE_DIM];   /* = A * P_rr_old */
    static float tmp2[KC_STATE_DIM][KC_STATE_DIM];
    static float P_rr_old[KC_STATE_DIM][KC_STATE_DIM];
    if (kalman_core.robocentric)
        memcpy(P_rr_old, kalman_core.P, sizeof(P_rr_old));
    mat9_mul(tmp1, A, kalman_core.P);
    mat9_trans(tmp2, A);
    mat9_mul(kalman_core.P, tmp1, tmp2);

    if (!kalman_core.robocentric) {
        /* world-anchored: features are static landmarks, P_rf <- A * P_rf */
        for (int fi = 0; fi < MAX_FEATURES; fi++) {
            if (!kalman_core.features[fi].active) continue;
            float new_Prf[KC_STATE_DIM][3];
            for (int r = 0; r < KC_STATE_DIM; r++)
                for (int c = 0; c < 3; c++) {
                    float s = 0.0f;
                    for (int k = 0; k < KC_STATE_DIM; k++)
                        s += A[r][k] * kalman_core.features[fi].P_rf[k][c];
                    new_Prf[r][c] = s;
                }
            memcpy(kalman_core.features[fi].P_rf, new_Prf, sizeof(new_Prf));
        }
    } else {
        /* robocentric TEST/WIP: features move with the camera, not in paper WIP.
         * propagate (nor,p) and the joint covariance with FD jacobians. */
        float v_b[3]   = { px, py, pz };
        float omega[3] = { gx, gy, gz };
        float Qb = kalman_core.procNoiseFeatBearing * kalman_core.procNoiseFeatBearing * dt;
        float Qd = kalman_core.procNoiseFeatDepth   * kalman_core.procNoiseFeatDepth   * dt;
        const float FEPS = 1e-4f;
        for (int fi = 0; fi < MAX_FEATURES; fi++) {
            feature_state_t *f = &kalman_core.features[fi];
            if (!f->active) continue;
            float nor[3] = { f->h[0], f->h[1], f->h[2] };
            float pf = f->p;

            /* nominal output + its tangent basis */
            float nor0[3], p0;  robo_propagate_feature(nor, pf, v_b, omega, dt, nor0, &p0);
            float T1o[3], T2o[3];  tangent_basis(nor0, T1o, T2o);
            float T1i[3], T2i[3];  tangent_basis(nor,  T1i, T2i);

            float F_ff[3][3];
            float F_fr[3][KC_STATE_DIM];
            for (int r = 0; r < 3; r++) for (int c = 0; c < KC_STATE_DIM; c++) F_fr[r][c] = 0.0f;

            /**/
            #define ROBO_COL(nq, pq, set0, set1, set2) do {                       \
                float dn0 = (nq)[0]-nor0[0], dn1 = (nq)[1]-nor0[1], dn2 = (nq)[2]-nor0[2]; \
                set0 = (T1o[0]*dn0 + T1o[1]*dn1 + T1o[2]*dn2) / FEPS;             \
                set1 = (T2o[0]*dn0 + T2o[1]*dn1 + T2o[2]*dn2) / FEPS;             \
                set2 = ((pq) - p0) / FEPS;                                        \
            } while (0)

            /* F_ff: perturb δ1, δ2 and p */
            for (int col = 0; col < 3; col++) {
                float np[3], pp;
                if (col < 2) {
                    const float *Tp = (col == 0) ? T1i : T2i;
                    float a0 = nor[0]+FEPS*Tp[0], a1 = nor[1]+FEPS*Tp[1], a2 = nor[2]+FEPS*Tp[2];
                    float m = sqrtf(a0*a0+a1*a1+a2*a2)+EPS;
                    np[0]=a0/m; np[1]=a1/m; np[2]=a2/m;  pp = pf;
                } else { np[0]=nor[0]; np[1]=nor[1]; np[2]=nor[2]; pp = pf + FEPS; }
                float nq[3], pq;  robo_propagate_feature(np, pp, v_b, omega, dt, nq, &pq);
                ROBO_COL(nq, pq, F_ff[0][col], F_ff[1][col], F_ff[2][col]);
            }
            /* F_fr: velocity columns (PX,PY,PZ) */
            for (int k = 0; k < 3; k++) {
                float vp[3] = { v_b[0], v_b[1], v_b[2] };  vp[k] += FEPS;
                float nq[3], pq;  robo_propagate_feature(nor, pf, vp, omega, dt, nq, &pq);
                int col = KC_STATE_PX + k;
                ROBO_COL(nq, pq, F_fr[0][col], F_fr[1][col], F_fr[2][col]);
            }
#if KC_STATE_DIM >= 12
            /* F_fr: gyro-bias columns (BGX,BGY,BGZ): ω = gyro−b_g ⇒ ∂/∂b_g = −∂/∂ω */
            for (int k = 0; k < 3; k++) {
                float wp[3] = { omega[0], omega[1], omega[2] };  wp[k] -= FEPS;
                float nq[3], pq;  robo_propagate_feature(nor, pf, v_b, wp, dt, nq, &pq);
                int col = KC_STATE_BGX + k;
                ROBO_COL(nq, pq, F_fr[0][col], F_fr[1][col], F_fr[2][col]);
            }
#endif
            #undef ROBO_COL

            /* ---- Covariance propagation ----
             * P_rf_new = (A P_rf) F_ff^T + (A P_rr) F_fr^T
             * P_ff_new = F_fr P_rr F_fr^T + F_fr P_rf F_ff^T + (·)^T + F_ff P_ff F_ff^T + Q */
            float Prf[KC_STATE_DIM][3];  memcpy(Prf, f->P_rf, sizeof(Prf));
            float Pff[3][3];             memcpy(Pff, f->P_ff, sizeof(Pff));

            /* AP_rf = A * Prf (12×3) */
            float APrf[KC_STATE_DIM][3];
            for (int r = 0; r < KC_STATE_DIM; r++)
                for (int c = 0; c < 3; c++) {
                    float s = 0.0f;
                    for (int k = 0; k < KC_STATE_DIM; k++) s += A[r][k]*Prf[k][c];
                    APrf[r][c] = s;
                }
            /* new P_rf = APrf*F_ff^T + tmp1(=A P_rr)*F_fr^T  (12×3) */
            float newPrf[KC_STATE_DIM][3];
            for (int r = 0; r < KC_STATE_DIM; r++)
                for (int c = 0; c < 3; c++) {
                    float s = 0.0f;
                    for (int k = 0; k < 3; k++) s += APrf[r][k]*F_ff[c][k];          /* F_ff^T */
                    for (int k = 0; k < KC_STATE_DIM; k++) s += tmp1[r][k]*F_fr[c][k]; /* F_fr^T */
                    newPrf[r][c] = s;
                }
            /* Frr_Prr = F_fr * P_rr_old (3×12);  Frr_Prf = F_fr * Prf (3×3) */
            float FrrPrr[3][KC_STATE_DIM];
            for (int r = 0; r < 3; r++)
                for (int c = 0; c < KC_STATE_DIM; c++) {
                    float s = 0.0f;
                    for (int k = 0; k < KC_STATE_DIM; k++) s += F_fr[r][k]*P_rr_old[k][c];
                    FrrPrr[r][c] = s;
                }
            float FrrPrf[3][3];
            for (int r = 0; r < 3; r++)
                for (int c = 0; c < 3; c++) {
                    float s = 0.0f;
                    for (int k = 0; k < KC_STATE_DIM; k++) s += F_fr[r][k]*Prf[k][c];
                    FrrPrf[r][c] = s;
                }
            /* Ta = FrrPrr * F_fr^T (3×3); Tb = FrrPrf * F_ff^T (3×3); Tc = Tb^T */
            float Ta[3][3], Tb[3][3], FffPff[3][3], Td[3][3];
            for (int r = 0; r < 3; r++)
                for (int c = 0; c < 3; c++) {
                    float sa = 0.0f, sb = 0.0f;
                    for (int k = 0; k < KC_STATE_DIM; k++) sa += FrrPrr[r][k]*F_fr[c][k];
                    for (int k = 0; k < 3; k++)            sb += FrrPrf[r][k]*F_ff[c][k];
                    Ta[r][c] = sa;  Tb[r][c] = sb;
                }
            /* FffPff = F_ff * Pff (3×3); Td = FffPff * F_ff^T */
            for (int r = 0; r < 3; r++)
                for (int c = 0; c < 3; c++) {
                    float s = 0.0f;
                    for (int k = 0; k < 3; k++) s += F_ff[r][k]*Pff[k][c];
                    FffPff[r][c] = s;
                }
            for (int r = 0; r < 3; r++)
                for (int c = 0; c < 3; c++) {
                    float s = 0.0f;
                    for (int k = 0; k < 3; k++) s += FffPff[r][k]*F_ff[c][k];
                    Td[r][c] = s;
                }
            float newPff[3][3];
            for (int r = 0; r < 3; r++)
                for (int c = 0; c < 3; c++)
                    newPff[r][c] = Ta[r][c] + Tb[r][c] + Tb[c][r] + Td[r][c];
            newPff[0][0] += Qb;  newPff[1][1] += Qb;  newPff[2][2] += Qd;

            memcpy(f->P_rf, newPrf, sizeof(newPrf));
            memcpy(f->P_ff, newPff, sizeof(newPff));

            /* propagate the feature state (nor, p) */
            f->h[0] = nor0[0];  f->h[1] = nor0[1];  f->h[2] = nor0[2];  f->p = p0;
        }
    }

    /* state propagation */
    float dx, dy, dz;
    if (kalman_core.is_flying) {
        /* no_accel_thrust: assume thrust = gravity so the v_z drive is 0.
         * measured a_z ->no longer enter v_z. */
        float zacc = kalman_core.no_accel_thrust ? (GRAVITY_MAGNITUDE * Rot[2][2]) : az;
        dx = px * dt;
        dy = py * dt;
        dz = pz * dt + zacc * dt2 / 2.0f;
        kalman_core.state[KC_STATE_X] += Rot[0][0]*dx + Rot[0][1]*dy + Rot[0][2]*dz;
        kalman_core.state[KC_STATE_Y] += Rot[1][0]*dx + Rot[1][1]*dy + Rot[1][2]*dz;
        kalman_core.state[KC_STATE_Z] += Rot[2][0]*dx + Rot[2][1]*dy + Rot[2][2]*dz - GRAVITY_MAGNITUDE * dt2 / 2.0f;

        float odr_x = gy * kalman_core.drag_rz - gz * kalman_core.drag_ry;
        float odr_y = gz * kalman_core.drag_rx - gx * kalman_core.drag_rz;
        float odr_z = gx * kalman_core.drag_ry - gy * kalman_core.drag_rx;

        kalman_core.state[KC_STATE_PX] += dt * ( gz*py - gy*pz - GRAVITY_MAGNITUDE*Rot[2][0]
                                                  - kalman_core.dragBx*px + kalman_core.dragBx*odr_x);
        kalman_core.state[KC_STATE_PY] += dt * (-gz*px + gx*pz - GRAVITY_MAGNITUDE*Rot[2][1]
                                                  - kalman_core.dragBy*py + kalman_core.dragBy*odr_y);
        kalman_core.state[KC_STATE_PZ] += dt * (zacc + gy*px - gx*py - GRAVITY_MAGNITUDE*Rot[2][2]
                                                  - kalman_core.dragBz*pz + kalman_core.dragBz*odr_z);
    } else {
        /* we actually do not use this, but */
        dx = px * dt + ax * dt2 / 2.0f;
        dy = py * dt + ay * dt2 / 2.0f;
        dz = pz * dt + az * dt2 / 2.0f;
        kalman_core.state[KC_STATE_X] += Rot[0][0]*dx + Rot[0][1]*dy + Rot[0][2]*dz;
        kalman_core.state[KC_STATE_Y] += Rot[1][0]*dx + Rot[1][1]*dy + Rot[1][2]*dz;
        kalman_core.state[KC_STATE_Z] += Rot[2][0]*dx + Rot[2][1]*dy + Rot[2][2]*dz - GRAVITY_MAGNITUDE * dt2 / 2.0f;

        kalman_core.state[KC_STATE_PX] += dt * (ax + gz*py - gy*pz - GRAVITY_MAGNITUDE*Rot[2][0]);
        kalman_core.state[KC_STATE_PY] += dt * (ay - gz*px + gx*pz - GRAVITY_MAGNITUDE*Rot[2][1]);
        kalman_core.state[KC_STATE_PZ] += dt * (az + gy*px - gx*py - GRAVITY_MAGNITUDE*Rot[2][2]);
    }

    /* small detour, wip not in paper*/
    if (kalman_core.vel_max > 0.0f) {
        float vx = kalman_core.state[KC_STATE_PX];
        float vy = kalman_core.state[KC_STATE_PY];
        float vz = kalman_core.state[KC_STATE_PZ];
        float sp = sqrtf(vx*vx + vy*vy + vz*vz);
        if (sp > kalman_core.vel_max) {
            float s = kalman_core.vel_max / sp;
            kalman_core.state[KC_STATE_PX] = vx * s;
            kalman_core.state[KC_STATE_PY] = vy * s;
            kalman_core.state[KC_STATE_PZ] = vz * s;
        }
    }

    /* quaternion integration */
    float dtwx = dt * gx;
    float dtwy = dt * gy;
    float dtwz = dt * gz;
    float angle = sqrtf(dtwx*dtwx + dtwy*dtwy + dtwz*dtwz) + EPS;
    float ca = cosf(angle / 2.0f);
    float sa = sinf(angle / 2.0f);
    float dq[4] = {ca, sa*dtwx/angle, sa*dtwy/angle, sa*dtwz/angle};

    float tmpq0 = dq[0]*q[0] - dq[1]*q[1] - dq[2]*q[2] - dq[3]*q[3];
    float tmpq1 = dq[1]*q[0] + dq[0]*q[1] + dq[3]*q[2] - dq[2]*q[3];
    float tmpq2 = dq[2]*q[0] - dq[3]*q[1] + dq[0]*q[2] + dq[1]*q[3];
    float tmpq3 = dq[3]*q[0] + dq[2]*q[1] - dq[1]*q[2] + dq[0]*q[3];

    if (!kalman_core.is_flying) {
        float keep = 1.0f - kalman_core.attitude_reversion;
        tmpq0 = keep*tmpq0 + kalman_core.attitude_reversion * kalman_core.initial_quat[0];
        tmpq1 = keep*tmpq1 + kalman_core.attitude_reversion * kalman_core.initial_quat[1];
        tmpq2 = keep*tmpq2 + kalman_core.attitude_reversion * kalman_core.initial_quat[2];
        tmpq3 = keep*tmpq3 + kalman_core.attitude_reversion * kalman_core.initial_quat[3];
    }

    float norm = sqrtf(tmpq0*tmpq0 + tmpq1*tmpq1 + tmpq2*tmpq2 + tmpq3*tmpq3) + EPS;
    q[0] = tmpq0 / norm;
    q[1] = tmpq1 / norm;
    q[2] = tmpq2 / norm;
    q[3] = tmpq3 / norm;

    kalman_core.is_updated = true;
}

/* predict using the configured fixed dt, causes problems, when going back onboard */
void kca_predict(float ax_mg, float ay_mg, float az_mg,
                  float gx_dps, float gy_dps, float gz_dps) {
    if (!kalman_core.initialized) return;
    float dt = 1.0f / kalman_core.update_frequency_hz;
    kca_predict_impl(ax_mg, ay_mg, az_mg, gx_dps, gy_dps, gz_dps, dt);
}

/* predict with caller-supplied dt in seconds, for replay, real dt better*/
void kca_predict_dt(float ax_mg, float ay_mg, float az_mg,
                     float gx_dps, float gy_dps, float gz_dps, float dt_s) {
    if (!kalman_core.initialized) return;
    kca_predict_impl(ax_mg, ay_mg, az_mg, gx_dps, gy_dps, gz_dps, dt_s);
}

/* -------------------------------------------------------------------------
 * EKF finalize step, attitude -> quaternion
 * ------------------------------------------------------------------------- */
bool kca_finalize(void) {
    if (!kalman_core.initialized || !kalman_core.is_updated) return false;

    float *q      = kalman_core.q;
    float (*Rot)[3] = kalman_core.R;

    float v0 = kalman_core.state[KC_STATE_D0];
    float v1 = kalman_core.state[KC_STATE_D1];
    float v2 = kalman_core.state[KC_STATE_D2];

    if ((fabsf(v0) > 1e-4f || fabsf(v1) > 1e-4f || fabsf(v2) > 1e-4f) &&
        (fabsf(v0) < 10.0f  && fabsf(v1) < 10.0f  && fabsf(v2) < 10.0f)) {

        float angle = sqrtf(v0*v0 + v1*v1 + v2*v2) + EPS;
        float ca = cosf(angle / 2.0f);
        float sa = sinf(angle / 2.0f);
        float dq[4] = {ca, sa*v0/angle, sa*v1/angle, sa*v2/angle};

        float tmpq0 = dq[0]*q[0] - dq[1]*q[1] - dq[2]*q[2] - dq[3]*q[3];
        float tmpq1 = dq[1]*q[0] + dq[0]*q[1] + dq[3]*q[2] - dq[2]*q[3];
        float tmpq2 = dq[2]*q[0] - dq[3]*q[1] + dq[0]*q[2] + dq[1]*q[3];
        float tmpq3 = dq[3]*q[0] + dq[2]*q[1] - dq[1]*q[2] + dq[0]*q[3];

        float norm = sqrtf(tmpq0*tmpq0 + tmpq1*tmpq1 + tmpq2*tmpq2 + tmpq3*tmpq3) + EPS;
        q[0] = tmpq0 / norm;
        q[1] = tmpq1 / norm;
        q[2] = tmpq2 / norm;
        q[3] = tmpq3 / norm;

        float d0 = v0 / 2.0f;
        float d1 = v1 / 2.0f;
        float d2 = v2 / 2.0f;

        static float A_cov[KC_STATE_DIM][KC_STATE_DIM];
        memset(A_cov, 0, sizeof(A_cov));
        for (int i = 0; i < KC_STATE_DIM; i++) A_cov[i][i] = 1.0f;  /* identity */

        A_cov[6][6] =  1.0f - d1*d1/2.0f - d2*d2/2.0f;
        A_cov[6][7] =  d2 + d0*d1/2.0f;
        A_cov[6][8] = -d1 + d0*d2/2.0f;

        A_cov[7][6] = -d2 + d0*d1/2.0f;
        A_cov[7][7] =  1.0f - d0*d0/2.0f - d2*d2/2.0f;
        A_cov[7][8] =  d0 + d1*d2/2.0f;

        A_cov[8][6] =  d1 + d0*d2/2.0f;
        A_cov[8][7] = -d0 + d1*d2/2.0f;
        A_cov[8][8] =  1.0f - d0*d0/2.0f - d1*d1/2.0f;

        static float tmp1[KC_STATE_DIM][KC_STATE_DIM];
        static float tmp2[KC_STATE_DIM][KC_STATE_DIM];
        mat9_mul(tmp1, A_cov, kalman_core.P);
        mat9_trans(tmp2, A_cov);
        mat9_mul(kalman_core.P, tmp1, tmp2);

        /* rotate P_rf attitude rows for active features */
        for (int fi = 0; fi < MAX_FEATURES; fi++) {
            if (!kalman_core.features[fi].active) continue;
            float prf_tmp[3][3];
            for (int i = 0; i < 3; i++) {
                for (int c = 0; c < 3; c++) {
                    prf_tmp[i][c] = A_cov[6+i][6]*kalman_core.features[fi].P_rf[6][c]
                                  + A_cov[6+i][7]*kalman_core.features[fi].P_rf[7][c]
                                  + A_cov[6+i][8]*kalman_core.features[fi].P_rf[8][c];
                }
            }
            memcpy(&kalman_core.features[fi].P_rf[6], prf_tmp, sizeof(prf_tmp));
        }
    }

    /* rebuild rotation matrix from the updated quaternion */
    Rot[0][0] = q[0]*q[0] + q[1]*q[1] - q[2]*q[2] - q[3]*q[3];
    Rot[0][1] = 2.0f*(q[1]*q[2] - q[0]*q[3]);
    Rot[0][2] = 2.0f*(q[1]*q[3] + q[0]*q[2]);
    Rot[1][0] = 2.0f*(q[1]*q[2] + q[0]*q[3]);
    Rot[1][1] = q[0]*q[0] - q[1]*q[1] + q[2]*q[2] - q[3]*q[3];
    Rot[1][2] = 2.0f*(q[2]*q[3] - q[0]*q[1]);
    Rot[2][0] = 2.0f*(q[1]*q[3] - q[0]*q[2]);
    Rot[2][1] = 2.0f*(q[2]*q[3] + q[0]*q[1]);
    Rot[2][2] = q[0]*q[0] - q[1]*q[1] - q[2]*q[2] + q[3]*q[3];

    kalman_core.state[KC_STATE_D0] = 0.0f;
    kalman_core.state[KC_STATE_D1] = 0.0f;
    kalman_core.state[KC_STATE_D2] = 0.0f;

    mat9_symmetrize_clamp(kalman_core.P);
    kalman_core.is_updated = false;
    return true;
}

/* -------------------------------------------------------------------------
 * set attitude from a quaternion, for gravity-aligned init.
 * normalises q, rebuilds R, zeros the attitude error. call after kca_reset().
 * ------------------------------------------------------------------------- */
void kca_set_attitude_quat(float qw, float qx, float qy, float qz) {
    if (!kalman_core.initialized) return;

    float norm = sqrtf(qw*qw + qx*qx + qy*qy + qz*qz) + EPS;
    float *q = kalman_core.q;
    q[0] = qw / norm;
    q[1] = qx / norm;
    q[2] = qy / norm;
    q[3] = qz / norm;

    float (*Rot)[3] = kalman_core.R;
    Rot[0][0] = q[0]*q[0] + q[1]*q[1] - q[2]*q[2] - q[3]*q[3];
    Rot[0][1] = 2.0f*(q[1]*q[2] - q[0]*q[3]);
    Rot[0][2] = 2.0f*(q[1]*q[3] + q[0]*q[2]);
    Rot[1][0] = 2.0f*(q[1]*q[2] + q[0]*q[3]);
    Rot[1][1] = q[0]*q[0] - q[1]*q[1] + q[2]*q[2] - q[3]*q[3];
    Rot[1][2] = 2.0f*(q[2]*q[3] - q[0]*q[1]);
    Rot[2][0] = 2.0f*(q[1]*q[3] - q[0]*q[2]);
    Rot[2][1] = 2.0f*(q[2]*q[3] + q[0]*q[1]);
    Rot[2][2] = q[0]*q[0] - q[1]*q[1] - q[2]*q[2] + q[3]*q[3];

    kalman_core.state[KC_STATE_D0] = 0.0f;
    kalman_core.state[KC_STATE_D1] = 0.0f;
    kalman_core.state[KC_STATE_D2] = 0.0f;
}

/* -------------------------------------------------------------------------
 * process noise, added to the covariance diagonal each step
 * ------------------------------------------------------------------------- */
void kca_add_process_noise(float dt_ms) {
    if (!kalman_core.initialized || dt_ms <= 0.0f) return;

    float dt    = dt_ms / 1000.0f;
    float dt3   = dt * dt * dt;
    float na_xy = kalman_core.procNoiseAcc_xy;
    float na_z  = kalman_core.procNoiseAcc_z;
    float nv    = kalman_core.procNoiseVel;
    float np    = kalman_core.procNoisePos;
    float nat   = kalman_core.procNoiseAtt;
    float ngr   = kalman_core.measNoiseGyro_rollpitch;
    float ngy   = kalman_core.measNoiseGyro_yaw;

    /* (variable dt) -> continuous-time discretisation, variance added per step scales with dt.
     * params are spectral densities [noise/√Hz] so tuned values transfer across rates. */
    float var_pos_xy = (na_xy*na_xy + nv*nv) * dt3 / 3.0f + (np*np) * dt;
    float var_pos_z  = (na_z *na_z  + nv*nv) * dt3 / 3.0f + (np*np) * dt;
    float var_vel_xy = (na_xy*na_xy + nv*nv) * dt;
    float var_vel_z  = (na_z *na_z  + nv*nv) * dt;
    float var_att_rp = (ngr*ngr + nat*nat) * dt;
    float var_att_yaw= (ngy*ngy + nat*nat) * dt;

    kalman_core.P[KC_STATE_X ][KC_STATE_X ] += var_pos_xy;
    kalman_core.P[KC_STATE_Y ][KC_STATE_Y ] += var_pos_xy;
    kalman_core.P[KC_STATE_Z ][KC_STATE_Z ] += var_pos_z;
    kalman_core.P[KC_STATE_PX][KC_STATE_PX] += var_vel_xy;
    kalman_core.P[KC_STATE_PY][KC_STATE_PY] += var_vel_xy;
    kalman_core.P[KC_STATE_PZ][KC_STATE_PZ] += var_vel_z;
    kalman_core.P[KC_STATE_D0][KC_STATE_D0] += var_att_rp;
    kalman_core.P[KC_STATE_D1][KC_STATE_D1] += var_att_rp;
    kalman_core.P[KC_STATE_D2][KC_STATE_D2] += var_att_yaw;

#if KC_STATE_DIM >= 12
    /* gyro-bias random walk, 0=frozen. not in paper */
    float nbg = kalman_core.procNoiseGyroBias;
    float var_bg = (nbg * nbg) * dt;
    kalman_core.P[KC_STATE_BGX][KC_STATE_BGX] += var_bg;
    kalman_core.P[KC_STATE_BGY][KC_STATE_BGY] += var_bg;
    kalman_core.P[KC_STATE_BGZ][KC_STATE_BGZ] += var_bg;
#endif

    mat9_symmetrize_clamp(kalman_core.P);
}

/* ---- State access ------------------------------------------------------ */

void kca_get_state(float state_out[9]) {
    /* 9 robot states only, gyro-bias is internal */
    for (int i = 0; i < 9; i++) state_out[i] = kalman_core.state[i];
}

void kca_get_quaternion(float q_out[4]) {
    q_out[0] = kalman_core.q[0];
    q_out[1] = kalman_core.q[1];
    q_out[2] = kalman_core.q[2];
    q_out[3] = kalman_core.q[3];
}

void kca_get_rotation_matrix(float R_out[9]) {
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            R_out[i*3+j] = kalman_core.R[i][j];
}

void kca_get_full_covariance(float P_out[81]) {
    /* 9×9 robot block only, gyro-bias covariance is internal */
    for (int i = 0; i < 9; i++)
        for (int j = 0; j < 9; j++)
            P_out[i*9+j] = kalman_core.P[i][j];
}

/* ---- feature map ------------------------------------------------------- */

/* ---- adaptive depth init ---------------------------------------------- */
/* only use mean rho when enough features with high enough certainty */
#define MIN_FEATURES_FOR_MEDIAN      3

static int _cmp_float(const void *a, const void *b) {
    float fa = *(const float *)a;
    float fb = *(const float *)b;
    return (fa > fb) - (fa < fb);
}

/* median inverse depth of active low-uncertainty features.
 * falls back to 'fallback' if too few qualify. */
float kca_compute_median_rho(float fallback) {
    if (!kalman_core.initialized) return fallback;
    float rho_buf[MAX_FEATURES];
    int count = 0;
    for (int i = 0; i < MAX_FEATURES; i++) {
        feature_state_t *f = &kalman_core.features[i];
        if (!f->active) continue;
        float d = depth_d_from_p(f->p);                 /* distance from world origin */
        if (!isfinite(d) || d < EPS) continue;
        /* relative depth uncertainty σ_d/d, parametrization-invariant */
        float sigma_d = sqrtf(fabsf(f->P_ff[2][2])) * fabsf(depth_ddist_dp(f->p));
        if ((sigma_d / d) < kalman_core.max_depth_uncertainty_ratio)
            rho_buf[count++] = 1.0f / d;                /* return inverse depth */
    }
    if (count < MIN_FEATURES_FOR_MEDIAN) return fallback;
    qsort(rho_buf, count, sizeof(float), _cmp_float);
    return rho_buf[count / 2];
}

int kca_add_feature(float u, float v, float rho_init) {
    if (!kalman_core.initialized) return -1;
    if (rho_init < 0.0f) return -1;
    if (rho_init == 0.0f) rho_init = kca_compute_median_rho(kalman_core.fallback_rho);

    int slot = -1;
    for (int i = 0; i < MAX_FEATURES; i++) {
        if (!kalman_core.features[i].active) { slot = i; break; }
    }
    if (slot < 0) return -2;  /* pool full */

    float zx, zy;
    pixel_to_normalised(u, v, &zx, &zy);

    float depth = 1.0f / rho_init;

    if (kalman_core.robocentric) {
        /* camera-frame anchoring, not in paper WIP. bearing = pixel ray, depth = camera distance.
         * R_cb is not applied here, features live entirely in the camera frame. */
        float nrm = sqrtf(zx*zx + zy*zy + 1.0f);
        feature_state_t *f = &kalman_core.features[slot];
        f->h[0] = zx/nrm;  f->h[1] = zy/nrm;  f->h[2] = 1.0f/nrm;
        float d = depth * nrm;                 /* distance from camera */
        f->p = depth_p_from_d(d);
        memset(f->P_ff, 0, sizeof(f->P_ff));
        f->P_ff[0][0] = INIT_STDDEV_BEARING * INIT_STDDEV_BEARING;
        f->P_ff[1][1] = INIT_STDDEV_BEARING * INIT_STDDEV_BEARING;
        float sig_p = kalman_core.init_stddev_idepth * d * depth_dp_ddist(d);
        f->P_ff[2][2] = sig_p * sig_p;
        memset(f->P_rf, 0, sizeof(f->P_rf));
        f->active = true;
        kalman_core.active_feature_count++;
        return slot;
    }

    float Xc[3] = { zx * depth, zy * depth, depth };   /* camera-frame ray, depth along optical axis */

    float (*Rotm)[3] = kalman_core.R;
    float (*Rcb)[3]  = kalman_core.R_cb;
    float tx = kalman_core.state[KC_STATE_X];
    float ty = kalman_core.state[KC_STATE_Y];
    float tz = kalman_core.state[KC_STATE_Z];

    /* camera→body: Xb = R_cb · Xc */
    float Xb[3];
    for (int i = 0; i < 3; i++)
        Xb[i] = Rcb[i][0]*Xc[0] + Rcb[i][1]*Xc[1] + Rcb[i][2]*Xc[2];

    float Xw[3];
    for (int i = 0; i < 3; i++)
        Xw[i] = Rotm[i][0]*Xb[0] + Rotm[i][1]*Xb[1] + Rotm[i][2]*Xb[2];
    Xw[0] += tx;  Xw[1] += ty;  Xw[2] += tz;

    if (Xw[2] < EPS) return -3;  /* behind world Z=0 */

    /* store bearing as unit vector on S², depth as the parameter p */
    float dist = sqrtf(Xw[0]*Xw[0] + Xw[1]*Xw[1] + Xw[2]*Xw[2]);
    if (dist < EPS) return -3;
    feature_state_t *f = &kalman_core.features[slot];
    f->h[0] = Xw[0] / dist;
    f->h[1] = Xw[1] / dist;
    f->h[2] = Xw[2] / dist;
    f->p    = depth_p_from_d(dist);  /* depth parameter (INVERSE => 1/dist) */

    memset(f->P_ff, 0, sizeof(f->P_ff));
    f->P_ff[0][0] = INIT_STDDEV_BEARING * INIT_STDDEV_BEARING;
    f->P_ff[1][1] = INIT_STDDEV_BEARING * INIT_STDDEV_BEARING;
    /* init_stddev_idepth is a relative depth uncertainty, converted to parameter space */
    float d_init = 1.0f / rho_init;
    float sig_p  = kalman_core.init_stddev_idepth * d_init * depth_dp_ddist(d_init);
    f->P_ff[2][2] = sig_p * sig_p;

    /* seed robot<->feature cross-cov so the landmark inherits pose uncertainty.
     * shelved behind prf_seed, not in paper. correct but fragile on dynamic flights. */
    if (kalman_core.prf_seed) {
        float T1[3], T2[3];
        tangent_basis(f->h, T1, T2);
        /* dXw/datt = -R*[Xb]x ; dXw/dpos = I ; dXw/dvel = 0 */
        float sk[3][3] = { { 0.0f, -Xb[2], Xb[1] }, { Xb[2], 0.0f, -Xb[0] }, { -Xb[1], Xb[0], 0.0f } };
        float dXw_da[3][3];
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++) {
                float s = 0.0f;
                for (int k = 0; k < 3; k++) s += Rotm[i][k] * sk[k][j];
                dXw_da[i][j] = -s;
            }
        /* df/dXw : rows (T1^T/dist, T2^T/dist, (dp/ddist)*h^T) */
        float dpdd = depth_dp_ddist(dist);
        float dfdX[3][3];
        for (int j = 0; j < 3; j++) {
            dfdX[0][j] = T1[j] / dist;
            dfdX[1][j] = T2[j] / dist;
            dfdX[2][j] = dpdd * f->h[j];
        }
        /* Gx (3 x KC_STATE_DIM): pos cols = df/dXw*I ; vel = 0 ; att cols = df/dXw*dXw_da */
        float Gx[3][KC_STATE_DIM];
        memset(Gx, 0, sizeof(Gx));
        for (int r = 0; r < 3; r++) {
            Gx[r][KC_STATE_X] = dfdX[r][0];
            Gx[r][KC_STATE_Y] = dfdX[r][1];
            Gx[r][KC_STATE_Z] = dfdX[r][2];
            for (int c = 0; c < 3; c++) {
                float s = 0.0f;
                for (int k = 0; k < 3; k++) s += dfdX[r][k] * dXw_da[k][c];
                Gx[r][KC_STATE_D0 + c] = s;
            }
        }
        /* P_rf = P_rr * Gx^T  (KC_STATE_DIM x 3) */
        for (int i = 0; i < KC_STATE_DIM; i++)
            for (int r = 0; r < 3; r++) {
                float s = 0.0f;
                for (int k = 0; k < KC_STATE_DIM; k++) s += kalman_core.P[i][k] * Gx[r][k];
                f->P_rf[i][r] = s;
            }
        /* P_ff += Gx * P_rr * Gx^T = Gx * P_rf  (3x3) */
        for (int a = 0; a < 3; a++)
            for (int b = 0; b < 3; b++) {
                float s = 0.0f;
                for (int k = 0; k < KC_STATE_DIM; k++) s += Gx[a][k] * f->P_rf[k][b];
                f->P_ff[a][b] += s;
            }
    } else {
        memset(f->P_rf, 0, sizeof(f->P_rf));
    }

    f->active = true;
    kalman_core.active_feature_count++;
    return slot;
}

int kca_feature_update(int id, float u, float v, float meas_var,
                        float *rx_out, float *ry_out, float *d2_out) {
    if (d2_out) *d2_out = -1.0f;   /* invalid update */
    if (!kalman_core.initialized) return -1;
    if (id < 0 || id >= MAX_FEATURES || !kalman_core.features[id].active) return -1;

    float obs_x, obs_y;
    pixel_to_normalised(u, v, &obs_x, &obs_y);
    if (meas_var <= 0.0f) meas_var = kalman_core.meas_noise_feature;

    feature_state_t *feat = &kalman_core.features[id];

    float *h  = feat->h;   /* unit bearing (world frame if anchored, camera frame if robocentric) */
    float p   = feat->p;   /* depth parameter */

    /* tangent basis {T1,T2} at the bearing, used by Hf and the S² bearing update */
    float T1[3], T2[3];
    tangent_basis(h, T1, T2);

    float zhat[2], r[2];
    float Hr[2][KC_STATE_DIM];
    float Hf[2][3];
    memset(Hr, 0, sizeof(Hr));

    if (kalman_core.robocentric) {
        /* camera-frame feature, not in paper WIP. project the bearing directly, depth cancels.
         * measurement depends only on bearing so Hr=0, robot corrected through P_rf. */
        if (fabsf(h[2]) < EPS) { if (rx_out) *rx_out = 0.0f; if (ry_out) *ry_out = 0.0f; return 0; }
        float inv_z = 1.0f / h[2], inv_z2 = inv_z * inv_z;
        zhat[0] = h[0] * inv_z;  zhat[1] = h[1] * inv_z;
        r[0] = obs_x - zhat[0];  r[1] = obs_y - zhat[1];
        float Jn[2][3] = { { inv_z, 0.0f, -h[0]*inv_z2 }, { 0.0f, inv_z, -h[1]*inv_z2 } };
        for (int m = 0; m < 2; m++) {
            Hf[m][0] = Jn[m][0]*T1[0] + Jn[m][1]*T1[1] + Jn[m][2]*T1[2];
            Hf[m][1] = Jn[m][0]*T2[0] + Jn[m][1]*T2[1] + Jn[m][2]*T2[2];
            Hf[m][2] = 0.0f;
        }
    } else {
        /* world-anchored: Xw = h·dist projected through the camera pose */
        float (*Rot)[3] = kalman_core.R;
        float tx = kalman_core.state[KC_STATE_X];
        float ty = kalman_core.state[KC_STATE_Y];
        float tz = kalman_core.state[KC_STATE_Z];
        float dist     = depth_d_from_p(p);
        float ddist_dp = depth_ddist_dp(p);
        if (!isfinite(dist) || dist < EPS) { if (rx_out) *rx_out = 0.0f; if (ry_out) *ry_out = 0.0f; return 0; }

        float Xw[3] = { h[0] * dist, h[1] * dist, h[2] * dist };
        float delta[3] = { Xw[0] - tx, Xw[1] - ty, Xw[2] - tz };
        float Xc[3];
        for (int i = 0; i < 3; i++)
            Xc[i] = Rot[0][i]*delta[0] + Rot[1][i]*delta[1] + Rot[2][i]*delta[2];

        /* body→camera: Xcam = R_cb^T · Xc, project the pinhole in the camera frame */
        float (*Rcb)[3] = kalman_core.R_cb;
        float Xcam[3];
        for (int i = 0; i < 3; i++)
            Xcam[i] = Rcb[0][i]*Xc[0] + Rcb[1][i]*Xc[1] + Rcb[2][i]*Xc[2];
        if (Xcam[2] < EPS) { if (rx_out) *rx_out = 0.0f; if (ry_out) *ry_out = 0.0f; return 0; }

        float inv_z = 1.0f / Xcam[2], inv_z2 = inv_z * inv_z;
        zhat[0] = Xcam[0] * inv_z;  zhat[1] = Xcam[1] * inv_z;
        r[0] = obs_x - zhat[0];   r[1] = obs_y - zhat[1];

        float Jpi[2][3] = { { inv_z, 0.0f, -Xcam[0]*inv_z2 }, { 0.0f, inv_z, -Xcam[1]*inv_z2 } };
        /* fold the camera→body rotation into the projection jacobian: J' = Jpi · R_cb^T */
        float Jp[2][3];
        for (int m = 0; m < 2; m++)
            for (int k = 0; k < 3; k++)
                Jp[m][k] = Jpi[m][0]*Rcb[k][0] + Jpi[m][1]*Rcb[k][1] + Jpi[m][2]*Rcb[k][2];

        for (int m = 0; m < 2; m++)
            for (int k = 0; k < 3; k++)
                Hr[m][k] = -(Jp[m][0]*Rot[k][0] + Jp[m][1]*Rot[k][1] + Jp[m][2]*Rot[k][2]);
        float skew[3][3] = {
            {  0.0f,   -Xc[2],  Xc[1] }, {  Xc[2],   0.0f,  -Xc[0] }, { -Xc[1],   Xc[0],  0.0f }
        };
        for (int m = 0; m < 2; m++)
            for (int k = 0; k < 3; k++)
                Hr[m][6+k] = Jp[m][0]*skew[0][k] + Jp[m][1]*skew[1][k] + Jp[m][2]*skew[2][k];

        float dXc_df[3][3];
        for (int i = 0; i < 3; i++) {
            float rotT1 = Rot[0][i]*T1[0] + Rot[1][i]*T1[1] + Rot[2][i]*T1[2];
            float rotT2 = Rot[0][i]*T2[0] + Rot[1][i]*T2[1] + Rot[2][i]*T2[2];
            float rotH  = Rot[0][i]*h[0]  + Rot[1][i]*h[1]  + Rot[2][i]*h[2];
            dXc_df[i][0] = rotT1 * dist;
            dXc_df[i][1] = rotT2 * dist;
            dXc_df[i][2] = rotH * ddist_dp;
        }
        for (int m = 0; m < 2; m++)
            for (int k = 0; k < 3; k++)
                Hf[m][k] = Jp[m][0]*dXc_df[0][k] + Jp[m][1]*dXc_df[1][k] + Jp[m][2]*dXc_df[2][k];
    }

    float PHT_r[KC_STATE_DIM][2];
    for (int ri = 0; ri < KC_STATE_DIM; ri++) {
        for (int c = 0; c < 2; c++) {
            float s = 0.0f;
            for (int k = 0; k < KC_STATE_DIM; k++) s += kalman_core.P[ri][k] * Hr[c][k];
            for (int k = 0; k < 3; k++)            s += feat->P_rf[ri][k]    * Hf[c][k];
            PHT_r[ri][c] = s;
        }
    }

    float PHT_f[3][2];
    for (int ri = 0; ri < 3; ri++) {
        for (int c = 0; c < 2; c++) {
            float s = 0.0f;
            for (int k = 0; k < KC_STATE_DIM; k++) s += feat->P_rf[k][ri] * Hr[c][k];
            for (int k = 0; k < 3; k++)            s += feat->P_ff[ri][k] * Hf[c][k];
            PHT_f[ri][c] = s;
        }
    }

    float S[2][2];
    for (int a = 0; a < 2; a++) {
        for (int b = 0; b < 2; b++) {
            float s = (a == b) ? meas_var : 0.0f;
            for (int k = 0; k < KC_STATE_DIM; k++) s += Hr[a][k] * PHT_r[k][b];
            for (int k = 0; k < 3; k++)            s += Hf[a][k] * PHT_f[k][b];
            S[a][b] = s;
        }
    }

    float det = S[0][0]*S[1][1] - S[0][1]*S[1][0];
    if (fabsf(det) < EPS) {
        if (rx_out) *rx_out = r[0];
        if (ry_out) *ry_out = r[1];
        return 0;
    }
    float inv_det = 1.0f / det;
    float Si[2][2] = {
        {  S[1][1]*inv_det, -S[0][1]*inv_det },
        { -S[1][0]*inv_det,  S[0][0]*inv_det }
    };

    /* normalized innovation squared (NIS), always computed and exported for diagnostics */
    float d2 = r[0]*(Si[0][0]*r[0] + Si[0][1]*r[1])
             + r[1]*(Si[1][0]*r[0] + Si[1][1]*r[1]);
    if (d2_out) *d2_out = d2;

    /* mahalanobis distance gate */
    if (kalman_core.innovation_gate > 0.0f && d2 > kalman_core.innovation_gate) {
        if (rx_out) *rx_out = r[0];
        if (ry_out) *ry_out = r[1];
        return 2;  /* gated — update skipped */
    }

    /* This was a small detour: huber robust down-weighting, not in paper WIP. inflates R on a high-innovation obs.
       off when huber_delta<=0, the gate above stays the hard backstop. */
    if (kalman_core.huber_delta > 0.0f && d2 > kalman_core.huber_delta * kalman_core.huber_delta) {
        float w   = kalman_core.huber_delta / sqrtf(d2);   /* <1 */
        float add = meas_var * (1.0f / w - 1.0f);          /* extra R on the diagonal */
        float h00 = S[0][0] + add, h11 = S[1][1] + add, h01 = S[0][1], h10 = S[1][0];
        float hd  = h00*h11 - h01*h10;
        if (fabsf(hd) >= EPS) {
            float ih = 1.0f / hd;
            Si[0][0] =  h11*ih;  Si[0][1] = -h01*ih;
            Si[1][0] = -h10*ih;  Si[1][1] =  h00*ih;
        }
    }

    float Kr[KC_STATE_DIM][2];
    for (int ri = 0; ri < KC_STATE_DIM; ri++) {
        Kr[ri][0] = PHT_r[ri][0]*Si[0][0] + PHT_r[ri][1]*Si[1][0];
        Kr[ri][1] = PHT_r[ri][0]*Si[0][1] + PHT_r[ri][1]*Si[1][1];
    }
    float Kf[3][2];
    for (int ri = 0; ri < 3; ri++) {
        Kf[ri][0] = PHT_f[ri][0]*Si[0][0] + PHT_f[ri][1]*Si[1][0];
        Kf[ri][1] = PHT_f[ri][0]*Si[0][1] + PHT_f[ri][1]*Si[1][1];
    }

    for (int i = 0; i < KC_STATE_DIM; i++)
        kalman_core.state[i] += Kr[i][0]*r[0] + Kr[i][1]*r[1];

    /* S² bearing update: move along the tangent plane then renormalise onto the sphere */
    float delta1 = Kf[0][0]*r[0] + Kf[0][1]*r[1];
    float delta2 = Kf[1][0]*r[0] + Kf[1][1]*r[1];
    float hn0 = feat->h[0] + T1[0]*delta1 + T2[0]*delta2;
    float hn1 = feat->h[1] + T1[1]*delta1 + T2[1]*delta2;
    float hn2 = feat->h[2] + T1[2]*delta1 + T2[2]*delta2;
    float nn  = sqrtf(hn0*hn0 + hn1*hn1 + hn2*hn2);
    if (nn > EPS) {
        feat->h[0] = hn0/nn;  feat->h[1] = hn1/nn;  feat->h[2] = hn2/nn;
    }
    feat->p += Kf[2][0]*r[0] + Kf[2][1]*r[1];

    for (int i = 0; i < KC_STATE_DIM; i++)
        for (int j = 0; j < KC_STATE_DIM; j++)
            kalman_core.P[i][j] -= PHT_r[i][0]*Kr[j][0] + PHT_r[i][1]*Kr[j][1];

    for (int i = 0; i < KC_STATE_DIM; i++)
        for (int j = 0; j < 3; j++)
            feat->P_rf[i][j] -= PHT_r[i][0]*Kf[j][0] + PHT_r[i][1]*Kf[j][1];

    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            feat->P_ff[i][j] -= PHT_f[i][0]*Kf[j][0] + PHT_f[i][1]*Kf[j][1];

    /* update cross-covariance for all other active features */
    for (int j = 0; j < MAX_FEATURES; j++) {
        if (j == id || !kalman_core.features[j].active) continue;
        float HrPrf[2][3];
        for (int m = 0; m < 2; m++) {
            for (int c = 0; c < 3; c++) {
                float s = 0.0f;
                for (int k = 0; k < KC_STATE_DIM; k++)
                    s += Hr[m][k] * kalman_core.features[j].P_rf[k][c];
                HrPrf[m][c] = s;
            }
        }
        for (int ri = 0; ri < KC_STATE_DIM; ri++)
            for (int c = 0; c < 3; c++)
                kalman_core.features[j].P_rf[ri][c] -=
                    Kr[ri][0]*HrPrf[0][c] + Kr[ri][1]*HrPrf[1][c];
    }

    mat9_symmetrize_clamp(kalman_core.P);

    /* symmetrise and clamp P_ff */
    for (int i = 0; i < 3; i++) {
        for (int j = i; j < 3; j++) {
            float p = 0.5f * feat->P_ff[i][j] + 0.5f * feat->P_ff[j][i];
            if (isnan(p) || p > MAX_COVARIANCE) {
                feat->P_ff[i][j] = feat->P_ff[j][i] = MAX_COVARIANCE;
            } else if (i == j && p < MIN_COVARIANCE) {
                feat->P_ff[i][j] = MIN_COVARIANCE;
            } else {
                feat->P_ff[i][j] = feat->P_ff[j][i] = p;
            }
        }
    }

    kalman_core.is_updated = true;

    if (rx_out) *rx_out = r[0];
    if (ry_out) *ry_out = r[1];
    return 0;
}

/* -------------------------------------------------------------------------
 * gravity / accel attitude reference, corrects roll/pitch in flight.
 * meas_var should be large since the instant accel is noisy, a magnitude gate
 * rejects high-acceleration samples. see methodology.
 * ------------------------------------------------------------------------- */
void kca_gravity_update(float ax_mg, float ay_mg, float az_mg, float meas_var, float mag_tol) {
    if (!kalman_core.initialized || meas_var <= 0.0f) return;
    apply_body_transform(&ax_mg, &ay_mg, &az_mg);
    float ax = ax_mg*MG_TO_MS2, ay = ay_mg*MG_TO_MS2, az = az_mg*MG_TO_MS2;
    float amag = sqrtf(ax*ax + ay*ay + az*az);
    if (amag < EPS) return;
    /* reject samples whose magnitude is far from g, linear acceleration present */
    if (mag_tol > 0.0f && fabsf(amag - GRAVITY_MAGNITUDE) > mag_tol) return;

    float ahat[3] = { ax/amag, ay/amag, az/amag };          /* measured up, body */
    float (*R)[3] = kalman_core.R;
    float ghat[3] = { R[2][0], R[2][1], R[2][2] };          /* predicted up, body */
    float r[3] = { ahat[0]-ghat[0], ahat[1]-ghat[1], ahat[2]-ghat[2] };

    /* H = [ĝ]×, only on the attitude-error columns */
    float Ha[3][3] = {
        {  0.0f,    -ghat[2],  ghat[1] },
        {  ghat[2],  0.0f,    -ghat[0] },
        { -ghat[1],  ghat[0],  0.0f    }
    };

    /* PHT (KC_STATE_DIM×3) = P[:,6:9] · Ha^T */
    float PHT[KC_STATE_DIM][3];
    for (int i = 0; i < KC_STATE_DIM; i++)
        for (int c = 0; c < 3; c++)
            PHT[i][c] = kalman_core.P[i][6]*Ha[c][0]
                      + kalman_core.P[i][7]*Ha[c][1]
                      + kalman_core.P[i][8]*Ha[c][2];

    /* S = Ha · P[6:9,:] · Ha^T + meas_var·I  (= Ha rows of PHT) */
    float S[3][3];
    for (int a = 0; a < 3; a++)
        for (int b = 0; b < 3; b++) {
            float s = (a == b) ? meas_var : 0.0f;
            s += Ha[a][0]*PHT[6][b] + Ha[a][1]*PHT[7][b] + Ha[a][2]*PHT[8][b];
            S[a][b] = s;
        }

    /* 3×3 inverse via cofactors */
    float c00 =  (S[1][1]*S[2][2]-S[1][2]*S[2][1]);
    float c01 = -(S[1][0]*S[2][2]-S[1][2]*S[2][0]);
    float c02 =  (S[1][0]*S[2][1]-S[1][1]*S[2][0]);
    float det = S[0][0]*c00 + S[0][1]*c01 + S[0][2]*c02;
    if (fabsf(det) < EPS) return;
    float invd = 1.0f/det;
    float Si[3][3] = {
        { c00*invd, (S[0][2]*S[2][1]-S[0][1]*S[2][2])*invd, (S[0][1]*S[1][2]-S[0][2]*S[1][1])*invd },
        { c01*invd, (S[0][0]*S[2][2]-S[0][2]*S[2][0])*invd, (S[0][2]*S[1][0]-S[0][0]*S[1][2])*invd },
        { c02*invd, (S[0][1]*S[2][0]-S[0][0]*S[2][1])*invd, (S[0][0]*S[1][1]-S[0][1]*S[1][0])*invd }
    };

    /* K = PHT · Si  (KC_STATE_DIM×3) */
    float K[KC_STATE_DIM][3];
    for (int i = 0; i < KC_STATE_DIM; i++)
        for (int c = 0; c < 3; c++)
            K[i][c] = PHT[i][0]*Si[0][c] + PHT[i][1]*Si[1][c] + PHT[i][2]*Si[2][c];

    for (int i = 0; i < KC_STATE_DIM; i++)
        kalman_core.state[i] += K[i][0]*r[0] + K[i][1]*r[1] + K[i][2]*r[2];

    /* P -= K · PHT^T */
    for (int i = 0; i < KC_STATE_DIM; i++)
        for (int j = 0; j < KC_STATE_DIM; j++)
            kalman_core.P[i][j] -= K[i][0]*PHT[j][0] + K[i][1]*PHT[j][1] + K[i][2]*PHT[j][2];

    /* cross-covariance to features: P_rf -= K · (Ha · P_rf[6:9]) */
    for (int fi = 0; fi < MAX_FEATURES; fi++) {
        if (!kalman_core.features[fi].active) continue;
        float (*Prf)[3] = kalman_core.features[fi].P_rf;
        float HPrf[3][3];
        for (int c = 0; c < 3; c++)
            for (int fc = 0; fc < 3; fc++)
                HPrf[c][fc] = Ha[c][0]*Prf[6][fc] + Ha[c][1]*Prf[7][fc] + Ha[c][2]*Prf[8][fc];
        for (int i = 0; i < KC_STATE_DIM; i++)
            for (int fc = 0; fc < 3; fc++)
                Prf[i][fc] -= K[i][0]*HPrf[0][fc] + K[i][1]*HPrf[1][fc] + K[i][2]*HPrf[2][fc];
    }

    mat9_symmetrize_clamp(kalman_core.P);
    kalman_core.is_updated = true;
}

/* DOES NOT WORK YET: event-flow velocity update, not in paper WIP. per-track pixel flow as a 2D
 * measurement, jacobian on the velocity block only. world-anchored features only. */
int kca_flow_update(int id, float flow_x_px, float flow_y_px,
                    float gx_dps, float gy_dps, float gz_dps, float meas_var) {
    if (!kalman_core.flow_update || !kalman_core.initialized || meas_var <= 0.0f) return -1;
    if (id < 0 || id >= MAX_FEATURES || !kalman_core.features[id].active) return -1;
    if (kalman_core.robocentric) return -1;  /* model is world-anchored */

    /* gyro -> body rate [rad/s] (same transform predict applies), then camera frame */
    apply_body_transform(&gx_dps, &gy_dps, &gz_dps);
    float wb[3] = { gx_dps*DEG2RAD, gy_dps*DEG2RAD, gz_dps*DEG2RAD };
    float (*Rcb)[3] = kalman_core.R_cb;
    float wc[3];  /* w_cam = R_cb^T · w_body */
    for (int i = 0; i < 3; i++) wc[i] = Rcb[0][i]*wb[0] + Rcb[1][i]*wb[1] + Rcb[2][i]*wb[2];

    /* feature projection (world-anchored), same geometry as kca_feature_update */
    feature_state_t *feat = &kalman_core.features[id];
    float *h = feat->h;
    float dist = depth_d_from_p(feat->p);
    if (!isfinite(dist) || dist < EPS) return -1;
    float (*Rot)[3] = kalman_core.R;
    float delta[3] = { h[0]*dist - kalman_core.state[KC_STATE_X],
                       h[1]*dist - kalman_core.state[KC_STATE_Y],
                       h[2]*dist - kalman_core.state[KC_STATE_Z] };
    float Xc[3];
    for (int i = 0; i < 3; i++) Xc[i] = Rot[0][i]*delta[0] + Rot[1][i]*delta[1] + Rot[2][i]*delta[2];
    float Xcam[3];
    for (int i = 0; i < 3; i++) Xcam[i] = Rcb[0][i]*Xc[0] + Rcb[1][i]*Xc[1] + Rcb[2][i]*Xc[2];
    if (Xcam[2] < EPS) return -1;
    float invZ = 1.0f / Xcam[2];
    float x = Xcam[0]*invZ, y = Xcam[1]*invZ;

    /* body velocity in camera frame: v_cam = R_cb^T · v_body */
    float vb[3] = { kalman_core.state[KC_STATE_PX], kalman_core.state[KC_STATE_PY], kalman_core.state[KC_STATE_PZ] };
    float vc[3];
    for (int i = 0; i < 3; i++) vc[i] = Rcb[0][i]*vb[0] + Rcb[1][i]*vb[1] + Rcb[2][i]*vb[2];

    /* predicted normalised flow: translational (x 1/Z) + rotational (gyro offset) */
    float rot_x = x*y*wc[0] - (1.0f + x*x)*wc[1] + y*wc[2];
    float rot_y = (1.0f + y*y)*wc[0] - x*y*wc[1] - x*wc[2];
    float pred_x = invZ*(-vc[0] + x*vc[2]) + rot_x;
    float pred_y = invZ*(-vc[1] + y*vc[2]) + rot_y;

    /* measured flow px/s -> normalised (ignore mild distortion derivative) */
    float fmx = flow_x_px / kalman_core.cam_fx;
    float fmy = flow_y_px / kalman_core.cam_fy;
    float r[2] = { fmx - pred_x, fmy - pred_y };

    /* jacobian on the velocity block (cols 3:5): Hv = invZ · A(x,y) · R_cb^T,
     * A = [[-1,0,x],[0,-1,y]].  (A·R_cb^T)[m][j] uses Rcb[j][k]. */
    float Hv[2][3];
    for (int j = 0; j < 3; j++) {
        Hv[0][j] = invZ * (-Rcb[j][0] + x*Rcb[j][2]);
        Hv[1][j] = invZ * (-Rcb[j][1] + y*Rcb[j][2]);
    }

    /* PHT = P[:,3:6] · Hv^T  (KC_STATE_DIM×2) */
    float PHT[KC_STATE_DIM][2];
    for (int i = 0; i < KC_STATE_DIM; i++)
        for (int c = 0; c < 2; c++)
            PHT[i][c] = kalman_core.P[i][3]*Hv[c][0] + kalman_core.P[i][4]*Hv[c][1] + kalman_core.P[i][5]*Hv[c][2];

    /* S = Hv · P[3:6,:] · Hv^T + meas_var·I  (2×2) */
    float S[2][2];
    for (int a = 0; a < 2; a++)
        for (int b = 0; b < 2; b++) {
            float s = (a == b) ? meas_var : 0.0f;
            s += Hv[a][0]*PHT[3][b] + Hv[a][1]*PHT[4][b] + Hv[a][2]*PHT[5][b];
            S[a][b] = s;
        }
    float det = S[0][0]*S[1][1] - S[0][1]*S[1][0];
    if (fabsf(det) < EPS) return -1;
    float invd = 1.0f / det;
    float Si[2][2] = { {  S[1][1]*invd, -S[0][1]*invd }, { -S[1][0]*invd,  S[0][0]*invd } };

    /* innovation gate (chi² 2-dof), reuse the feature gate */
    float d2 = r[0]*(Si[0][0]*r[0] + Si[0][1]*r[1]) + r[1]*(Si[1][0]*r[0] + Si[1][1]*r[1]);
    if (kalman_core.innovation_gate > 0.0f && d2 > kalman_core.innovation_gate) return 2;

    /* K = PHT · Si  (KC_STATE_DIM×2) */
    float K[KC_STATE_DIM][2];
    for (int i = 0; i < KC_STATE_DIM; i++) {
        K[i][0] = PHT[i][0]*Si[0][0] + PHT[i][1]*Si[1][0];
        K[i][1] = PHT[i][0]*Si[0][1] + PHT[i][1]*Si[1][1];
    }
    for (int i = 0; i < KC_STATE_DIM; i++)
        kalman_core.state[i] += K[i][0]*r[0] + K[i][1]*r[1];

    /* P -= K · PHT^T */
    for (int i = 0; i < KC_STATE_DIM; i++)
        for (int j = 0; j < KC_STATE_DIM; j++)
            kalman_core.P[i][j] -= K[i][0]*PHT[j][0] + K[i][1]*PHT[j][1];

    /* robot-feature cross-cov: P_rf -= K · (Hv · P_rf[3:6]) for all active features */
    for (int fi = 0; fi < MAX_FEATURES; fi++) {
        if (!kalman_core.features[fi].active) continue;
        float (*Prf)[3] = kalman_core.features[fi].P_rf;
        float HPrf[2][3];
        for (int c = 0; c < 2; c++)
            for (int fc = 0; fc < 3; fc++)
                HPrf[c][fc] = Hv[c][0]*Prf[3][fc] + Hv[c][1]*Prf[4][fc] + Hv[c][2]*Prf[5][fc];
        for (int i = 0; i < KC_STATE_DIM; i++)
            for (int fc = 0; fc < 3; fc++)
                Prf[i][fc] -= K[i][0]*HPrf[0][fc] + K[i][1]*HPrf[1][fc];
    }

    mat9_symmetrize_clamp(kalman_core.P);
    kalman_core.is_updated = true;
    return 0;
}

int kca_remove_feature(int id) {
    if (id < 0 || id >= MAX_FEATURES || !kalman_core.features[id].active) return -1;
    kalman_core.features[id].active = false;
    kalman_core.active_feature_count--;
    return 0;
}

int kca_get_active_feature_count(void) {
    return kalman_core.active_feature_count;
}

int kca_get_feature(int id, float feature_out[6]) {
    if (id < 0 || id >= MAX_FEATURES || !kalman_core.features[id].active) return -1;
    feature_state_t *f = &kalman_core.features[id];
    float dist     = depth_d_from_p(f->p);     /* distance from world origin */
    float inv_h2   = 1.0f / (fabsf(f->h[2]) + EPS);
    feature_out[0] = f->h[0] * inv_h2;         /* bx = h0/h2 (flat projection) */
    feature_out[1] = f->h[1] * inv_h2;         /* by = h1/h2 */
    feature_out[2] = 1.0f / (fabsf(dist) + EPS);  /* inverse depth (rho) */
    feature_out[3] = f->h[0] * dist;           /* Xw */
    feature_out[4] = f->h[1] * dist;           /* Yw */
    feature_out[5] = f->h[2] * dist;           /* Zw */
    return 0;
}

int kca_get_all_features(float *features_out, int max_count) {
    int count = 0;
    for (int i = 0; i < MAX_FEATURES && count < max_count; i++) {
        if (!kalman_core.features[i].active) continue;
        feature_state_t *f = &kalman_core.features[i];
        float dist = depth_d_from_p(f->p);   /* distance from world origin */
        float *out  = features_out + count * 7;
        float inv_h2 = 1.0f / (fabsf(f->h[2]) + EPS);
        out[0] = (float)i;
        out[1] = f->h[0] * inv_h2;      /* bx = h0/h2 (flat projection) */
        out[2] = f->h[1] * inv_h2;      /* by = h1/h2 */
        out[3] = 1.0f / (fabsf(dist) + EPS);  /* inverse depth (rho) */
        out[4] = f->h[0] * dist;        /* Xw */
        out[5] = f->h[1] * dist;        /* Yw */
        out[6] = f->h[2] * dist;        /* Zw */
        count++;
    }
    return count;
}

int kca_get_feature_cov_diag(int id, float diag_out[3]) {
    if (id < 0 || id >= MAX_FEATURES || !kalman_core.features[id].active) return -1;
    feature_state_t *f = &kalman_core.features[id];
    diag_out[0] = f->P_ff[0][0];  /* var(bx)  */
    diag_out[1] = f->P_ff[1][1];  /* var(by)  */
    diag_out[2] = f->P_ff[2][2];  /* var(rho) */
    return 0;
}

/* chi2 tesst(2, p=0.95)=5.991 Set 0.0 to disable. */
void kca_set_innovation_gate(float threshold) {
    kalman_core.innovation_gate = (threshold > 0.0f) ? threshold : 0.0f;
}

void kca_set_vel_max(float vel_max) {
    kalman_core.vel_max = (vel_max > 0.0f) ? vel_max : 0.0f;
}

void kca_set_depth_type(int type) {
    if (type >= DEPTH_REGULAR && type <= DEPTH_HYPERBOLIC)
        kalman_core.depth_type = type;
}

void kca_set_gyro_bias_noise(float sigma) {
    kalman_core.procNoiseGyroBias = (sigma > 0.0f) ? sigma : 0.0f;
}

void kca_set_init_stddev_gyro_bias(float stddev) {
    kalman_core.init_stddev_gyro_bias = (stddev > 0.0f) ? stddev : 0.0f;
}

void kca_get_gyro_bias(float bias_out[3]) {
#if KC_STATE_DIM >= 12
    bias_out[0] = kalman_core.state[KC_STATE_BGX];
    bias_out[1] = kalman_core.state[KC_STATE_BGY];
    bias_out[2] = kalman_core.state[KC_STATE_BGZ];
#else
    bias_out[0] = bias_out[1] = bias_out[2] = 0.0f;  /* dim-9: no bias state */
#endif
}

void kca_set_robocentric(int enable) {
    kalman_core.robocentric = enable ? 1 : 0;
}

void kca_set_prf_seed(int enable) {
    kalman_core.prf_seed = enable ? 1 : 0;
}

void kca_set_fej(int enable) {
    kalman_core.fej = enable ? 1 : 0;
}

void kca_set_flow_update(int enable) {
    kalman_core.flow_update = enable ? 1 : 0;
}

void kca_set_no_accel_thrust(int enable) {
    kalman_core.no_accel_thrust = enable ? 1 : 0;
}

void kca_set_huber_delta(float delta) {
    kalman_core.huber_delta = (delta > 0.0f) ? delta : 0.0f;
}

void kca_set_feat_process_noise(float bearing, float depth) {
    if (bearing >= 0.0f) kalman_core.procNoiseFeatBearing = bearing;
    if (depth   >= 0.0f) kalman_core.procNoiseFeatDepth   = depth;
}

