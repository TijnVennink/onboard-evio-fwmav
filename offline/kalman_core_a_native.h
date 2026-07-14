#pragma once
/*
 * kalman_core_a_native.h
 *
 * pure-C interface to the EKF core, same math as the onboard!
 *
 * build shared library:
 *   gcc -O2 -fPIC -shared -I. -o kalman_core_a.so kalman_core_a_native.c -lm
 *
 * predict() takes pre-recorded body-frame imu directly. recordings from
 * record_features_imu.py are already in body frame, so imu_to_body_R stays identity(onboard we do use a transform for this!)
 */

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- cycle --------------------------------------------------------- */

void  kca_init(void);
void  kca_reset(float x, float y, float z, float yaw_rad);
bool  kca_is_initialized(void);

/* ---- configuration ----------------------------------------------------- */

/* row-major 3×3, defaults to identity */
void  kca_set_imu_to_body_rotation(float r00, float r01, float r02,
                                    float r10, float r11, float r12,
                                    float r20, float r21, float r22);

/* row-major 3×3 camera→body rotation, maps the optical axis onto the body axis */
void  kca_set_camera_to_body(float r00, float r01, float r02,
                             float r10, float r11, float r12,
                             float r20, float r21, float r22);

void  kca_set_update_rate(float hz);
float kca_get_update_period_ms(void);

/* all 6 drag params required, pass 0 for unused ones */
void  kca_set_drag_params(float bx, float by, float bz,
                           float rx, float ry, float rz);

/* all 8 noise params required */
void  kca_set_process_noise(float acc_xy, float acc_z, float vel, float pos,
                              float att, float gyro_rp, float gyro_yaw,
                              float att_reversion);

void  kca_reset_params(void);
void  kca_set_flying(bool flying);

/* all 9 intrinsic params required, 0 distortion disables it */
void  kca_set_camera_intrinsics(float fx, float fy, float cx, float cy,
                                  float k1, float k2, float p1, float p2,
                                  float k3);

void  kca_set_fallback_rho(float fallback);
void  kca_set_feature_meas_noise(float variance);
/* init covariance std-devs applied on every kca_reset(), pass 0 to keep current */
void  kca_set_init_stddev(float pos_xy, float pos_z,
                          float vel,
                          float att_rp, float att_yaw);
/* relative std-dev for inverse-depth covariance at feature add, default 0.001 */
void  kca_set_init_stddev_idepth(float stddev);
/* relative-depth threshold for the median-rho, only features below it count.
 * default 0.5, lower = more conservative. */
void  kca_set_max_depth_uncertainty_ratio(float ratio);

/* ---- EKF steps --------------------------------------------------------- */

/*
 * predict(), imu propagation. pass body-frame imu in CSV (for offline running) units:
 *   ax_mg, ay_mg, az_mg  : accelerometer [mg]
 *   gx_dps, gy_dps, gz_dps : gyroscope [deg/s]
 * imu_to_body_R is applied internally, leave it identity for body-frame recordings.
 */
void  kca_predict(float ax_mg, float ay_mg, float az_mg,
                   float gx_dps, float gy_dps, float gz_dps);
/* same as kca_predict but with explicit dt in seconds, for offline replay */
void  kca_predict_dt(float ax_mg, float ay_mg, float az_mg,
                      float gx_dps, float gy_dps, float gz_dps, float dt_s);

/* returns true if a prediction was pending, false if predict() was not called */
bool  kca_finalize(void);

/* override attitude after kca_reset() for gravity-aligned init.
   normalises q, rebuilds R, zeros the attitude error. */
void  kca_set_attitude_quat(float qw, float qx, float qy, float qz);

void  kca_add_process_noise(float dt_ms);

/* ---- state access ------------------------------------------------------ */

/* state_out[9] = [x, y, z, px, py, pz, d0, d1, d2] */
void  kca_get_state(float state_out[9]);

/* q_out[4] = [w, x, y, z] */
void  kca_get_quaternion(float q_out[4]);

/* R_out[9] = row-major 3×3 body→world rotation */
void  kca_get_rotation_matrix(float R_out[9]);

/* P_out[81] = row-major 9×9 covariance */
void  kca_get_full_covariance(float P_out[81]);

/* ---- feature map ------------------------------------------------------- */

/*
 * add_feature(), init a new feature from a pixel observation.
 *   u, v      : pixel coordinates
 *   rho_init  : inverse depth [1/m], positive. pass 0 for the adaptive median.
 * returns slot id on success, -1 not initialized, -2 pool full, -3 behind camera.
 */
int   kca_add_feature(float u, float v, float rho_init);

/*
 * median rho of active low-uncertainty features, falls back to 'fallback'.
 * called by kca_add_feature() when rho_init == 0.
 */
float kca_compute_median_rho(float fallback);

/*
 * feature_update(), EKF measurement update for one feature.
 *   meas_var : measurement noise variance, 0 uses the module default
 *   rx, ry   : innovation residual out-params, may be NULL
 * returns 0 on success, 2 if gated, -1 if id is invalid/inactive.
 */
int   kca_feature_update(int id, float u, float v, float meas_var,
                          float *rx, float *ry, float *d2_out);

/*
 * set_innovation_gate(), mahalanobis d² threshold, obs above it are skipped.
 * chi²(2,0.95)=5.991  chi²(2,0.99)=9.210. pass 0 to disable.
 */
void  kca_set_innovation_gate(float threshold);

/* WIPL velocity clamp [m/s] at the end of predict, bounds the gravity-leak runaway.
 * can only reduce |v|. pass 0 to disable. */
void  kca_set_vel_max(float vel_max);

/* depth parametrization: 0=REGULAR, 1=INVERSE, 2=LOG, 3=HYPERBOLIC.
 * only inverse used in paper, others WIP. */
void  kca_set_depth_type(int type);

/* gyro-bias estimation, not in paper. both default 0 = disabled, bias frozen at 0.
 *   gyro_bias_noise: random walk [rad/s/√Hz]
 *   init_stddev_gyro_bias: initial bias std [rad/s] */
void  kca_set_gyro_bias_noise(float sigma);
void  kca_set_init_stddev_gyro_bias(float stddev);
void  kca_get_gyro_bias(float bias_out[3]);

/* robocentric feature model, not in paper WIP. 0=world-anchored, 1=camera-frame.
 * feat_process_noise: per-step bearing/depth random walk. */
void  kca_set_robocentric(int enable);
void  kca_set_feat_process_noise(float bearing, float depth);

/* gravity/accel attitude reference, corrects roll/pitch. call once per predict.
 * meas_var large = weak, mag_tol rejects |accel| far from g [m/s²], 0=off. */
void  kca_gravity_update(float ax_mg, float ay_mg, float az_mg, float meas_var, float mag_tol);

/* returns 0 on success, -1 if id is invalid/inactive */
int   kca_remove_feature(int id);

int   kca_get_active_feature_count(void);

/* retrieve [bx, by, rho, Xw, Yw, Zw] for slot id. returns 0, or -1 if inactive. */
int   kca_get_feature(int id, float feature_out[6]);

/*
 * fill features_out with active.
 * returns the number copied as well max_count).
 */
int   kca_get_all_features(float *features_out, int max_count);

/* retrieve P_ff diagonal [var_bx, var_by, var_rho]. returns 0, or -1 if inactive. */
int   kca_get_feature_cov_diag(int id, float diag_out[3]);

#ifdef __cplusplus
}
#endif
