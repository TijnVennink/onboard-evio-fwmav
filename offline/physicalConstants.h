#ifndef __PHYSICAL_CONSTANTS_H__
#define __PHYSICAL_CONSTANTS_H__

// gravitational acceleration [m/s^2]
#define GRAVITY_MAGNITUDE (9.81f)

// -------------------------------------------------------------------------
// default drag coefficients for the Flapper Nimble+
// (from platform_defaults_flapper.h in the Crazyflie firmware)
// -------------------------------------------------------------------------

// linear drag coefficients [1/s], applied to body-frame velocity
#define EKF_DRAG_BX (4.2f)
#define EKF_DRAG_BY (1.8f)
#define EKF_DRAG_BZ (0.3f)  /* onboard value (was 0.9) */

// centre-of-pressure to centre-of-mass offset vector [m], used for the
// drag torque correction odr = omega x r_CoP.
// NOTE: EKF_DRAG_RX is not constant, it depends on the dihedral angle and
//       is updated at runtime via set_drag_params().
#define EKF_DRAG_RX (0.0f)
#define EKF_DRAG_RY (0.0f)
#define EKF_DRAG_RZ (0.03f)  /* onboard value (was 0.06) */

// -------------------------------------------------------------------------
// covariance bounds (matching kalman_core_methodA.c)
// -------------------------------------------------------------------------
#define MAX_COVARIANCE (100.0f)
#define MIN_COVARIANCE (1e-6f)

// small epsilon to prevent division by zero
#define EPS (1e-6f)

#endif // __PHYSICAL_CONSTANTS_H__
