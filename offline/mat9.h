/*
 * 9x9 float matrix helpers for the EKF Kalman core.
 */

#ifndef MAT9_H
#define MAT9_H

#include "physicalConstants.h"
#include <math.h>

/* C = A * B  (9x9 float multiply) */
static inline void mat9_mul(float C[KC_STATE_DIM][KC_STATE_DIM],
                             const float A[KC_STATE_DIM][KC_STATE_DIM],
                             const float B[KC_STATE_DIM][KC_STATE_DIM]) {
    for (int i = 0; i < KC_STATE_DIM; i++) {
        for (int j = 0; j < KC_STATE_DIM; j++) {
            float s = 0.0f;
            for (int k = 0; k < KC_STATE_DIM; k++) {
                s += A[i][k] * B[k][j];
            }
            C[i][j] = s;
        }
    }
}

/* B = A^T  (9x9 float transpose) */
static inline void mat9_trans(float B[KC_STATE_DIM][KC_STATE_DIM], const float A[KC_STATE_DIM][KC_STATE_DIM]) {
    for (int i = 0; i < KC_STATE_DIM; i++) {
        for (int j = 0; j < KC_STATE_DIM; j++) {
            B[i][j] = A[j][i];
        }
    }
}

/*
 * symmetrize and clamp a 9x9 covariance matrix P in-place, each entry
 * becomes (P[i][j] + P[j][i]) / 2. diagonal is clamped to
 * [MIN_COVARIANCE, MAX_COVARIANCE], off-diagonal to MAX_COVARIANCE on NaN.
 */
static inline void mat9_symmetrize_clamp(float P[KC_STATE_DIM][KC_STATE_DIM]) {
    for (int i = 0; i < KC_STATE_DIM; i++) {
        for (int j = i; j < KC_STATE_DIM; j++) {
            float p = 0.5f * P[i][j] + 0.5f * P[j][i];
            if (isnan(p) || p > MAX_COVARIANCE) {
                P[i][j] = P[j][i] = MAX_COVARIANCE;
            } else if (i == j && p < MIN_COVARIANCE) {
                P[i][j] = P[j][i] = MIN_COVARIANCE;
            } else {
                P[i][j] = P[j][i] = p;
            }
        }
    }
}

#endif /* MAT9_H */
