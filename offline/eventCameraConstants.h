
#ifndef __EVENT_CAMERA_CONSTANTS_H__
#define __EVENT_CAMERA_CONSTANTS_H__

/* focal lengths [pixels] */
#define CAM_FX  (184.46f)
#define CAM_FY  (182.85f)

/* principal point [pixels] */
#define CAM_CX  (160.37f)
#define CAM_CY  (153.25f)

/* distortion coefficients, opencv convention (k1,k2,p1,p2,k3) */
#define CAM_K1  ( 0.083f)
#define CAM_K2  (-0.171f)
#define CAM_P1  (-0.0017f)
#define CAM_P2  (-0.0024f)
#define CAM_K3  ( 0.201f)

#endif /* __EVENT_CAMERA_CONSTANTS_H__ */
