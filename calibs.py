# -*- coding: utf-8 -*-
"""
calibration matrices taken from kitti's .txt files
reformatted to fit our coordinates
calib_extrinsics translates lidar points to camera POV
calib_projections translates lidar points to image space (third value is depth)
imgshapes is number of pixels in image, used to remove truncated objects
view_by_day is tangent of camera view, used to prune estimates
"""
import numpy as np

# kitti camera reference is right-down-forward, ours is forward-left-up
cam2straight = np.array([[0.,0,1,0],[-1,0,0,0],[0,-1,0,0],[0,0,0,1]])

calib_extrinsics = []
calib_projections = []
imgshapes = []
view_by_day = []

# 09_26
velo2cam = [[7.533745e-03, -9.999714e-01, -6.166020e-04, -4.069766e-03],
            [1.480249e-02, 7.280733e-04, -9.998902e-01, -7.631618e-02],
            [9.998621e-01, 7.523790e-03, 1.480755e-02, -2.717806e-01],[0,0,0,1]]
Rrect0 = [[9.999239e-01, 9.837760e-03, -7.445048e-03, 0],
         [-9.869795e-03, 9.999421e-01, -4.278459e-03, 0],
         [7.402527e-03, 4.351614e-03, 9.999631e-01, 0], [0,0,0,1]]
Prect2 = [[7.215377e+02, 0.000000e+00, 6.095593e+02, 4.485728e+01],
          [0.000000e+00, 7.215377e+02, 1.728540e+02, 2.163791e-01],
          [0.000000e+00, 0.000000e+00, 1.000000e+00, 2.745884e-03]]
total = np.dot(np.dot(Prect2, Rrect0), velo2cam)[[1,0,2]]
calib_extrinsics.append(np.dot(cam2straight, velo2cam))
calib_projections.append(total)
imgshapes.append((375,1242))
view_by_day.append(1242*.5/721.5)

# 09_28
velo2cam = [[6.927964e-03, -9.999722e-01, -2.757829e-03, -2.457729e-02],
            [-1.162982e-03, 2.749836e-03, -9.999955e-01, -6.127237e-02],
            [9.999753e-01, 6.931141e-03, -1.143899e-03, -3.321029e-01],[0,0,0,1]]
Rrect0 = [[9.999128e-01, 1.009263e-02, -8.511932e-03, 0],
          [-1.012729e-02, 9.999406e-01, -4.037671e-03, 0],
          [8.470675e-03, 4.123522e-03, 9.999556e-01, 0], [0,0,0,1]]
Prect2 = [[7.070493e+02, 0.000000e+00, 6.040814e+02, 4.575831e+01],
          [0.000000e+00, 7.070493e+02, 1.805066e+02, -3.454157e-01],
          [0.000000e+00, 0.000000e+00, 1.000000e+00, 4.981016e-03]]
total = np.dot(np.dot(Prect2, Rrect0), velo2cam)[[1,0,2]]
calib_extrinsics.append(np.dot(cam2straight, velo2cam))
calib_projections.append(total)
imgshapes.append((370,1224))
view_by_day.append(1224*.5/707)

# 09_29
velo2cam = [[7.755449e-03, -9.999694e-01, -1.014303e-03, -7.275538e-03],
            [2.294056e-03, 1.032122e-03, -9.999968e-01, -6.324057e-02],
            [9.999673e-01, 7.753097e-03, 2.301990e-03, -2.670414e-01],[0,0,0,1]]
Rrect0 = [[9.999478e-01, 9.791707e-03, -2.925305e-03, 0],
          [-9.806939e-03, 9.999382e-01, -5.238719e-03, 0],
          [2.873828e-03, 5.267134e-03, 9.999820e-01, 0], [0,0,0,1]]
Prect2 = [[7.183351e+02, 0.000000e+00, 6.003891e+02, 4.450382e+01],
          [0.000000e+00, 7.183351e+02, 1.815122e+02, -5.951107e-01],
          [0.000000e+00, 0.000000e+00, 1.000000e+00, 2.616315e-03]]
total = np.dot(np.dot(Prect2, Rrect0), velo2cam)[[1,0,2]]
calib_extrinsics.append(np.dot(cam2straight, velo2cam))
calib_projections.append(total)
imgshapes.append((374,1238))
view_by_day.append(1238*.5/718.3)

# 09_30
velo2cam = [[7.027555000e-03, -9.99975300e-01, 2.59961600e-05, -7.13774800e-03],
            [-2.25483700e-03, -4.18431200e-05, -9.99997500e-01, -7.48265600e-02],
            [9.99972800e-01, 7.02747900e-03, -2.25507500e-03, -3.33632400e-01],
            [0,0,0,1]]
Rrect0 = [[9.99928000e-01, 8.08598500e-03, -8.86679700e-03, 0],
          [-8.12320500e-03, 9.99958300e-01, -4.16975000e-03, 0],
          [8.83271100e-03, 4.24147700e-03, 9.99952000e-01, 0],
          [0, 0, 0, 1]]
Prect2 = [[7.070912000000e+02, 0.00000e+00, 6.018873000e+02, 4.68878300e+01],
          [0.000000e+00, 7.070912000000e+02, 1.831104000e+02, 1.178601000e-01],
          [0.00000e+00, 0.000000000000e+00, 1.000000000000e+00, 6.203223000000e-03]]
total = np.dot(np.dot(Prect2, Rrect0), velo2cam)[[1,0,2]]
calib_extrinsics.append(np.dot(cam2straight, velo2cam))
calib_projections.append(total)
imgshapes.append((370,1226))
view_by_day.append(1226*.5/707.1)

# 10_03
velo2cam = [[7.9675140000e-03, -9.9996790000e-01, -8.4622640000e-04, -1.3777690000e-02],
            [-2.7710530000e-03, 8.2417100000e-04, -9.9999580000e-01, -5.5421170000e-02],
            [9.9996440000e-01, 7.9698250000e-03, -2.7643970000e-03, -2.9185890000e-01],
            [0,0,0,1]]
Rrect0 = [[9.9994540000e-01, 7.259129000000e-03, -7.519551000000e-03, 0],
          [-7.292213000000e-03, 9.999638000000e-01, -4.381729000000e-03, 0],
          [7.487471000000e-03, 4.436324000000e-03, 9.999621000000e-01, 0], [0,0,0,1]]
Prect2 = [[7.1885600000e+02, 0.0000000000e+00, 6.0719280000e+02, 4.538225000000e+01],
          [0.0000000000e+00, 7.1885600000e+02, 1.8521570000e+02, -1.1308870000e-01],
          [0.0000000000e+00, 0.0000000000e+00, 1.0000000000e+00, 3.7797610000e-03]]
total = np.dot(np.dot(Prect2, Rrect0), velo2cam)[[1,0,2]]
calib_extrinsics.append(np.dot(cam2straight, velo2cam))
calib_projections.append(total)
imgshapes.append((376,1241))
view_by_day.append(1241*.5/718.9)


calib_map_training = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,2,2,4]