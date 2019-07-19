# -*- coding: utf-8 -*-
""" taken from
https://github.com/utiasSTARS/pykitti/blob/master/pykitti/odometry.py """

"""Provides helper methods for loading and parsing KITTI data."""

from collections import namedtuple

import numpy as np
#from PIL import Image

__author__ = "Lee Clement"
__email__ = "lee.clement@robotics.utias.utoronto.ca"

# Per dataformat.txt
OxtsPacket = namedtuple('OxtsPacket',
                        'lat, lon, alt, ' +
                        'roll, pitch, yaw, ' +
                        'vn, ve, vf, vl, vu, ' +
                        'ax, ay, az, af, al, au, ' +
                        'wx, wy, wz, wf, wl, wu, ' +
                        'pos_accuracy, vel_accuracy, ' +
                        'navstat, numsats, ' +
                        'posmode, velmode, orimode')

# Bundle into an easy-to-access structure
OxtsData = namedtuple('OxtsData', 'packet, T_w_imu')


def subselect_files(files, indices):
    try:
        files = [files[i] for i in indices]
    except:
        pass
    return files


def rotx(t):
    """Rotation about the x-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1,  0,  0],
                     [0,  c, -s],
                     [0,  s,  c]])


def roty(t):
    """Rotation about the y-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                     [0,  1,  0],
                     [-s, 0,  c]])


def rotz(t):
    """Rotation about the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])


def transform_from_rot_trans(R, t):
    """Transforation matrix from rotation matrix and translation vector."""
    R = R.reshape(3, 3)
    t = t.reshape(3, 1)
    return np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))


def read_calib_file(filepath):
    """Read in a calibration file and parse into a dictionary."""
    data = {}

    with open(filepath, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data


def pose_from_oxts_packet(packet, scale):
    """Helper method to compute a SE(3) pose matrix from an OXTS packet.
    """
    er = 6378137.  # earth radius (approx.) in meters

    # Use a Mercator projection to get the translation vector
    tx = scale * packet.lon * np.pi * er / 180.
    ty = scale * er * \
        np.log(np.tan((90. + packet.lat) * np.pi / 360.))
    tz = packet.alt
    t = np.array([tx, ty, tz])

    # Use the Euler angles to get the rotation matrix
    Rx = rotx(packet.roll)
    Ry = roty(packet.pitch)
    Rz = rotz(packet.yaw)
    R = Rz.dot(Ry.dot(Rx))

    # Combine the translation and rotation into a homogeneous transform
    return R, t


def load_oxts_packets_and_poses(oxts_files):
    """Generator to read OXTS ground truth data.

       Poses are given in an East-North-Up coordinate system 
       whose origin is the first GPS position.
    """
    # Scale for Mercator projection (from first lat value)
    scale = None
    # Origin of the global coordinate system (first GPS position)
    origin = None

    oxts = []

    for filename in oxts_files:
        with open(filename, 'r') as f:
            for line in f.readlines():
                line = line.split()
                # Last five entries are flags and counts
                line[:-5] = [float(x) for x in line[:-5]]
                line[-5:] = [int(float(x)) for x in line[-5:]]

                packet = OxtsPacket(*line)

                if scale is None:
                    scale = np.cos(packet.lat * np.pi / 180.)
                    print("GPS/IMU scale {:f}".format(scale))

                R, t = pose_from_oxts_packet(packet, scale)

                if origin is None:
                    origin = t
                    print("GPS/IMU origin {:f}".format(origin))

                T_w_imu = transform_from_rot_trans(R, t - origin)

                oxts.append(OxtsData(packet, T_w_imu))

    return oxts


def load_oxt(filename):
    scale = None
    # Origin of the global coordinate system (first GPS position)
    origin = None

    oxts = []

    with open(filename, 'r') as f:
        for line in f.readlines():
            line = line.split()
            # Last five entries are flags and counts
            line[:-5] = [float(x) for x in line[:-5]]
            line[-5:] = [int(float(x)) for x in line[-5:]]

            packet = OxtsPacket(*line)

            if scale is None:
                scale = np.cos(packet.lat * np.pi / 180.)
                print("GPS/IMU scale {:f}".format(scale))

            R, t = pose_from_oxts_packet(packet, scale)

            if origin is None:
                origin = t
                print("GPS/IMU origin {:f} {:f} {:f}".format(*origin))

            T_w_imu = transform_from_rot_trans(R, t - origin)

            oxts.append(T_w_imu)
    return oxts


def loadSelfTransformations(oxt_file):
    poses = load_oxt(oxt_file)
    transforms = [np.eye(4)]
    for file_idx in range(1,len(poses)):
        transform = np.linalg.inv(poses[file_idx]).dot(poses[file_idx-1])
        transforms.append(transform)
        assert transform[2,2] > .995
    return transforms


if __name__ == '__main__':
    from imageio import imread
    from cv2 import imshow, waitKey, destroyWindow
    def clear(): destroyWindow('a')
    
    oxt_file = '../tracking/training/oxts/{:04d}.txt'
    img_files = '../tracking/training/image_02/{:04d}/{:06d}.png'
    lidar_files = '../tracking/training/velodyne/{:04d}/{:06d}.bin'
    scene_idx = 1
    files = range(400)#range(154)
    
    poses = load_oxt(oxt_file.format(scene_idx))
    draw_on_img = np.zeros((80,160,3), dtype=np.uint8) + 255
    
    speeds = []
    for file_idx in files[1:]:
        img = imread(img_files.format(scene_idx, file_idx))[:,:,::-1]
        pose = np.linalg.inv(poses[file_idx-1]).dot(poses[file_idx])
        assert abs(pose[2,3]) < .2 # no up-down motion
        assert abs(pose[1,3]) < .5 # no significant lateral motion
        assert pose[0,3] > -.2 # not moving backwards
        assert pose[0,3] < 4. # not very high speed
        assert pose[2,2] > .99 # nearly same vertical orientation
        assert pose[0,0] > .96 # mostly in same direction
        
        # check whether angle approximations work
        angleapprox1 = np.arctan2(pose[0,1], pose[0,0])
        angleapprox2 = np.arctan2(-pose[1,0], pose[1,1])
        assert abs(angleapprox1-angleapprox2) < .03
        
        speed_instant = np.hypot(pose[0,3], pose[1,3])
        speed = speed_instant * 10. # convert to m/s
        speeds.append(speed)
        
#        direction_correct = np.hypot(pose[0,0], pose[0,1])
#        direction_cos = pose[0,0] / direction_correct
#        direction_sin = pose[0,1] / direction_correct
        wheel_angle = np.arctan2(pose[0,1], pose[0,0]) * 2. / max(.1, speed_instant)
        wheel_cos = np.cos(wheel_angle)
        wheel_sin = np.sin(wheel_angle)
        
        # draw arrow indicate distance, with color indicating speed
        speed_color = np.array((255-speed*8, speed*10, speed*2))
        speed_color = np.minimum(np.maximum(speed_color, 0), 255).astype(np.uint8)

        angle_shake = 2./80
        draw_on_img2 = draw_on_img.copy()
        for x in range(int(80*wheel_cos)):
            ylo = 80 + int((80-x)*(wheel_sin-angle_shake))
            yhi = 80 + int((80-x)*(wheel_sin+angle_shake)) + 1
            draw_on_img2[x,ylo:yhi] = speed_color
        
        img[:80,:160] = draw_on_img2
        imshow('a', img)
        if waitKey(100) == ord('q'):
            break
    clear()