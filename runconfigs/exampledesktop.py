# -*- coding: utf-8 -*-
"""
Files like this is read by track.py.
This one uses saved PointRCNN detections for the first 10 scenes of the
kitti tracking training dataset.
"""

lidar_files = ('/home/motrom/Downloads/kitti_devkit/tracking/training/velodyne/'+
               '{:04d}/{:06d}.bin')

gt_files = '/home/motrom/Downloads/kitti_devkit/tracking/training/label_02/{:04d}.txt'

oxt_files = '/home/motrom/Downloads/kitti_devkit/tracking/training/oxts/{:04d}.txt'

ground_files = ('/home/motrom/Downloads/kitti_devkit/tracking/trackingground/'+
                'training/{:02d}f{:06d}.npy')

save_estimates = True
estimate_files = '/home/motrom/Downloads/kitti_devkit/testestimates/{:02d}f{:04d}.npy'

display_video = False
save_video = False
video_file = 'results.mp4'
img_files =  ('/home/motrom/Downloads/kitti_devkit/tracking/training/image_02'+
              '/{:04d}/{:06d}.png')

"""
(scene number, starting file number, ending file number, calibration idx (aka date))
notice that not all files are used!
scene 1 -- there is a big gap where no lidar data was given
some other scenes are missing the last file, which also has incomplete lidar data
"""
scenes = [(0,0,153,0), (1,181,447,0), (2,0,233,0), (3,0,144,0), (4,0,314,0),
          (5,0,296,0), (6,0,270,0), (7,0,799,0), (8,0,390,0), (9,0,802,0)]
#scenes = [(3,0,20,0)]