# Overview/Status
This code is for the paper "Vehicular Multi-Object Tracking with Persistent Detector Failures", currently in review for IEEE Transactions on Intelligent Vehicles. A preprint is at https://arxiv.org/abs/1907.11306.  
This branch (v2) has been tested in simple cases (default settings and data, PointRCNN detections). I will be testing it more thoroughly in the coming week, to make sure it matches the paper's information. It does not run on the final example in the paper - the one with sensor fusion, which makes everything messier. Working code for said final example is in a different repository github.com/motrom/kittifusion-pdeft - it is similar to this code in function but harder to work with. An older version of the code is in the v1 branch, it works but is not exactly what was submitted (similar) and is not very clean.

# Description
Vehicle tracking on the Kitti dataset! Any detector can be used, though we have tested primarily on Point-RCNN and on our detector https://github.com/motrom/voxeljones.  We made sure to train said detectors w/o any images from the tracking scenes that were tested; see the Input section to acquire the resulting detections.  
Each vehicle is tracked as a 2D rectangle (bird's eye view) with a simple bicycle motion model. A single-hypothesis tracker is used in combination with an occupancy grid for yet-undetected objects. The primary innovation of our method is careful handling of both missed detections and false detections, which for imperfect detectors at high frame rates are likely to occur frequently for certain vehicles or inanimate objects. Most model-based trackers assume that detection errors are independent over time, which obviously causes problems in such cases.

# Usage
## Dependencies
1. Python (python 3.6 has been used primarily, but 2.7 should be fine)
2. numpy & scipy, imageio & matplotlib for visualizations
3. OpenCV version 4
4. numba (only used to speed up some functions, can be disabled easily if you don't want to install numba)

## Input
Detections from PointRCNN and VoxelJones are available at https://utexas.box.com/s/kabrfde65me39wovlwextyryr4fko3tp and https://utexas.box.com/shared/static/ptt0hrs7aekno22i7valqypqv3osq862.zip. Detection formatting is handled by presavedSensorXXX.py, the easiest way to handle new input would be make a new sensor file like these.  
The tracker also uses Kitti's lidar and positioning data. For visualization and performance evaluation, it needs the left images and the kitti annotation text files. All data files are specified by a file in runconfigs/ (currently there is an example.py) using `somestring.format(scene_number)` or `somestring.format(scene_number, frame_number)`, so make appropriate modifications to each `somestring` for your file organization.  
Finally, the code currently handles the first ten scenes from the kitti training set. I will expand it to all scenes from the training and testing set soon... only minor details like the length of each scene are missing.

## Running
First, set up a file in runconfigs/ that specifies the data to test on. Ground planes for a scene and timestep must be created via ground.py. track.py can then be run on any scene, and saves and/or visualizes results. Saved results can be evaluated with evaluate.py and evaluate_kittiofficial.py.

# Performance
A short video of examples is available at https://utexas.box.com/shared/static/4y4tv8j28x3axh137s4rp3mgaeortaa3.mp4

Otherwise, performance is best seen in the paper. MOTA for 0.3 BEV IoU, on the 10 tracking scenes used, is 86. MOTA for 0.5 2D IoU is 72 on the same data... meaning the bird's eye box estimates are not translating that precisely to image space. Hence this method hasn't been submitted to the normal Kitti tracking benchmark.

# Acknowledgements
Using code from https://github.com/utiasSTARS/pykitti/blob/master/pykitti/odometry.py to obtain the host vehicle's motion.  
Included kitti's 2D benchmark code along with the requisite assignment algorithm code.
