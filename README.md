# Description
arxiv paper released soon...  
Vehicle tracking on the Kitti dataset. Any detector can be used, though we have tested primarily on Point-RCNN detections from https://github.com/xinshuoweng/AB3DMOT and on our detector https://github.com/motrom/voxeljones.  
Each vehicle is tracked as a 2D rectangle (bird's eye view) with a nearly-constant velocity and heading model. A hypothesis-oriented multi-bernoulli mixture tracker is used in combination with an occupancy grid for yet-undetected objects. The primary innovation of our method is careful handling of both missed detections and false detections, which for imperfect detectors at high frame rates are likely to occur frequently for certain vehicles or inanimate objects. Most model-based trackers assume that detection errors are independent over time, which obviously causes problems in such cases.

# Usage
## Dependencies
1. Python (python 3 has been used primarily, but 2.7 should be fine)
2. numpy & scipy, imageio & matplotlib for visualizations
3. OpenCV version 4
4. numba (only used to speed up some functions, can be disabled easily if you don't want to install numba)
5. github.com/motrom/fastmurty for data association -- modify the makefile to compile the dense (not sparse) version, then run 'make', then move mhtda.so to this folder
6. motmetrics for evaluation

## Input
Detections from VoxelJones are available at https://utexas.box.com/shared/static/ptt0hrs7aekno22i7valqypqv3osq862.zip. Detection formatting is handled by presavedDetector.py, the easiest way to handle new input would be to replace this code with your own.  
The tracker also uses Kitti's lidar and positioning data. For visualization and performance evaluation, it needs the left images and the kitti annotation text files. All data files are accessed using `somestring.format(scene_number)` or `somestring.format(scene_number, frame_number)`, so make appropriate modifications to each `somestring` for your file organization.  
Finally, the code currently handles the first ten scenes from the kitti training set. I will expand it to all scenes from the training and testing set soon... only minor details like the length of each scene are missing.

## Running
First, ground planes for a scene and timestep must be created via ground.py. Track.py can then be run on any scene, and saves and visualizes results. Evaluate.py and evaluate_kittiofficial.py can be run on saved results.

# Performance
A short video of examples is available at https://utexas.box.com/shared/static/4y4tv8j28x3axh137s4rp3mgaeortaa3.mp4

Kitti doesn't have a standard 3D tracking metric, here are some MOTChallenge metrics with the Kitti detection benchmark's cutoff of .7 BEV IoU.

| actual # of cars| MOTA | MOTP | ID Switches | Mostly Tracked | Mostly Lost |
| --- | --- | --- | --- | --- | --- |
| 281 | .64 | .94 | 135 | 118 | 36 |


# Acknowledgements
using code from https://github.com/utiasSTARS/pykitti/blob/master/pykitti/odometry.py to obtain the host vehicle's motion.
