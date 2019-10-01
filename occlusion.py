# -*- coding: utf-8 -*-
"""
makes image-like map of point cloud, and uses this to approximate visibility of 
objects

this relies on the fact that kitti's point cloud is actually ordered by laser
arbitrary lidar point clouds will require different code
"""
import numpy as np
import numba as nb

from grid import grnd2checkgrid, gridstart, gridstep

anglestart = -1.
anglestop = 1.
angle_resolution = .01
nangles = int((anglestop-anglestart)/angle_resolution)
anglevector = np.arange(anglestart, anglestop, angle_resolution)
tangentvector = np.tan(anglevector)

## Kitti's Velodyne laser parameters
# sort lasers in decreasing-height order
# this way, merges can be checked to make sense (objects don't expand beyond their base)
laser_angles = np.arange(0., 64)
laser_angles[:32] = -.33333*laser_angles[:32] + 3.
laser_angles[32:] = -.5*laser_angles[32:] + laser_angles[31] + 31*.5
laser_angles = np.tan(laser_angles * np.pi/180.)
#laser_angles = laser_angles[lasers]
laser_angles += -.02 # correction for observed angles
laser_intercepts = np.zeros(64)
laser_intercepts[:32] = .209 - .00036*np.arange(32,dtype=float)
laser_intercepts[32:] = .126 - .00032*np.arange(32,dtype=float)
# laser angle space * current laser gap * some multiplier for missing lasers
height_angle_slack = .01*4*2
lasers = list(range(55))
laser_angles = laser_angles[lasers]
laser_intercepts = laser_intercepts[lasers]

occlusionimgshape = (len(lasers), nangles)
min_visibility_val = .05
max_visibility_val = 1.
max_visibility_npixels = 40
visibility_npixels2val = (max_visibility_val-min_visibility_val)/max_visibility_npixels
partial_visibility_val = .3


@nb.njit(nb.void(nb.f8[:],nb.i8[:],nb.f8[:]))
def imgUpdate(row, idxs, vals):
    for xidx in range(len(idxs)):
        rowidx = idxs[xidx]
        row[rowidx] = max(row[rowidx], vals[xidx])

def pointCloud2OcclusionImg(pts, occlusionimg=None):
    if occlusionimg is None:
        occlusionimg = np.zeros(occlusionimgshape)
    occlusionimg[:] = 0.
    starts = np.where(np.diff(np.sign(pts[:,1])) > 0)[0]
    starts = np.concatenate(([0], starts+1, [len(pts)]))
    true_starts = np.append(np.diff(starts) > 2, [True])
    starts = starts[true_starts]
    assert starts.shape[0] > 55, "too few lasers in this point cloud!"
    
    angles = np.arctan2(pts[:,1], pts[:,0])
    pixels = np.floor((angles - anglestart) / angle_resolution).astype(int)
    included = (pixels >= 0) & (pixels < nangles)
    dists = np.hypot(pts[:,1], pts[:,0])
    
    for laser in lasers:
        datastart = starts[laser]
        datastop = starts[laser+1]
        if datastop - datastart < 3:
            continue
        pixelsl = pixels[datastart:datastop][included[datastart:datastop]]
        distsl = dists[datastart:datastop][included[datastart:datastop]]
        imgUpdate(occlusionimg[laser], pixelsl, distsl)
    return occlusionimg

### currently slowest part of tracker!
occlusion_buffer_distance = 3.
def occlusionImg2Grid(occlusionimg, grid, grnd):
    laserheights = np.zeros(len(lasers))
    for tilex, tiley in grnd2checkgrid:
        tilenearx = tilex*gridstep[0] + gridstart[0]
        tilelefty = tiley*gridstep[1] + gridstart[1]
        if tiley*gridstep[1] < gridstart[1]:
            tileneary = tilelefty + gridstep[1]
            startangle = np.arctan2(tilelefty, tilenearx)
            endangle = np.arctan2(tilelefty+gridstep[1], tilenearx+gridstep[0])
        else:
            tileneary = tilelefty
            startangle = np.arctan2(tilelefty, tilenearx+gridstep[1])
            endangle = np.arctan2(tilelefty+gridstep[1], tilenearx+gridstep[0])
        startapixel = int((startangle-anglestart)/angle_resolution)
        endapixel = int((endangle-anglestart)/angle_resolution)+1
        startapixel = max(min(startapixel, nangles-1), 0)
        endapixel = max(min(endapixel, nangles), 1)
        distance = np.hypot(tilenearx, tileneary)
        tilegrnd = grnd[tilex,tiley]
        height = tilegrnd[3]-tilenearx*tilegrnd[0]-tilelefty*tilegrnd[1]
        laserheights[:] = height - laser_intercepts - laser_angles*distance
        starthpixel, endhpixel = np.searchsorted(laserheights, (-2., -.2))
        subimg = occlusionimg[starthpixel:endhpixel, startapixel:endapixel]
        if np.all(subimg==0):
            # unclear whether there is occlusion or just nothing
            visibility = partial_visibility_val
        else:
            visiblepixels = np.sum(subimg > distance-occlusion_buffer_distance)
            visiblepixels += np.sum(subimg == 0)
            visibility = visibility_npixels2val*visiblepixels + min_visibility_val
            visibility = min(max_visibility_val, visibility)
        grid[tilex, tiley] = visibility
        
"""
looking at core region of a detection, checking for lidar points that went through it
would imply that core region is empty... is which case the detection is likely false
current core region for car is (-.9,-.5,.4):(.9,.5,1.2)
"""
def boxTransparent(box, occlusionimg, grnd):
    x = box[0]
    y = box[1]
    l = .9
    w = .5
    tilex = int((x-gridstart[0])/gridstep[0])
    tiley = int((y-gridstart[1])/gridstep[1])
    grndhere = grnd[tilex, tiley]
    adjustedheight = grndhere[3]-grndhere[0]*x-grndhere[1]*y
    cos = np.cos(box[2])
    sin = np.sin(box[2])
    corners = [(x+l*cos-w*sin, y+l*sin+w*cos),
               (x+l*cos+w*sin, y+l*sin-w*cos),
               (x-l*cos+w*sin, y-l*sin-w*cos),
               (x-l*cos-w*sin, y-l*sin+w*cos)]
    corners = np.array(corners)
    cornerangles = np.arctan2(corners[:,1], corners[:,0])
    startangle = np.min(cornerangles)
    endangle = np.max(cornerangles)
    startapixel = int((startangle-anglestart)/angle_resolution)
    endapixel = int((endangle-anglestart)/angle_resolution)
    startapixel = max(min(startapixel, nangles-1), 0)
    endapixel = max(min(endapixel, nangles), 1)
    distance = np.hypot(x,y)
    laserheights = adjustedheight - laser_intercepts - laser_angles*distance
    starthpixel, endhpixel = np.searchsorted(laserheights, (-1.2, -.4))
    starthpixel += 1
    subimg = occlusionimg[starthpixel:endhpixel, startapixel:endapixel]
    subimgsize = (endhpixel-starthpixel)*(endapixel-startapixel)
    return np.sum(subimg > distance + 3) < max(subimgsize*.2, 5)