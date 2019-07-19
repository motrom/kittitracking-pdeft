#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
last mod 5/21/19
"""
import numpy as np

predictionview = np.array((4., 50., -40., 40.))

grndstep = np.array((3., 3.))
max_road_slope = .1 # tangent

def floor(arr):
    # casting rounds up for negative numbers, which is a problem for grid lookup
    return np.floor(arr).astype(int) if type(arr) == np.ndarray else int(arr//1)

nlocaltiles = 2
# determine range of ground tiles
grndstart = floor(predictionview[[0,2]]/grndstep) - nlocaltiles
grndlen = floor(predictionview[[1,3]]/grndstep) + nlocaltiles+1 - grndstart
if grndlen[0]%2 == 1: grndlen[0] += 1
if grndlen[1]%2 == 1: grndlen[1] += 1

# grid of cells that will be checked for objects
# go ahead and remove cells that are outside of kitti annotated range
grnd2checkgrid = np.mgrid[nlocaltiles:grndlen[0]-nlocaltiles,
                          nlocaltiles:grndlen[1]-nlocaltiles].reshape((2,-1)).T.copy()
_grnd2checkclosery = np.maximum(grnd2checkgrid[:,1]+grndstart[1]+1,
                                -grnd2checkgrid[:,1]-grndstart[1])
_grnd2checkgridinclude = ((grnd2checkgrid[:,0]+grndstart[0]+1)*grndstep[0] >
                          _grnd2checkclosery*grndstep[1])
_grnd2checkgridinclude &= (grnd2checkgrid[:,0]+grndstart[0]+1)*grndstep[0] > 3
grnd2checkgrid = grnd2checkgrid[_grnd2checkgridinclude]


# which scenes corresponding to which day of kitti data collection
calib_map_training = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,2,2,3]

## there are some cases of incomplete or completely missing lidar frames
## usually these are at the end of the scene, with the exception of scene 1
## these ranges don't have any of the offending frames
sceneranges = [(0,153), (181,447), (0,233), (0,144), (0,314), (0, 296), (0,270),
               (0,799), (0,390), (0,802)]