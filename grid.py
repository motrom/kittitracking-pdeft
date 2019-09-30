# -*- coding: utf-8 -*-
"""
defines the grid shape for each file
also defines which tiles are relevant for tracking (for this dataset and benchmark)
"""
import numpy as np

predictionview = np.array(((3., 50.), (-42., 42.)))
gridstep = np.array((3., 3.))
nlocaltiles = 2

def floor(arr):
    # casting rounds up for negative numbers, which is a problem for grid lookup
    return np.floor(arr).astype(int) if type(arr) == np.ndarray else int(arr//1)

# determine range of ground tiles
gridstart = predictionview[:,0] - nlocaltiles*gridstep
gridlen = np.ceil((predictionview[:,1]-predictionview[:,0])/gridstep).astype(int) + 2*nlocaltiles
if gridlen[0]%2 == 1: gridlen[0] += 1
if gridlen[1]%2 == 1: gridlen[1] += 1

# grid of cells that will be checked for objects
# go ahead and remove cells that are outside of kitti annotated range
grnd2checkgrid = np.mgrid[nlocaltiles:gridlen[0]-nlocaltiles,
                          nlocaltiles:gridlen[1]-nlocaltiles].reshape((2,-1)).T.copy()
checkgrid = grnd2checkgrid * gridstep
_checkclosery = np.maximum(checkgrid[:,1]+gridstart[1]+gridstep[1],
                                -checkgrid[:,1]-gridstart[1])
_checkgridinclude = (checkgrid[:,0]+gridstart[0]+gridstep[0] > _checkclosery)
_checkgridinclude &= checkgrid[:,0]+gridstart[0]+gridstep[0] > 3
grnd2checkgrid = grnd2checkgrid[_checkgridinclude]