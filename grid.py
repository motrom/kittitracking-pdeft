# -*- coding: utf-8 -*-
"""
defines the grid shape for each file
also defines which tiles are relevant for tracking (for this dataset and benchmark)
"""
import numpy as np

predictionview = np.array(((3., 50.), (-42., 42.)))
grndstep = np.array((3., 3.))
nlocaltiles = 2

def floor(arr):
    # casting rounds up for negative numbers, which is a problem for grid lookup
    return np.floor(arr).astype(int) if type(arr) == np.ndarray else int(arr//1)

# determine range of ground tiles
#grndstart = floor(predictionview[:,0]/grndstep) - nlocaltiles
#grndlen = floor(predictionview[:,1]/grndstep) + nlocaltiles+1 - grndstart
grndstart = predictionview[:,0] - nlocaltiles*grndstep
grndlen = np.ceil((predictionview[:,1]-predictionview[:,0])/grndstep).astype(int) + 2*nlocaltiles
if grndlen[0]%2 == 1: grndlen[0] += 1
if grndlen[1]%2 == 1: grndlen[1] += 1

# grid of cells that will be checked for objects
# go ahead and remove cells that are outside of kitti annotated range
grnd2checkgrid = np.mgrid[nlocaltiles:grndlen[0]-nlocaltiles,
                          nlocaltiles:grndlen[1]-nlocaltiles].reshape((2,-1)).T.copy()
checkgrid = grnd2checkgrid * grndstep
_checkclosery = np.maximum(checkgrid[:,1]+grndstart[1]+grndstep[1],
                                -checkgrid[:,1]-grndstart[1])
_checkgridinclude = (checkgrid[:,0]+grndstart[0]+grndstep[0] > _checkclosery)
_checkgridinclude &= checkgrid[:,0]+grndstart[0]+grndstep[0] > 3
grnd2checkgrid = grnd2checkgrid[_checkgridinclude]