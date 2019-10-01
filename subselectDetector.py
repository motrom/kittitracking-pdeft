#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Determines the 3m x 3m tiles that are most important to apply object detection on.
This can be used to speed up an object detector, at the cost of lowered accuracy
because of missed detections (but that's the reason to carefully choose tiles to miss).
At the moment, this effect is only simulated (detections outside the tiles are
not used by the tracker).

currently not a modular function, a.k.a. based on the specific object parameterization
from singleIntegrator.py
"""
import numpy as np
from grid import grnd2checkgrid, gridstart, gridstep, gridlen
from singleIntegrator import soPositionDistribution, ft_pexist
from gridtools import mapNormal2Subgrid


""" determines whether tile with tracked object will be checked
set so that well-tracked (steady-state) object is as important as unviewable tile
or tile adjacent to border
"""
_steadystatedistentropy = .9
_maxdistentropy = 3.
_existentropymultiplier = .1/.25 * _maxdistentropy/_steadystatedistentropy
#def objectEntropy(obj, existprob):
#    variances = obj[[7,15,23,31,39]]
#    distentropy = np.clip(np.sum(np.sqrt(variances)), 0, _maxdistentropy)
#    distentropy *= .05 / _steadystatedistentropy
#    return existprob*distentropy + (1 - existprob)*existprob*_existentropymultiplier
def objectEntropy(posdist, existprob):
    xvar,yvar,xycov = posdist[2:5]
    # root sum square of deviation in major directions
    distentropy = np.clip(np.sqrt(xvar+yvar), 0, _maxdistentropy)
    distentropy *= .05 / _steadystatedistentropy
    return existprob*distentropy + (1 - existprob)*existprob*_existentropymultiplier


def subselectDetector(objects, objecthypweights, occupancy, visibility, empty, ratio):
    ntiles = int(np.prod(occupancy.shape) * ratio)
    tilescores = occupancy.copy()
    for objidx in range(len(objects)):
        obj = objects[objidx]
        objectexistprob = obj[ft_pexist] * objecthypweights[objidx]
        if objectexistprob < 1e-3: continue
        #objuncertainty = objectEntropy(obj, objectexistprob)
        positiondist = soPositionDistribution(obj)
        objuncertainty = objectEntropy(positiondist, objectexistprob)
        subgridloc, occupysubgrid = mapNormal2Subgrid(positiondist,
                                        gridstart, gridstep, gridlen, subsize=2)
        subgridend = subgridloc + occupysubgrid.shape
        tilescores[subgridloc[0]:subgridend[0],
                   subgridloc[1]:subgridend[1]] += occupysubgrid * objuncertainty
    tilescores *= visibility # no point in checking undetectable tiles
    tilescores[empty] = 0 # always "detect" empty tiles
    emptytiles = grnd2checkgrid[np.where(empty[grnd2checkgrid[:,0],
                                               grnd2checkgrid[:,1]])[0]]
    tiles2detect = np.argsort(tilescores[grnd2checkgrid[:,0], grnd2checkgrid[:,1]])
    ntilespossible = sum(tilescores[grnd2checkgrid[:,0],grnd2checkgrid[:,1]]>0)
    tiles2detect = grnd2checkgrid[tiles2detect[-min(ntiles, ntilespossible):]]
    tiles2detect = np.append(tiles2detect, emptytiles, axis=0)
    # scatter detected tiles to binary grid
    tiles2detectgrid = np.zeros(gridlen, dtype=bool)
    tiles2detectgrid[tiles2detect[:,0], tiles2detect[:,1]] = 1
    return tiles2detectgrid