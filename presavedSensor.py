#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
last mod 7/2/18

usage -- function that returns nX6 numpy array every timestep
n detections with 6 fields:
    x (forward) position in meters
    y (right to left) position in meters
    angle (starting forwards, counterclockwise) in radians
    length (from center to end) in meters
    width " "
    probability that this detection is genuine (a.k.a. score, normalized to [0,1])
"""
import numpy as np
from config import grndstart, grndstep, grndlen

dataformat = './detectionsVoxelJones/{:02d}f{:04d}.npy'


def initializeSensor(scene_idx, startfile):
    return np.array((scene_idx, startfile))

def scoreToProb(score): # BT630 nu
    out = np.where(score < -3, score*.0025 + .07,
                   .33 + .11*score - .01*score*score)
    return np.maximum(np.minimum(out, .99), .01)

def getMsmts(sensorinfo):
    fileidx = sensorinfo[1]
    data = np.load(dataformat.format(sensorinfo[0], fileidx))
    data[data[:,2]>np.pi, 2] -= 2*np.pi
    data[:,5] = scoreToProb(data[:,5])
    data = data[data[:,5] > .05]
    sensorinfo[1] += 1
    return data

def getMsmtsInTile(msmts, tilex, tiley):
    xlo = (tilex + grndstart[0])*grndstep[0]
    xhi = xlo + grndstep[0]
    ylo = (tiley + grndstart[1])*grndstep[1]
    yhi = ylo + grndstep[1]
    intile = ((msmts[:,0] >= xlo) & (msmts[:,0] < xhi) &
              (msmts[:,1] >= ylo) & (msmts[:,1] < yhi))
    assert sum(intile) <= 2 # for this simulation
    return msmts[intile].copy()


"""
check detector performance w/o tracking
"""
if __name__ == '__main__':
    # analyze score distribution for true and false detections
    from scipy.optimize import linear_sum_assignment
    
    from evaluate import MetricAvgPrec
    from kittiGT import readGroundTruthFileTracking
    from config import sceneranges
    
    gt_files = '/home/m2/Data/kitti/tracking_gt/{:04d}.txt'
    scene_idxs = list(range(10))
    
    scoresmatch = []
    scorescrop = []
    scoresmiss = []
    nmissed = 0
    nmissedcrop = 0
    metric = MetricAvgPrec()
    
    for scene_idx in scene_idxs:
        startfileidx, endfileidx = sceneranges[scene_idx]
        
        sensorinfo = initializeSensor(scene_idx, startfileidx)
        with open(gt_files.format(scene_idx), 'r') as fd: gtfilestr = fd.read()
        gt_all, gtdontcares = readGroundTruthFileTracking(gtfilestr, ('Car', 'Van'))
        
        for fileidx in range(startfileidx, endfileidx):
            gt = gt_all[fileidx]
            gtscored = np.array([gtobj['scored'] for gtobj in gt])
            gtboxes = np.array([gtobj['box'] for gtobj in gt])
            gtdiff = np.array([gtobj['difficulty'] for gtobj in gt])
            msmts = getMsmts(sensorinfo)
            
            ngt = gtscored.shape[0]
            nmsmts = msmts.shape[0]
            matches = np.zeros((ngt, nmsmts))
            for gtidx, msmtidx in np.ndindex(ngt, nmsmts):
                gtbox = gtboxes[gtidx]
                msmt = msmts[msmtidx]
                closeness = np.hypot(*(gtbox[:2]-msmt[:2])) * .4
                closeness += ((gtbox[2]-msmt[2]+np.pi)%(2*np.pi)-np.pi) * 1.
                closeness += np.hypot(*(gtbox[3:]-msmt[3:5])) * .3
                matches[gtidx, msmtidx] = min(closeness - 1, 0)
            matchesnonmiss = matches < 0
            rowpairs, colpairs = linear_sum_assignment(matches)
            msmtsmissed = np.ones(nmsmts, dtype=bool)
            for rowidx, colidx in zip(rowpairs, colpairs):
                nonmiss = matchesnonmiss[rowidx, colidx]
                noncrop = gtscored[rowidx]
                if nonmiss:
                    msmtsmissed[colidx] = False
                    if noncrop:
                        scoresmatch.append(msmts[colidx,5])
                    else:
                        scorescrop.append(msmts[colidx,5])
                else:
                    nmissed += 1
                    if noncrop:
                        nmissedcrop += 1
            for msmtidx in range(nmsmts):
                if msmtsmissed[msmtidx]:
                    scoresmiss.append(msmts[msmtidx,5])
            
            metric.add(gtboxes, gtscored, gtdiff, msmts[:,:5], msmts[:,5])
    
    avgprec = metric.calc()