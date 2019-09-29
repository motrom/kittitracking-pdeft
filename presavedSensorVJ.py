#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
last mod 7/2/18

usage for new detector:
    first disable metrics check
    min_sensor_prob to <<0
    use simple scoreToProb
use the plots to figure out a good scoreToProb function
then you can run metrics check

current avg precisions:
    .5 iou -- .855, .783, .774 --> .863, .834, .837
    .7 iou -- .538, .597, .619
    
n ground truths: 2608 easy, 6890 moderate, 8307 hard?

monogrnet at .3 IoU:: .815, .72, .653
voxeljones at .3: .91, .79, .76

"""
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.linear_model import LinearRegression
from config import grndstart, grndstep, grndlen

min_sensor_prob_to_report = .03

dataformat = '/home/m2/Data/kitti/estimates/detectionsBT630/{:02d}f{:04d}.npy'
#dataformat = '/home/m2/Data/kitti/estimates/detectionsMGR/{:02d}f{:04d}.npy'

#def scoreToProb(score): return score+30 ### default before you've checked performance

#def scoreToProb(score): # BT323
#    score = 1/(1+np.exp(-.3*score+1))
#    if score < .25: return 0
#    return max(0, min(1, 0.10 - 1.05*score + 2.04*score*score))*.8 + .1

#def scoreToProb(score): # BT528
#    return np.minimum(.2 + score*.09 - score*score*.003, 1)

#def scoreToProb(score): # BT620
#    return np.maximum(np.minimum(.2 + .11*score - .0025*score*score, .95), .05)

#def scoreToProb(score): # BT630 pre 8/14/19
#    out = np.where(score < -3, score*.0025 + .07,
#                   .33 + .11*score - .01*score*score)
#    return np.maximum(np.minimum(out, .99), .01)

def scoreToProb(score): # BT630 post 8/14/19
    out = np.where(score < -3, score*.0025 + .07, 1/(1+np.exp(1.-score*.82)))
    return np.maximum(np.minimum(out, .99), .01)

def getMsmts(sceneidx, fileidx):
    data = np.load(dataformat.format(sceneidx, fileidx))
    if data.shape == (0,):
        data = np.zeros((0,6))
    data[data[:,2]>np.pi, 2] -= 2*np.pi
    data[:,5] = scoreToProb(data[:,5])
    data = data[data[:,5] > min_sensor_prob_to_report]
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


if __name__ == '__main__':
    # analyze score distribution for true and false detections
    from scipy.optimize import linear_sum_assignment
    import matplotlib.pyplot as plt
    
    from evaluate import MetricAvgPrec, soMetricIoU
    from analyzeGT import readGroundTruthFileTracking
    from trackinginfo import sceneranges
    
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
        
        with open(gt_files.format(scene_idx), 'r') as fd: gtfilestr = fd.read()
        gt_all, gtdontcares = readGroundTruthFileTracking(gtfilestr, ('Car', 'Van'))
        selfposT = None # isn't actually used
        
        for fileidx in range(startfileidx, endfileidx):
            gt = gt_all[fileidx]
            gtscored = np.array([gtobj['scored'] for gtobj in gt])
            gtboxes = np.array([gtobj['box'] for gtobj in gt])
            gtdiff = np.array([gtobj['difficulty'] for gtobj in gt])
            msmts = getMsmts(scene_idx, fileidx)
            
            ngt = gtscored.shape[0]
            nmsmts = msmts.shape[0]
            matches = np.zeros((ngt, nmsmts))
            for gtidx, msmtidx in np.ndindex(ngt, nmsmts):
                gtbox = gtboxes[gtidx]
                msmt = msmts[msmtidx]
                #closeness = np.hypot(*(gtbox[:2]-msmt[:2])) * .4
                #closeness += ((gtbox[2]-msmt[2]+np.pi)%(2*np.pi)-np.pi) * 1.
                #closeness += np.hypot(*(gtbox[3:]-msmt[3:5])) * .3
                #closeness -= 1
                #closeness = np.hypot(*(gtbox[:2]-msmt[:2])) - 1.5
                closeness = soMetricIoU(gtbox, msmt, cutoff=.1)
                matches[gtidx, msmtidx] = min(closeness, 0)
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
            
    scoresmatch.sort()
    scorescrop.sort()
    scoresmiss.sort()
    nmatches = len(scoresmatch)
    nmisses = len(scoresmiss)
    relmatches = float(nmatches) / (nmatches + nmisses)
    allscores = scoresmatch + scorescrop + scoresmiss
    minscore = np.percentile(allscores, .5)
    maxscore = np.percentile(allscores, 99.5)
    scorearray = np.linspace(minscore, maxscore, 100)
    kd = KernelDensity(bandwidth = (maxscore-minscore)/50, kernel='gaussian')
    scoreT = kd.fit(np.array(scoresmatch)[:,None]).score_samples(
                        scorearray[:,None])
    scoreT = np.exp(scoreT) * relmatches
    scoreF = kd.fit(np.array(scoresmiss)[:,None]).score_samples(
                        scorearray[:,None])
    scoreF = np.exp(scoreF) * (1-relmatches)
    ratio = scoreT / np.maximum(scoreT + scoreF, 1e-8)
    # fit a quadratic model to the ratio of true to false
    X = np.column_stack((scorearray, scorearray**2))
    lm = LinearRegression(fit_intercept=True, normalize=True).fit(X, ratio)
    coefs = (lm.intercept_, lm.coef_[0], lm.coef_[1])
    print(coefs)
    ests = coefs[0] + coefs[1]*scorearray + coefs[2]*scorearray**2
    plt.plot(scorearray, ratio, 'b', scorearray, ests, 'g--')
    
    
    avgprec = metric.calc()