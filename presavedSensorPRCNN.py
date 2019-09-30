# -*- coding: utf-8 -*-
"""
takes detection files as output by the PointRCNN default code
https://github.com/sshaoshuai/PointRCNN

getMsmts() returns numpy arrays of the BEV information for a single timestep

__main__ code let's you check the distribution of detector score, to map it as
well as possible to probability of genuity
"""
from os.path import isfile
import numpy as np

#dataformat = '/home/m2/Data/kitti/estimates/detectionsPRCNNtext/{:02d}f{:04d}.txt'
dataformat = '/home/motrom/Downloads/kitti_devkit/detectionsPRCNN/{:02d}f{:04d}.txt'

# better results on nofake setting if this is set a bit higher (.15)
min_sensor_prob_to_report = .03

def scoreToProb(score):
    return np.minimum(1./(1+np.exp(-1.*score+3.)), .99)

def getMsmts(sceneidx, fileidx):
    filename = dataformat.format(sceneidx,fileidx)
    if isfile(filename):
        with open(filename, 'r') as fd: fdstr = fd.read()
    else:
        fdstr = ''
    fdlist = fdstr.split('\n')
    if fdlist[-1] == '': fdlist.pop()
    data = np.zeros((len(fdlist), 6))
    for idx, fdline in enumerate(fdlist):
        fdline = fdline.split(' ')
        gtang = 4.7124 - float(fdline[14])
        gtang = gtang - 6.2832 if gtang > 3.1416 else gtang
        score = float(fdline[15])
        score = scoreToProb(score)
        data[idx] = (float(fdline[13]), -float(fdline[11]), gtang,
                     float(fdline[10])/2, float(fdline[9])/2, score)
    include = data[:,5] > min_sensor_prob_to_report
    include &= (data[:,0] < 57) & (abs(data[:,1]) < 48)
    data = data[include]
    return data



if __name__ == '__main__':
    # analyze score distribution for true and false detections
    from sklearn.neighbors import KernelDensity
    from scipy.optimize import linear_sum_assignment
    import matplotlib.pyplot as plt
    
    from evaluate import soMetricIoU
    from kittiGT import readGroundTruthFileTracking
    
    gt_files = '/home/m2/Data/kitti/tracking_gt/{:04d}.txt'
    scene_idxs = list(range(10))
    
    scoresmatch = []
    scorescrop = []
    scoresmiss = []
    nmissed = 0
    nmissedcrop = 0
    
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
    scoreT = kd.fit(np.array(scoresmatch)[:,None]).score_samples(scorearray[:,None])
    scoreT = np.exp(scoreT) * relmatches
    scoreF = kd.fit(np.array(scoresmiss)[:,None]).score_samples(
                        scorearray[:,None])
    scoreF = np.exp(scoreF) * (1-relmatches)
    ratio = scoreT / np.maximum(scoreT + scoreF, 1e-8)
    plt.plot(scorearray, ratio, 'b')