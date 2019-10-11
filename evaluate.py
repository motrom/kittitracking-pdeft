# -*- coding: utf-8 -*-
"""
tools for evaluation
__main__ code plots precision-recall and reports MOTA for selected results files

why not just use py-motmetrics?
because that code doesn't check different score thresholds -- you have to pick
one in advance
this code compares methods for various false-negative/false-positive tradeoffs
and uses the optimal score cutoff (for each method) when reporting MOTA
"""

import numpy as np
from scipy.optimize import linear_sum_assignment 

def soMetricIoU(boxa, boxb):
    x = boxa[0]-boxb[0]
    y = boxa[1]-boxb[1]
    cb, sb = np.cos(boxb[2]), np.sin(boxb[2])
    x,y = (x*cb+y*sb, -x*sb+y*cb)
    c = np.cos(boxa[2]-boxb[2])
    s = np.sin(boxa[2]-boxb[2])
    la,wa = boxa[3:5]
    lb,wb = boxb[3:5]
    # gating
    if np.hypot(x,y) > np.hypot(la,wa) + np.hypot(lb,wb):
        return 0.
    # simple cases where orientations are about the same
    if abs(s) < 1e-3:
        area = (min(lb, x+la)-max(-lb, x-la))*(min(wb, y+wa)-max(-wb, y-wa))
        return area / (la*wa*4 + lb*wb*4 - area)
    if abs(c) < 1e-3:
        area = (min(lb, x+wa)-max(-lb, x-wa))*(min(wb, y+la)-max(-wb, y-la))
        return area / (la*wa*4 + lb*wb*4 - area)
    # calculate all corners and line intersections
    pts = np.array(((-lb,-wb),(-lb,wb),(lb,wb),(lb,-wb), # corners of boxb
                   ((c*(-y-wb) + wa)/s + x, -wb), # -w boxa, -w boxb
                   ((c*(-y-wb) - wa)/s + x, -wb), # +w boxa, -w boxb
                   ((s*(y+wb) + la)/c + x, -wb), # +l boxa, -w boxb
                   ((s*(y+wb) - la)/c + x, -wb), # -l boxa, -w boxb
                   ((c*(-y+wb) + wa)/s + x, wb), # -w boxa, +w boxb
                   ((c*(-y+wb) - wa)/s + x, wb), # +w boxa, +w boxb
                   ((s*(y-wb) + la)/c + x, wb), # +l boxa, +w boxb
                   ((s*(y-wb) - la)/c + x, wb), # -l boxa, +w boxb
                   (lb, (s*(lb-x) + wa)/c + y), # -w boxa, +l boxb
                   (lb, (s*(lb-x) - wa)/c + y), # +w boxa, +l boxb
                   (lb, (c*(x-lb) + la)/s + y), # +l boxa, +l boxb
                   (lb, (c*(x-lb) - la)/s + y), # -l boxa, +l boxb
                   (-lb, (-s*(lb+x) + wa)/c + y), # -w boxa, -l boxb
                   (-lb, (-s*(lb+x) - wa)/c + y), # +w boxa, -l boxb
                   (-lb, (c*(x+lb) + la)/s + y), # +l boxa, -l boxb
                   (-lb, (c*(x+lb) - la)/s + y), # -l boxa, -l boxb
                   (x-la*c+wa*s, y-la*s-wa*c), # -l,-w boxa
                   (x-la*c-wa*s, y-la*s+wa*c),  # -l,w boxa
                   (x+la*c-wa*s, y+la*s+wa*c), # l,w boxa
                   (x+la*c+wa*s, y+la*s-wa*c), # l,-w boxa
                  ))
    # determine which corners/intersections are within both boxes
    ptsin = abs(pts[:,0]) < lb+1e-6
    ptsin &= abs(pts[:,1]) < wb+1e-6
    ptsin &= abs(c*(pts[:,0]-x)+s*(pts[:,1]-y)) < la+1e-6
    ptsin &= abs(-s*(pts[:,0]-x)+c*(pts[:,1]-y)) < wa+1e-6
    if not np.any(ptsin): return 0.
    assert np.sum(ptsin) > 2
    # sort points, calculate area of convex polygon
    pts = pts[ptsin]
    centerpt = np.mean(pts, axis=0)
    ptsangle = np.arctan2(pts[:,1]-centerpt[1],pts[:,0]-centerpt[0])
    pts = pts[np.argsort(ptsangle)]
    # http://stackoverflow.com/questions/24467972/
    #    calculate-area-of-polygon-given-x-y-coordinates
    area = 0.5*np.abs(np.dot(pts[:,0],np.roll(pts[:,1],1)) -
                      np.dot(pts[:,1],np.roll(pts[:,0],1)))
    return area / (la*wa*4 + lb*wb*4 - area)


overlapres = 10
overlapbox = np.mgrid[:float(overlapres), :float(overlapres)]
overlapbox += .5
overlapbox *= 2./overlapres
overlapbox -= 1
overlapbox = overlapbox.transpose((1,2,0))
def soMetricIoUApprox(boxa, boxb):
    """
    much simpler point-based approximation of 2D rectangle overlap
    also fast @ low resolution
    """
    relx = boxa[0]-boxb[0]
    rely = boxa[1]-boxb[1]
    ca, sa = np.cos(boxa[2]), np.sin(boxa[2])
    cb, sb = np.cos(boxb[2]), np.sin(boxb[2])
    la,wa = boxa[3:5]
    lb,wb = boxb[3:5]
    R = np.array([[la/lb*(ca*cb+sa*sb), wa/lb*(ca*sb-cb*sa)],
                  [la/wb*(cb*sa-ca*sb), wa/wb*(ca*cb+sa*sb)]])
    t = np.array([(cb*relx + sb*rely)/lb, (cb*rely - sb*relx)/wb])
    grid = np.einsum(R, [0,1], overlapbox, [2,3,1], [2,3,0]) + t
    intersection = np.sum(np.all(abs(grid) < 1, axis=2))
    ioa = float(intersection) / overlapres**2
    return ioa / (1 - ioa + lb*wb/la/wa)

class MetricMine():
    """
    for each timestep, match annotations & estimates and store scores of tps/fps
    returns precision/recall curves + ID switches, as calculated by CLEAR
    """
    def __init__(self):
        self.dets = []
        self.nmissed = 0
        self.previousids = {}
        self.newscene = True
    def newScene(self):
        self.newscene = True
    def okMetric(self, boxa, boxb):
        return soMetricIoU(boxa, boxb) > .3
    def goodMetric(self, boxa, boxb):
        return soMetricIoU(boxa, boxb) > .7
    def add(self, gt, gtscored, gtdifficulty, gtids, ests, scores, estids):
        ngt = gt.shape[0]
        assert gtscored.shape[0] == ngt
        assert gtdifficulty.shape[0] == ngt
        nests = ests.shape[0]
        assert scores.shape[0] == nests
        gtscored = gtscored & (gtdifficulty < 3)
        matches = np.zeros((ngt, nests), dtype=bool)
        for gtidx, estidx in np.ndindex(ngt, nests):
            matches[gtidx, estidx] = self.okMetric(gt[gtidx], ests[estidx])
        matchscores = np.zeros((ngt, nests))
        matchscores[matches] = -1
        matchscores *= scores
        matchscores[gtscored, :] *= 2 # want scored boxes to match more
        rowpairs, colpairs = linear_sum_assignment(matchscores)
        gtmisses = np.ones(ngt, dtype=bool)
        estmisses = np.ones(nests, dtype=bool)
        currentids = {}
        for gtidx, estidx in zip(rowpairs, colpairs):
            if matches[gtidx, estidx]:
                gtmisses[gtidx] = False
                estmisses[estidx] = False
                if gtscored[gtidx]:
                    goodmatch = self.goodMetric(gt[gtidx], ests[estidx])
                    switch = (not self.newscene and 
                              gtids[gtidx] in self.previousids and
                              self.previousids[gtids[gtidx]] != estids[estidx])
                    self.dets.append((scores[estidx], True, goodmatch, switch))
                    currentids[gtids[gtidx]] = estids[estidx]
        for gtidx in range(ngt):
            if gtmisses[gtidx] and gtscored[gtidx]:
                self.nmissed += 1
        for estidx in range(nests):
            if estmisses[estidx]:
                self.dets.append((scores[estidx], False, False, False))
        self.previousids = currentids
        self.newscene = False
    def calc(self):
        dets = np.array(sorted(self.dets)[::-1])
        ndets = len(dets)
        nt = sum(dets[:,1]) + self.nmissed
        tps = np.cumsum(dets[:,1])
        checkpts = np.append(np.where(np.diff(dets[:,0]))[0], ndets-1)
        rec = tps[checkpts] / nt
        prec = tps[checkpts] / (checkpts+1)
        goodtpr = (np.cumsum(dets[:,2]))[checkpts] / nt
        switches = np.cumsum(dets[:,3])[checkpts]
        rec = np.concatenate(([0.], rec, [rec[-1]]))
        prec = np.concatenate(([1.], prec, [0.]))
        goodtpr = np.concatenate(([0.], goodtpr, [goodtpr[-1]]))
        switches = np.concatenate(([switches[0]], switches, [switches[-1]]))
        return np.array((rec, prec, goodtpr, switches)).T
    def calcMOTA(self):
        dets = np.array(sorted(self.dets)[::-1])
        ndets = len(dets)
        nt = sum(dets[:,1]) + self.nmissed
        tps = np.cumsum(dets[:,1])
        checkpts = np.append(np.where(np.diff(dets[:,0]))[0], ndets-1)
        switches = np.cumsum(dets[:,3])[checkpts]
        mota = (2*tps[checkpts] - checkpts-1 - switches) / float(nt)
        return max(mota)
    
    
class MetricMineApprox():
    """
    like metricMine but uses greedy assignment instead of optimal assignment
    basically the same result, nearly twice as fast
    """
    def __init__(self):
        self.dets = []
        self.switchscores = []
        self.nmissed = 0
        self.previousids = {}
        self.previousscores = {}
        self.newscene = True
    def newScene(self):
        self.previousids = {}
        self.previousscores = {}
    def okMetric(self, boxa, boxb):
        return soMetricIoUApprox(boxa, boxb) > .3
    def goodMetric(self, boxa, boxb):
        return soMetricIoUApprox(boxa, boxb) > .7
    def add(self, gt, gtscored, gtdifficulty, gtids, ests, scores, estids):
        ngt = gt.shape[0]
        assert gtscored.shape[0] == ngt
        assert gtdifficulty.shape[0] == ngt
        nests = ests.shape[0]
        assert scores.shape[0] == nests
        gtscored = gtscored & (gtdifficulty < 3)
        estorder = np.argsort(scores)[::-1]
        gtopen = np.ones(ngt, dtype=bool)
        currentids = {}
        currentscores = {}
        for estidx in estorder:
            bestgtGood = False
            bestgtScored = False
            bestgtidx = None
            for gtidx in range(ngt):
                if gtopen[gtidx] and self.okMetric(gt[gtidx], ests[estidx]):
                    keep = False
                    swap = bestgtidx is None
                    goodfit = self.goodMetric(gt[gtidx], ests[estidx])
                    isscored = gtscored[gtidx]
                    if not swap:
                        keep = bestgtGood and not goodfit
                        swap = bestgtGood and goodfit
                    if not keep and not swap:
                        swap = not bestgtScored and isscored
                    if swap:
                        bestgtidx = gtidx
                        bestgtGood = goodfit
                        bestgtScored = isscored
            if bestgtidx is None:
                self.dets.append((scores[estidx], False, False))
            else:
                gtopen[bestgtidx] = False
            if bestgtScored:
                self.dets.append((scores[estidx], True, bestgtGood))
                # search for id swap
                gtid = gtids[bestgtidx]
                switch = (gtid in self.previousids and
                          self.previousids[gtid] != estids[estidx])
                if switch:
                    switchscore = min(self.previousscores[gtid], scores[estidx])
                    self.switchscores.append(switchscore)
                currentids[gtid] = estids[estidx]
                currentscores[gtid] = scores[estidx]
        self.nmissed += sum(gtopen & gtscored)
        self.previousids = currentids
        self.previousscores = currentscores
    def calc(self):
        dets = np.array(sorted(self.dets)[::-1])
        switchscores = -np.array(sorted(self.switchscores)[::-1])
        ndets = len(dets)
        nt = sum(dets[:,1]) + self.nmissed
        tps = np.cumsum(dets[:,1])
        checkpts = np.append(np.where(np.diff(dets[:,0]))[0], ndets-1)
        rec = tps[checkpts] / nt
        prec = tps[checkpts] / (checkpts+1)
        goodtpr = (np.cumsum(dets[:,2]))[checkpts] / nt
        switches = np.searchsorted(switchscores, -dets[checkpts,0])
        rec = np.concatenate(([0.], rec, [rec[-1]]))
        prec = np.concatenate(([1.], prec, [0.]))
        goodtpr = np.concatenate(([0.], goodtpr, [goodtpr[-1]]))
        switches = np.concatenate(([switches[0]], switches, [switches[-1]]))
        return np.array((rec, prec, goodtpr, switches)).T
    def calcMOTA(self):
        dets = np.array(sorted(self.dets)[::-1])
        switchscores = -np.array(sorted(self.switchscores)[::-1])
        ndets = len(dets)
        nt = sum(dets[:,1]) + self.nmissed
        tps = np.cumsum(dets[:,1])
        checkpts = np.append(np.where(np.diff(dets[:,0]))[0], ndets-1)
        switches = np.searchsorted(switchscores, -dets[checkpts,0])
        mota = (2*tps[checkpts] - checkpts-1 - switches) / float(nt)
        return max(mota)



if __name__ == '__main__':
    """
        evaluate multiple estimates, plot together
        formatForKittiScore gets rid of things kitti didn't annotate
    """
    import matplotlib.pyplot as plt
    from calibs import calib_extrinsics, calib_projections, imgshapes
    from runconfigs.example import scenes, gt_files, ground_files
    from kittiGT import readGroundTruthFileTracking, formatForKittiScoreTracking
    
    nframesahead = 0
    estfiles = '{:s}/{:02d}f{:04d}.npy'
    tests = [('/home/m2/Data/kitti/estimates/gitresultPRC', 'pdeft w/ PRCNN', 'b')]
    
    results = []
    motas = []
    
    for testfolder, testname, testcolor in tests:
        metric = MetricMine()#Approx()
        
        for scene_idx, startfile, endfile, calib_idx in scenes:
            # run some performance metrics on numpy-stored results
            startfile += nframesahead
            calib_extrinsic = calib_extrinsics[calib_idx].copy()
            calib_projection = calib_projections[calib_idx]
            calib_projection = calib_projection.dot(np.linalg.inv(calib_extrinsic))
            imgshape = imgshapes[calib_idx]
            with open(gt_files.format(scene_idx), 'r') as fd: gtfilestr = fd.read()
            gt_all, gtdontcares = readGroundTruthFileTracking(gtfilestr,('Car','Van'))
            metric.newScene()
            
            for fileidx in range(startfile, endfile):
                ground = np.load(ground_files.format(scene_idx, fileidx))
                ground[:,:,3] -= 1.65 # TEMP!!!
                
                ests = np.load(estfiles.format(testfolder, scene_idx, fileidx))
                estids = ests[:,6].astype(int)
                scores = ests[:,5]
                ests = ests[:,:5]
                rede = formatForKittiScoreTracking(ests, estids, scores, fileidx,
                                                   ground, calib_projection, imgshape,
                                                   gtdontcares, kitti_format=False)
                ests = np.array([redd[0] for redd in rede])
                scores = np.array([redd[2] for redd in rede])
                estids = np.array([redd[1] for redd in rede])
                
                gthere = gt_all[fileidx]
                gtboxes = np.array([gtobj['box'] for gtobj in gthere])
                gtscores = np.array([gtobj['scored'] for gtobj in gthere],dtype=bool)
                gtdifficulty = np.array([gtobj['difficulty'] for gtobj in gthere],
                                        dtype=int)
                gtids = np.array([gtobj['id'] for gtobj in gthere],dtype=int)
                gtdontcareshere = gtdontcares[fileidx]
                
                metric.add(gtboxes, gtscores, gtdifficulty, gtids,
                           ests, scores, estids)
        restest = metric.calc()
        results.append((testname, restest, testcolor))
        motas.append(metric.calcMOTA())

    print("MOTAs")
    print(motas)

    fig, axeses = plt.subplots(1, 3, figsize=(12., 3.))
    plt1, plt2, plt3 = axeses.flat
    plt1.set_xlim((.5, 1.))
    plt2.set_xlim((.5, 1.))
    plt3.set_xlim((.5, 1.))
    plt1.set_ylim((.5, 1.))
    plt2.set_ylim((0., 1.))
    plt1.set_title('Precision vs Recall')
    plt2.set_title('Close fit recall vs Recall')
    plt3.set_title('# identity swaps vs Recall')
    maxswaps = int(max(np.max(result[1][:,3]) for result in results))+1
    plt3.set_yticks(list(range(0, maxswaps, maxswaps//5+1)))
    for testname, result, color in results:
        plt1.plot(result[:,0], result[:,1], color, label=testname)
        plt2.plot(result[:,0], result[:,2], color, label=testname)
        plt3.plot(result[:,0], result[:,3], color, label=testname)
    plt3.legend(bbox_to_anchor = (1.04, 1), loc="upper left")
    plt.show()