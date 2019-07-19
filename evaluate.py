#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
last mod 6/4/19
"""

import numpy as np
from scipy.optimize import linear_sum_assignment 
from sklearn.metrics import average_precision_score
import motmetrics

overlapres = 50
overlapbox = np.mgrid[:float(overlapres), :float(overlapres)]
overlapbox += .5
overlapbox *= 2./overlapres
overlapbox -= 1
overlapbox = overlapbox.transpose((1,2,0))
def soMetricIoU(boxa, boxb, cutoff = .7):
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
    iou = ioa / (1 - ioa + lb*wb/la/wa)
    return cutoff - iou
    

def soMetricEuc(boxa, boxb):
    closeness = np.hypot(*(boxa[:2]-boxb[:2])) * .4
    closeness += ((boxa[2]-boxb[2]+np.pi)%(2*np.pi)-np.pi) * 1.
    closeness += np.hypot(*(boxa[3:]-boxb[3:5])) * .3
    return closeness - 1

class MetricAvgPrec():
    """
    builds histogram of scores for true and false samples
    calculates average precision based on histogram (slight approximation)
    """
    def __init__(self, resolution=391, soMetric=soMetricIoU):
        #self.cutoffs = np.append(np.linspace(-29, 10, resolution-1), [1000.])
        self.cutoffs = np.append(np.linspace(-4, 9, resolution-1), [1000.])
        self.counts = np.zeros((resolution, 5), dtype=int)
        self.nmissed = np.zeros(4, dtype=int)
        self.soMetric = soMetric
        
        self.faketru = np.zeros(resolution*2 + 1, dtype=bool)
        self.faketru[:resolution] = True
        self.faketru[-1] = True
        self.fakeest = np.concatenate((np.arange(resolution),
                                       np.arange(resolution), [-1.]))
    
    def add(self, gt, gtscored, gtdifficulty, ests, scores):
        ngt = gt.shape[0]
        assert gtscored.shape[0] == ngt
        assert gtdifficulty.shape[0] == ngt
        nests = ests.shape[0]
        assert scores.shape[0] == nests
        matches = np.zeros((ngt, nests))
        for gtidx, estidx in np.ndindex(ngt, nests):
            score = self.soMetric(gt[gtidx], ests[estidx])
            matches[gtidx, estidx] = min(score, 0)
        matchesnonmiss = matches < 0
        matches[:] = 0
        matches[matchesnonmiss] = -1
        matches *= scores
        rowpairs, colpairs = linear_sum_assignment(matches)
        gtmisses = np.ones(ngt, dtype=bool)
        estmisses = np.ones(nests, dtype=bool)
        for gtidx, estidx in zip(rowpairs, colpairs):
            if matchesnonmiss[gtidx, estidx]:
                gtmisses[gtidx] = False
                estmisses[estidx] = False
                gtdiffidx = gtdifficulty[gtidx] if gtscored[gtidx] else 3
                scoreidx = np.searchsorted(self.cutoffs, scores[estidx])
                self.counts[scoreidx, gtdiffidx:4] += 1
        for gtidx in range(ngt):
            if gtmisses[gtidx]:
                gtdiffidx = gtdifficulty[gtidx] if gtscored[gtidx] else 3
                self.nmissed[gtdiffidx:] += 1
        for estidx in range(nests):
            if estmisses[estidx]:
                scoreidx = np.searchsorted(self.cutoffs, scores[estidx])
                self.counts[scoreidx, 4] += 1
        
    def calc(self):
        avgprec = np.zeros(4)
        for difficulty in range(4):
            fakeweights = np.concatenate((self.counts[:,difficulty],
                                          self.counts[:,4],
                                          [self.nmissed[difficulty]]))
            fakeweights = np.maximum(fakeweights, 1e-8)
            avgprec[difficulty] = average_precision_score(self.faketru,
                                   self.fakeest, sample_weight = fakeweights)
        return avgprec
    
    def calcKitti(self):
        nthings = 11
        avgprecs = np.zeros(4)
        for difficulty in range(4):
            totalhitcount = np.sum(self.counts[:,difficulty])
            RR = np.cumsum(self.counts[:,difficulty])
            RR = totalhitcount - RR + self.counts[:,difficulty]
            totalcount = totalhitcount + self.nmissed[difficulty]
            recallsteps = np.linspace(0, totalcount, nthings, endpoint=False)
            recallsteps = recallsteps[recallsteps < totalhitcount]
            steps = np.searchsorted(RR[::-1], recallsteps[::-1])[::-1]
            area = 0.
            for step in steps:
                tp = float(RR[step])
                area += tp / (tp + np.sum(self.counts[step:,4]) + 1e-8)
            avgprecs[difficulty] = area / nthings
                
    

class MetricPrecRec():
    def __init__(self, cutoff = .5, soMetric = soMetricEuc):
        self.cutoff = cutoff
        self.tp = np.zeros(4, dtype=int)
        self.t = np.zeros(4, dtype=int)
        self.p = 0
        self.soMetric = soMetric
        
    def add(self, gt, gtscored, gtdifficulty, ests, scores):
        for gtidx in range(gt.shape[0]):
            gtbox = gt[gtidx]
            difficultyidx = gtdifficulty[gtidx] if gtscored[gtidx] else 3
            self.t[difficultyidx:] += 1
            matches = False
            for estidx, est in enumerate(ests):
                if scores[estidx] > self.cutoff and self.soMetric(gtbox, est)<0:
                    assert not matches, "object had 2 estimates that match"
                    self.tp[difficultyidx:] += 1
                    matches = True
        self.p += np.sum(scores > self.cutoff)
        
    def calc(self):
        tp = self.tp.astype(float)
        return np.append(tp/self.t, tp[-1:]/self.p)


class MetricMOT():
    def __init__(self, cutoff = .45, soMetric = soMetricEuc):
        self.accumulator = None#motmetrics.MOTAccumulator(auto_id=True)
        self.cutoff = cutoff
        self.soMetric = soMetric
        self.sooptions = {}
        self.accumulators = []
        self.metricnames = ['mota','motp','num_switches','mostly_tracked',
                            'mostly_lost','num_unique_objects']
    def newScene(self):
        if not self.accumulator is None:
            self.accumulators.append(self.accumulator)
        self.accumulator = motmetrics.MOTAccumulator(auto_id=True)
    def add(self, gt, gtscored, gtdifficulty, gtids, ests, scores, estids):
        ests = ests[scores > self.cutoff]
        estids = estids[scores > self.cutoff]
        gt = gt[gtscored & (gtdifficulty < 3)]
        gtids = gtids[gtscored & (gtdifficulty < 3)]
        ngt = gt.shape[0]
        nests = ests.shape[0]
        matches = np.zeros((ngt, nests))
        for gtidx, estidx in np.ndindex(ngt, nests):
            score = self.soMetric(gt[gtidx], ests[estidx], **self.sooptions)
            matches[gtidx, estidx] = min(score, 0)
        matches[matches > 0] = np.nan
        matches += 1.
        self.accumulator.update(gtids, estids, matches)
    def calc(self):
        allaccs = self.accumulators + [self.accumulator]
        mh = motmetrics.metrics.create()
        summary = mh.compute_many(allaccs,metrics=self.metricnames,
                                  generate_overall=True)
        #summary = mh.compute(self.accumulator, metrics=
        #            ['mota','motp','num_switches','mostly_tracked','mostly_lost'],
        #            name = 'acc')
        return summary.loc['OVERALL']
    
    
#img = imread(img_files.format(file_idx))[:,:,::-1]
#
#gtboxes = []
#with open(gt_files.format(file_idx), 'r') as fd: gtstr = fd.read()
#gtstr = gtstr.split('\n')
#if gtstr[-1] == '': gtstr = gtstr[:-1]
#for gtrow in gtstr:
#    gtrow = gtrow.split(' ')
#    if gtrow[0] == 'DontCare' or gtrow[0]=='Misc': continue
#    gtboxes.append((float(gtrow[13]), -float(gtrow[11]),
#        1.5708-float(gtrow[14]), float(gtrow[10])/2, float(gtrow[9])/2))
#        
#estboxes = []
#with open(output_files.format(file_idx), 'r') as fd: gtstr = fd.read()
#gtstr = gtstr.split('\n')
#if gtstr[-1] == '': gtstr = gtstr[:-1]
#for gtrow in gtstr:
#    gtrow = gtrow.split(' ')
#    if gtrow[0] == 'DontCare' or gtrow[0]=='Misc': continue
#    estboxes.append((float(gtrow[13]), -float(gtrow[11]),
#        1.5708-float(gtrow[14]), float(gtrow[10])/2, float(gtrow[9])/2))
#    
#ngt = len(gtboxes)
#nest = len(estboxes)
#match_mtx = np.zeros((ngt, nest))
#for gtidx, gtbox in enumerate(gtboxes):
#    gtbox_uv = xy2uv(gtbox)
#    area_gt = gtbox[3]*gtbox[4]*4
#    for estidx, estbox in enumerate(estboxes):
#        estbox_uv = xy2uv(estbox)
#        area_est = estbox[3]*estbox[4]*4
#        iou = overlap(gtbox_uv, estbox_uv)
#        iou /= (area_gt + area_est - iou)
#        match_mtx[gtidx, estidx] = 1.-iou
#ff = assignment(match_mtx, .3)




def MAPfromfile(filename):
    with open(filename, 'r') as fd:
        all_results = fd.read()
    results = [[float(score) for score in result.split(' ') if score != '']
                    for result in all_results.split('\n')]
    print([sum(result[::4])/11. for result in results])
#MAPfromfile('../object/estimates/b/stats_car_detection_ground.txt')
    
    
    
if __name__ == '__main__':
    """
        runs a single accuracy metric across multiple scenes
        formatForKittiScore gets rid of things kitti didn't annotate
    """
    from calibs import calib_extrinsics, calib_projections, view_by_day
    from config import sceneranges
    from config import calib_map_training as calib_map
    from analyzeGT import readGroundTruthFileTracking, formatForKittiScoreTracking
    from imageio import imread
    
    scenes = [0,1,2,3,4,5,6,7,8,9]#[0,4,5]#
    gt_files = 'Data/tracking_gt/{:04d}.txt'
    estfiles = 'Data/estimates/trackingresults0/{:02d}f{:04d}.npy'
    img_files = 'Data/tracking_image/training/{:04d}/000000.png'
    ground_plane_files = 'Data/tracking_ground/training/{:02d}f{:06d}.npy'
    
    metric = MetricAvgPrec()
    metric2 = MetricMOT(soMetric = soMetricIoU)#
    #metric2.sooptions['cutoff'] = .5
    
    for scene_idx in scenes:
        # run some performance metrics on numpy-stored results
        startfile, endfile = sceneranges[scene_idx]
        calib_idx = calib_map[scene_idx]
        calib_extrinsic = calib_extrinsics[calib_idx].copy()
        calib_extrinsic[2,3] += 1.65
        view_angle = view_by_day[calib_idx]
        calib_projection = calib_projections[calib_idx]
        calib_projection = calib_projection.dot(np.linalg.inv(calib_extrinsic))
        imgshape = imread(img_files.format(scene_idx)).shape[:2]
        with open(gt_files.format(scene_idx), 'r') as fd: gtfilestr = fd.read()
        gt_all, gtdontcares = readGroundTruthFileTracking(gtfilestr, ('Car', 'Van'))
        metric2.newScene()
        
        for fileidx in range(startfile, endfile):
            ground = np.load(ground_plane_files.format(scene_idx, fileidx))
            
            ests = np.load(estfiles.format(scene_idx, fileidx))
            estids = ests[:,6].astype(int)
            scores = ests[:,5]
            ests = ests[:,:5]
            rede = formatForKittiScoreTracking(ests, estids, scores, fileidx,
                                ground, calib_projection, imgshape, gtdontcares)
            ests = np.array([redd[0] for redd in rede])
            scores = np.array([redd[2] for redd in rede])
            estids = np.array([redd[1] for redd in rede])
            
            gthere = gt_all[fileidx]
            gtboxes = np.array([gtobj['box'] for gtobj in gthere])
            gtscores = np.array([gtobj['scored'] for gtobj in gthere],dtype=bool)
            gtdifficulty = np.array([gtobj['difficulty'] for gtobj in gthere],dtype=int)
            gtids = np.array([gtobj['id'] for gtobj in gthere],dtype=int)
            gtdontcareshere = gtdontcares[fileidx]
            
            
            metric2.add(gtboxes, gtscores, gtdifficulty, gtids, ests, scores, estids)
            metric.add(gtboxes, gtscores, gtdifficulty, ests, scores)
        
    print(metric.calc())
    print(metric2.calc())
        
if False:
    from evaluate_tracking import trackingEvaluation, Mail
    mail = Mail("")
    e = trackingEvaluation(t_sha=
                    "/home/m2/Data/kitti/estimates/trackingresults0/kitti/",
                           mail=mail,cls="car")
    assert e.loadTracker()
    assert e.loadGroundtruth()
    assert len(e.groundtruth) == len(e.tracker)
    e.createEvalDir()
    e.compute3rdPartyMetrics()
    e.saveToStats()
    mail.finalize(True,"tracking",
                  "/home/m2/Data/kitti/estimates/trackingresults0/kitti/","")