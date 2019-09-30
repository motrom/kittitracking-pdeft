#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
last mod 8/10/19

made to address the "hypothesis hopping" issue, in which the labels from the 
previous best hypothesis don't correspond to the labels from the current best one

labels are typically obtained from the first timestep & detection of an object
hopping comes from disagreement on these

There doesn't seem to be an accepted fast solution (see Lingji Chen's paper)
but here is a heuristic one. To minimize id switches with the report from the last
timestep, find the lowest-cost assignment.


... in general the situation is worse than i thought, specifically there may be
cases where 'better' choice of fixing a bad labelling leads to two label switches,
(one to fix the mistake) while keep the bad labelling is just one
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

def assignment(costmatrix, maxcost):
    costmatrix = costmatrix - maxcost
    matchesallowed = costmatrix < maxcost
    costmatrix = np.maximum(costmatrix - maxcost, 0)
    rowpairs, colpairs = linear_sum_assignment(costmatrix)
    matchedpairs = matchesallowed[rowpairs,colpairs]
    rowpairs = rowpairs[matchedpairs]
    colpairs = colpairs[matchedpairs]
    rowmiss = np.ones(costmatrix.shape[0], dtype=bool)
    rowmiss[rowpairs] = False
    rowmiss = np.where(rowmiss)[0]
    colmiss = np.ones(costmatrix.shape[1], dtype=bool)
    colmiss[colpairs] = False
    colmiss = np.where(colmiss)[0]
    return rowpairs, colpairs, rowmiss, colmiss

"""
matching based on similarity of current hypothesis and previous best hyp,
based on object estimates in the previous timestep
consistent, b.c. assignment between label sets is used
not completely trivial to extend further back in time with consistency
(probably need to enforce that only the most recent label sighting is considered)
"""
def swapLabelsForHopping(thisreports, prevobjsthishyp, prevreports, labelcount):
    labels = thisreports[:,6]
    reportmatchmtx = np.hypot(prevobjsthishyp[:,None,0]-prevreports[None,:,0],
                              prevobjsthishyp[:,None,1]-prevreports[None,:,1])
    thismatch, prevmatch, thismiss, prevmiss = assignment(reportmatchmtx, 1.5)
    labels[thismatch] = prevreports[prevmatch,6]
    labels[thismiss] = np.arange(labelcount, labelcount+thismiss.shape[0])
    prevprevreports = prevreports[prevmiss].copy() # keep around for next time
    return labelcount+thismiss.shape[0]


"""
this version looks further back in time
but only uses msmt idxs as matching criterion
consistent, b.c. only most recent sighting of label is used for matching
could make this faster by wrapping pairslog rather than shifting it left
"""
class HypHopLabeler:
    def __init__(self, nobjects, nlabels, timeback):
        # [[label, last seen timestep, msmt idx at that timestep],...]
        self.labellog = np.zeros((nlabels, 3),dtype=int) - 1
        # copies of updatepairs (object idx, msmt idx) from the tracker
        self.pairslog = np.zeros((timeback, nobjects, 2),dtype=int) - 1
        self.labelcount = 1
        
    def add(self, pairs, reports):
        # for each report, search for label that matches timestep+msmtidx
        labelidxs = np.zeros(len(reports), dtype=int) - 1
        for reportidx, report in enumerate(reports):
            report = pairs[report,0]
            if report == -1: continue
            for timeback in range(self.pairslog.shape[0]-1,-1,-1):
                report, msmtidx = self.pairslog[timeback, report]
                if msmtidx != -1:
                    labelmatches = ((self.labellog[:,1]==timeback) &
                                    (self.labellog[:,2]==msmtidx))
                    if np.any(labelmatches):
                        labelidxs[reportidx] = np.where(labelmatches)[0][0]
                        break
                if report == -1:
                    break
        # unassigned reports get new labels
        neednewlabels = labelidxs == -1
        nnewlabels = sum(neednewlabels)
        availablelabelidxs = self.labellog[:,1]<0
        assert nnewlabels <= sum(availablelabelidxs)
        newlabelidxs = np.where(availablelabelidxs)[0][:nnewlabels]
        labelidxs[neednewlabels] = newlabelidxs
        assert np.all(np.diff(np.sort(labelidxs))>0) # current labels unique
        self.labellog[newlabelidxs,0] = np.arange(nnewlabels)+self.labelcount
        self.labelcount += nnewlabels
        # update logs
        for reportidx, report in enumerate(reports):
            if pairs[report, 1] != -1:
                self.labellog[labelidxs[reportidx], 1] = self.pairslog.shape[0]
                self.labellog[labelidxs[reportidx], 2] = pairs[report, 1]
        self.labellog[:,1] -= 1 # move forward 1 timestep
        self.pairslog[:-1] = self.pairslog[1:]
        self.pairslog[-1] = pairs
        # get current labels
        return self.labellog[labelidxs,0].copy()