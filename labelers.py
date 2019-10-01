# -*- coding: utf-8 -*-
import numpy as np


class SingleHypLabeler:
    """
    assigns unique labels for single-hypothesis tracking
    using standard approach (first seen detection)
    this is slightly different, in that only reported objects get labelled
    so less labels all in all, if report cutoff is high
    """
    def __init__(self, nobjects, nlabels=0, timeback=0):
        # nlabels, timeback only for compatibility with HypHopLabeler
        self.idx2label = np.zeros((nobjects, ), dtype=int) - 1
        self.labelcount = 1
        
    def reset(self):
        self.idx2label[:] = -1
        self.labelcount = 1
        
    def add(self, pairs, reports):
        """ takes updatepairs (maps new object idx to old object idx)
            and new object idxs to report
        """
        haveoldidx = pairs[:,0] >= 0
        oldlabels = self.idx2label[pairs[haveoldidx,0]].copy()
        self.idx2label[haveoldidx] = oldlabels
        self.idx2label[haveoldidx==False] = -1
        newlabelidxs = self.idx2label[reports] == -1
        nnewlabels = sum(newlabelidxs)
        self.idx2label[reports[newlabelidxs]] = range(self.labelcount,
                                                      self.labelcount+nnewlabels)
        self.labelcount += nnewlabels
        return self.idx2label[reports]


class HypHopLabeler:
    """
    Made to address the "hypothesis hopping" issue for MHT, in which
    the labels from the previous best hypothesis don't correspond to the labels from
    the current best one.
    
    Labels are typically obtained from the first timestep & detection of an object.
    Hopping essentially comes from disagreement on these.
    
    There doesn't seem to be an efficient accurate solution (see Lingji Chen's paper)
    but here is a heuristic one.
    -- uses msmt idxs as matching criterion
    -- searches back a certain number of timesteps
    -- consistent, b.c. only most recent sighting of label is used for matching
    
    could make this faster by wrapping pairslog rather than shifting it left
    """
    def __init__(self, nobjects, nlabels, timeback):
        # [[label, last seen timestep, msmt idx at that timestep],...]
        self.labellog = np.zeros((nlabels, 3),dtype=int) - 1
        # copies of updatepairs (object idx, msmt idx) from the tracker
        self.pairslog = np.zeros((timeback, nobjects, 2),dtype=int) - 1
        self.labelcount = 1
        
    def reset(self):
        self.labellog [:] = -1
        self.pairslog[:] = -1
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